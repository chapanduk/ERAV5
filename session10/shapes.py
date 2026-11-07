"""Q1: every tensor shape in one real training step, each with one line on what its dimensions mean."""
import torch, torch.nn.functional as F
from common import Cfg, GPT, ShapeLog, load_corpus, get_batch, pick_device

torch.manual_seed(0)
dev = pick_device()
c = Cfg()
B, T = 32, c.ctx
model = GPT(c).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
train, _ = load_corpus()
x, y = get_batch(train, B, T, torch.Generator().manual_seed(0), dev)

log = ShapeLog()
print(f"config: B={B} T={T} D={c.d} H={c.n_head} hd={c.d // c.n_head} L={c.n_layer} V={c.vocab}  device={dev}\n")
print("=== forward ===")
logits = model(x, log=log)
log("targets y", y, "B, T: the byte that actually comes next at every position (x shifted left by one)")
flat_logits = logits.view(-1, c.vocab)
log("logits.view(-1, V)", flat_logits, "B*T, V: every position flattened into one long list of next-byte classifications")
logp = F.log_softmax(flat_logits, -1)
log("log_softmax(logits)", logp, "B*T, V: log-probability of every possible next byte")
tok_loss = F.cross_entropy(flat_logits, y.view(-1), reduction="none")
log("per-token loss", tok_loss, "B*T: -log p(correct byte) for each position")
loss = tok_loss.mean()
log("loss = mean", loss, "(): one scalar, the average over all B*T tokens; backward() starts here")

print("=== backward ===")
loss.backward()
for name, p in [("tok.weight", model.tok.weight), ("pos.weight", model.pos.weight),
                ("blocks.0.qkv.weight", model.blocks[0].qkv.weight), ("blocks.0.qkv.bias", model.blocks[0].qkv.bias),
                ("blocks.0.fc.weight", model.blocks[0].fc.weight), ("blocks.0.out.weight", model.blocks[0].out.weight),
                ("blocks.0.ln1.weight", model.blocks[0].ln1.weight), ("head.weight", model.head.weight)]:
    log(f"{name}.grad", p.grad, {
        "tok.weight": "V, D: one row per byte id; rows for bytes absent from the batch get exactly zero gradient",
        "pos.weight": "ctx, D: one row per position",
        "blocks.0.qkv.weight": "3D, D: nn.Linear stores (out_features, in_features); grad always matches its parameter",
        "blocks.0.qkv.bias": "3D: one bias per output feature",
        "blocks.0.fc.weight": "4D, D: MLP up-projection, (out, in)",
        "blocks.0.out.weight": "D, 4D: MLP down-projection, (out, in)",
        "blocks.0.ln1.weight": "D: per-feature LayerNorm gain",
        "head.weight": "V, D: one row of D weights per output byte",
    }[name])
gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
log("clip_grad_norm_ result", gn, "(): L2 norm of all gradients concatenated into one vector, before clipping")

print("=== optimizer step ===")
opt.step()
st = opt.state[model.blocks[0].fc.weight]
log("adam exp_avg (fc.weight)", st["exp_avg"], "4D, D: running mean of the gradient (first moment), same shape as the parameter")
log("adam exp_avg_sq (fc.weight)", st["exp_avg_sq"], "4D, D: running mean of grad^2 (second moment), same shape again")
log("adam step", st["step"], "(): step counter used for bias correction")
log.dump()

n = sum(p.numel() for p in model.parameters())
print(f"\nparams: {n:,} total, {model.n_matmul_params():,} in matmuls. "
      f"Optimizer state = 2 x params = {2 * n:,} floats.")
