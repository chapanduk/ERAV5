"""Q3: break gradient accumulation on purpose.

Each optimizer step = 4 micro-batches of 16 rows, but every micro-batch has its own sequence length
drawn from {16, 32, 64, 128, 256}, so their token counts differ by up to 16x.

  correct : sum of per-token losses over ALL micro-batches / total tokens   (== one big masked batch)
  wrong   : mean over micro-batches of (that micro-batch's mean loss)       ("average of the averages")

Two models, identical init, identical data stream. Only the loss reduction differs.
"""
import json, sys, time
import torch, torch.nn.functional as F
from common import Cfg, GPT, load_corpus, get_batch, pick_device

dev = pick_device()
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 800
SEEDS = [0, 1]
N_MICRO, ROWS, LENS = 4, 16, [16, 32, 64, 128, 256]
c = Cfg(d=256, n_layer=4, n_head=4, ctx=256)
train, val = load_corpus()

vg = torch.Generator().manual_seed(1234)
VAL = [get_batch(val, 32, 256, vg, dev) for _ in range(8)]


def step_data(seed, step):
    g = torch.Generator().manual_seed(seed * 1_000_003 + step)
    lens = [LENS[i] for i in torch.randint(len(LENS), (N_MICRO,), generator=g)]
    return [get_batch(train, ROWS, L, g, dev) for L in lens]


def grads(model, micro, mode):
    """Accumulate grads over micro-batches. Returns (true token-weighted loss, loss this mode would log)."""
    model.zero_grad(set_to_none=True)
    total = sum(y.numel() for _, y in micro)
    tok_sum, reported = 0.0, 0.0
    for x, y in micro:
        lsum = F.cross_entropy(model(x).view(-1, c.vocab), y.view(-1), reduction="sum")
        if mode == "correct":
            (lsum / total).backward()
            reported += lsum.item() / total
        else:
            lmean = lsum / y.numel()
            (lmean / len(micro)).backward()
            reported += lmean.item() / len(micro)
        tok_sum += lsum.item()
    return tok_sum / total, reported


def padded_single_batch_grad(model, micro):
    """Reference: all micro-batches right-padded to 256 and run as ONE batch with a masked mean."""
    xs, ys = [], []
    for x, y in micro:
        pad = 256 - x.shape[1]
        xs.append(F.pad(x, (0, pad), value=0))
        ys.append(F.pad(y, (0, pad), value=-100))   # -100 = ignore_index
    x, y = torch.cat(xs), torch.cat(ys)
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(x).view(-1, c.vocab), y.view(-1), ignore_index=-100).backward()
    return torch.cat([p.grad.flatten() for p in model.parameters()]).clone()


def flat_grad(model):
    return torch.cat([p.grad.flatten() for p in model.parameters()]).clone()


@torch.no_grad()
def evaluate(model):
    per_pos = torch.zeros(256, device=dev)
    for x, y in VAL:
        l = F.cross_entropy(model(x).view(-1, c.vocab), y.view(-1), reduction="none").view(x.shape)
        per_pos += l.mean(0)
    per_pos /= len(VAL)
    return per_pos.mean().item(), per_pos.cpu().tolist()


def grad_comparison(model, micro):
    ref = padded_single_batch_grad(model, micro)
    grads(model, micro, "correct"); gc = flat_grad(model)
    grads(model, micro, "wrong"); gw = flat_grad(model)
    rel = lambda a, b: ((a - b).norm() / b.norm()).item()
    cos = lambda a, b: F.cosine_similarity(a, b, dim=0).item()
    return dict(correct_vs_ref_rel=rel(gc, ref), wrong_vs_ref_rel=rel(gw, ref), wrong_vs_ref_cos=cos(gw, ref))


results = {}
for seed in SEEDS:
    torch.manual_seed(seed)
    init = GPT(c).state_dict()
    runs = {}
    for mode in ["correct", "wrong"]:
        model = GPT(c).to(dev)
        model.load_state_dict(init)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
        log = dict(step=[], true=[], reported=[], val_step=[], val=[], gradcmp=[])
        t0 = time.time()
        for s in range(STEPS + 1):
            micro = step_data(seed, s)
            if mode == "correct" and s % 200 == 0:
                gcmp = grad_comparison(model, micro); gcmp["step"] = s
                log["gradcmp"].append(gcmp)
                print(f"  [seed {seed}] grad check @ step {s}: {gcmp}", flush=True)
            if s % 25 == 0:
                v, _ = evaluate(model)
                log["val_step"].append(s); log["val"].append(v)
            if s == STEPS:
                break
            true, rep = grads(model, micro, mode)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            log["step"].append(s); log["true"].append(true); log["reported"].append(rep)
        _, log["per_pos"] = evaluate(model)
        print(f"seed {seed} {mode:<7} final val {log['val'][-1]:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        runs[mode] = log
    results[seed] = runs

json.dump(results, open("accum_results.json", "w"))
