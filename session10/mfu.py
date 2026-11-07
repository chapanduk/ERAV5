"""Q5: MFU, plus ablations that test what is costing the distance to 40%."""
import time, torch, torch.nn.functional as F
from common import Cfg, GPT, load_corpus, get_batch, flops_per_token

dev = torch.device("mps")
sync = torch.mps.synchronize
train, _ = load_corpus()

# Spec-sheet peak. Apple publishes no TFLOPS figure; derive it: 40 GPU cores x 128 FP32 lanes x 2 (FMA) x clock.
# The M4 Max GPU clock is not published either; ~1.58 GHz is the commonly measured figure -> ~16.2 TFLOPS.
SPEC = 40 * 128 * 2 * 1.578e9


def matmul_ceiling(dtype, n=4096, iters=30):
    a, b = torch.randn(n, n, device=dev, dtype=dtype), torch.randn(n, n, device=dev, dtype=dtype)
    for _ in range(5): a @ b
    sync(); t = time.perf_counter()
    for _ in range(iters): a @ b
    sync()
    return 2 * n ** 3 * iters / (time.perf_counter() - t)


def timed(c, B, steps=40, warm=10, sync_every_step=True, amp=None, compile_=False):
    torch.manual_seed(0)
    m = GPT(c).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
    run = torch.compile(m) if compile_ else m
    g = torch.Generator().manual_seed(0)
    batches = [get_batch(train, B, c.ctx, g, dev) for _ in range(8)]   # pre-staged: exclude host-side batching
    for i in range(warm + steps):
        if i == warm: sync(); t = time.perf_counter()
        x, y = batches[i % 8]
        with torch.autocast("mps", dtype=amp, enabled=amp is not None):
            loss = F.cross_entropy(run(x).view(-1, c.vocab).float(), y.view(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if sync_every_step: loss.item()
    sync()
    dt = (time.perf_counter() - t) / steps
    return flops_per_token(m, c.ctx) * B * c.ctx / dt, dt


print("== ceilings (4096^3 matmul) ==")
ceil = {}
for name, dt in [("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)]:
    ceil[name] = matmul_ceiling(dt)
    print(f"  {name}: {ceil[name] / 1e12:6.2f} TFLOPS")
print(f"  spec-derived fp32 peak: {SPEC / 1e12:.2f} TFLOPS\n")

base = Cfg()
import statistics, sys
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
cases = [
    ("baseline loop (10.9M, B=32, fp32, .item() every step)", base, 32, {}),
    ("same, no per-step .item() sync", base, 32, dict(sync_every_step=False)),
    ("B=8   (smaller matmuls)", base, 8, {}),
    ("B=128 (bigger matmuls)", base, 128, dict(steps=15, warm=5)),
    ("wider: D=1024, L=8 (101M), B=16", Cfg(d=1024, n_layer=8, n_head=16), 16, dict(steps=15, warm=5)),
    ("baseline + bf16 autocast", base, 32, dict(amp=torch.bfloat16)),
    ("baseline + torch.compile", base, 32, dict(compile_=True)),
]
res = {n: [] for n, *_ in cases}
for r in range(REPS):                       # interleaved, so background drift hits every case alike
    for name, c, B, kw in cases:
        try:
            res[name].append(timed(c, B, **kw))
        except Exception as e:
            print(f"{name} failed: {e}"); res[name] = None
    print(f"rep {r + 1}/{REPS} done", flush=True)
print(f"\n{'case':<56} {'ms/step med [min-max]':>24} {'TFLOPS med':>10} {'MFU vs spec med [range]':>26} {'vs fp32 matmul':>15}")
for name, *_ in cases:
    v = res[name]
    if not v: continue
    fl = [x[0] for x in v]; dt = [x[1] * 1e3 for x in v]
    m = statistics.median(fl)
    print(f"{name:<56} {statistics.median(dt):>7.1f} [{min(dt):.0f}-{max(dt):.0f}]{'':>6} {m / 1e12:>10.2f} "
          f"{m / SPEC:>11.1%} [{min(fl) / SPEC:.0%}-{max(fl) / SPEC:.0%}]{'':>3} {m / ceil['fp32']:>14.1%}")
