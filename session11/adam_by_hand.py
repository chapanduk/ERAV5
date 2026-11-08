"""Task 1: Adam by hand for one weight and five gradients, checked against torch.optim.Adam."""
import math, torch

w0, lr, b1, b2, eps = 0.5, 0.1, 0.9, 0.999, 1e-8
grads = [0.20, -0.10, 0.30, 0.05, -0.25]

print(f"w0 = {w0}, lr = {lr}, beta1 = {b1}, beta2 = {b2}, eps = {eps}, no weight decay\n")

# ---- by hand (plain Python floats = IEEE fp64)
hand, m, v, w = [], 0.0, 0.0, w0
for t, g in enumerate(grads, start=1):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    mh = m / (1 - b1 ** t)
    vh = v / (1 - b2 ** t)
    step = lr * mh / (math.sqrt(vh) + eps)
    w = w - step
    hand.append(dict(t=t, g=g, m=m, v=v, mh=mh, vh=vh, step=step, w=w))

# ---- PyTorch: same weight, gradients injected by hand each step
p = torch.nn.Parameter(torch.tensor([w0], dtype=torch.float64))
opt = torch.optim.Adam([p], lr=lr, betas=(b1, b2), eps=eps, foreach=False)
torch_rows = []
for t, g in enumerate(grads, start=1):
    before = p.item()
    p.grad = torch.tensor([g], dtype=torch.float64)
    opt.step()
    st = opt.state[p]
    m_t, v_t, step_t = st["exp_avg"].item(), st["exp_avg_sq"].item(), int(st["step"].item())
    torch_rows.append(dict(m=m_t, v=v_t, mh=m_t / (1 - b1 ** step_t), vh=v_t / (1 - b2 ** step_t),
                           step=before - p.item(), w=p.item()))

hdr = f"{'t':>2} {'g':>6} | {'quantity':<5} {'by hand':>22} {'PyTorch':>22} {'abs diff':>9}"
print(hdr); print("-" * len(hdr))
worst = 0.0
for h, tr in zip(hand, torch_rows):
    for k, label in [("m", "m"), ("v", "v"), ("mh", "m^"), ("vh", "v^"), ("step", "step"), ("w", "w")]:
        d = abs(h[k] - tr[k]); worst = max(worst, d)
        print(f"{h['t']:>2} {h['g']:>6.2f} | {label:<5} {h[k]:>22.15e} {tr[k]:>22.15e} {d:>9.1e}")
    print()
print(f"worst absolute difference over all 30 numbers: {worst:.1e}")
print("(m^ and v^ are not stored by PyTorch; they are rebuilt from its exp_avg / exp_avg_sq and its step counter.\n"
      " The step is measured as the change PyTorch actually made to the weight.)")

# The thing worth seeing: Adam's step is ~lr regardless of the gradient's size, early on.
print("\nstep / lr at each t:", [round(h["step"] / lr, 4) for h in hand])
print("gradients          :", grads)
