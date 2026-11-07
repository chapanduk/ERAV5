"""Q2: nudge single weights, measure the loss change, and compare with what backward() reported."""
import torch, torch.nn.functional as F
from common import Cfg, GPT, load_corpus, get_batch

train, _ = load_corpus()
c = Cfg(d=128, n_layer=4, n_head=4, ctx=64)
B, T = 8, 64


def build(dtype):
    torch.manual_seed(0)
    m = GPT(c).to(dtype)
    m.eval()  # no dropout in this model anyway, but be explicit: the loss must be a deterministic function of w
    return m


x, y = get_batch(train, B, T, torch.Generator().manual_seed(1), "cpu")


def loss_of(m):
    return F.cross_entropy(m(x).view(-1, c.vocab), y.view(-1))


def probe(m):
    """(label, param, index) triples spread across very different parts of the network."""
    present = int(x[0, 0])
    return [
        ("tok.weight[byte in batch, 5]", m.tok.weight, (present, 5)),
        ("pos.weight[10, 3]", m.pos.weight, (10, 3)),
        ("blocks.0.qkv.weight[7, 11]  (a query weight)", m.blocks[0].qkv.weight, (7, 11)),
        ("blocks.1.qkv.weight[200, 50] (a key weight)", m.blocks[1].qkv.weight, (200, 50)),
        ("blocks.2.fc.weight[17, 101]", m.blocks[2].fc.weight, (17, 101)),
        ("blocks.3.ln2.weight[9]", m.blocks[3].ln2.weight, (9,)),
        ("head.weight[ord('e'), 0]", m.head.weight, (ord("e"), 0)),
    ]


def fd(m, p, idx, eps):
    with torch.no_grad():
        w0 = p[idx].item()
        p[idx] = w0 + eps; lp = loss_of(m).item()
        p[idx] = w0 - eps; lm = loss_of(m).item()
        p[idx] = w0
    return (lp - lm) / (2 * eps)


print("== float64, central difference, eps = 1e-6 ==")
m = build(torch.float64)
loss_of(m).backward()
print(f"{'weight':<46} {'backward()':>16} {'finite diff':>16} {'rel err':>9}")
for label, p, idx in probe(m):
    g = p.grad[idx].item()
    n = fd(m, p, idx, 1e-6)
    print(f"{label:<46} {g:>16.10e} {n:>16.10e} {abs(g - n) / max(abs(g), 1e-30):>9.1e}")

print("\n== same check in float32: relative error vs eps (blocks.2.fc.weight[17,101]) ==")
m32 = build(torch.float32)
loss_of(m32).backward()
label, p, idx = probe(m32)[4]
g_true = probe(m)[4][1].grad[idx].item()  # fp64 value as ground truth
print(f"fp32 backward: {p.grad[idx].item():.10e}   fp64 backward: {g_true:.10e}")
print(f"{'eps':>8} {'fp32 finite diff':>18} {'rel err vs fp64':>16}")
for eps in [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    n = fd(m32, p, idx, eps)
    print(f"{eps:>8.0e} {n:>18.10e} {abs(n - g_true) / abs(g_true):>16.1e}")

print("\n== the key weight that only agreed to ~4 digits: float64 eps sweep ==")
label, p, idx = probe(m)[3]
g = p.grad[idx].item()
print(f"backward: {g:.10e}   loss = {loss_of(m).item():.4f}, fp64 machine eps = 2.2e-16")
print(f"{'eps':>8} {'finite diff':>18} {'abs err':>10} {'roundoff floor ~ eps_mach*loss/eps':>36}")
for eps in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    n = fd(m, p, idx, eps)
    print(f"{eps:>8.0e} {n:>18.10e} {abs(n - g):>10.1e} {2.2e-16 * 5.5 / eps:>36.1e}")

print("\n== key bias: softmax is shift-invariant, so this gradient should be exactly zero ==")
D = c.d
for j in [D + 0, D + 37]:   # rows D..2D-1 of qkv are the key projection
    gb = m.blocks[1].qkv.bias.grad[j].item()
    nb = fd(m, m.blocks[1].qkv.bias, (j,), 1e-4)
    print(f"blocks.1.qkv.bias[{j}] (key)   backward = {gb:.3e}   finite diff = {nb:.3e}")
qb = m.blocks[1].qkv.bias.grad[5].item()
print(f"blocks.1.qkv.bias[5]   (query) backward = {qb:.3e}   (non-zero, for contrast)")
