"""Q6: check the hand derivation of 0.1 in fp32 / bf16 / fp8 E4M3 against the real casts."""
import struct, torch

def bits_of(x, dtype):
    t = torch.tensor([x], dtype=torch.float64).to(dtype)
    n = {torch.float32: 32, torch.bfloat16: 16, torch.float8_e4m3fn: 8}[dtype]
    u = t.view({32: torch.int32, 16: torch.int16, 8: torch.uint8}[n]).item() & ((1 << n) - 1)
    return format(u, f"0{n}b"), t.to(torch.float64).item()

layout = {torch.float32: (8, 127), torch.bfloat16: (8, 127), torch.float8_e4m3fn: (4, 7)}
for dt in [torch.float32, torch.bfloat16, torch.float8_e4m3fn]:
    b, v = bits_of(0.1, dt)
    e, bias = layout[dt]
    s, ex, m = b[0], b[1:1 + e], b[1 + e:]
    mant = 1 + sum(int(ch) * 2 ** -(i + 1) for i, ch in enumerate(m))
    print(f"{str(dt).replace('torch.', ''):<14} {s} {ex} {m:<23}  hex {int(b, 2):0{len(b)//4}X}  "
          f"= {mant:.10f} x 2^({int(ex, 2)}-{bias}) = {v:.12g}   rel err {abs(v - 0.1) / 0.1:.2e}")
print(f"\nfp64 for reference: 0.1 -> {0.1:.20f} (0.1 is not exact even there)")
print(f"neighbours of 0.1 in E4M3: {[bits_of(x, torch.float8_e4m3fn)[1] for x in (0.09, 0.1, 0.11)]}")
print(f"E4M3 range: max {torch.finfo(torch.float8_e4m3fn).max}, smallest normal {torch.finfo(torch.float8_e4m3fn).tiny}")
print(f"bf16 range: max {torch.finfo(torch.bfloat16).max:.3e}  (same exponent field as fp32: {torch.finfo(torch.float32).max:.3e})")
print(f"fp16 range: max {torch.finfo(torch.float16).max}  (why fp16 training needs loss scaling and bf16 doesn't)")
print(f"\nbf16 swamping: 1.0 + 0.001 in bf16 = {(torch.tensor(1.0, dtype=torch.bfloat16) + torch.tensor(0.001, dtype=torch.bfloat16)).item()}"
      f"  (ulp at 1.0 is 2^-7 = {2**-7}) -> why the optimizer's weights and update stay in fp32")
