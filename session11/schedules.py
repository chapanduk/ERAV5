"""Task 4: the same model under cosine and under WSD, both planned for 300 steps, both stopped at step 200.

Tune both sides first: each schedule gets its own peak-LR sweep, scored on val loss at step 200 (the
point being compared). Then each is rerun at its own best LR on 5 seeds. Same seed = same init and same
batches across the two schedules, so per-seed differences are paired.

Extra (clearly not part of the 300/200 comparison): the WSD step-200 checkpoint followed by a 40-step
linear cooldown to 0, to price what keeping the WSD checkpoint is worth.
"""
import math, statistics
import torch, torch.nn.functional as F
from common import Cfg, ResultStore, train, evaluate, get_batch, corpus, lr_at

CFG, TOTAL, STOP, WARMUP = Cfg(d=256), 300, 200, 30
LRS = [5e-4, 7.07e-4, 1e-3, 1.41e-3, 2e-3, 4e-3, 8e-3, 1.6e-2]   # factor-2 grid, refined to sqrt(2) around the optimum
TUNE_SEEDS, FINAL_SEEDS, COOLDOWN = [0, 1], [0, 1, 2, 3, 4], 40
EVAL_AT = tuple(range(25, STOP + 1, 25))
store = ResultStore("schedule_results.json")


def run(sched, lr, seed):
    key = f"{sched}|lr={lr}|seed={seed}"
    if key not in store:
        log = train(CFG, lr, sched=sched, total=TOTAL, stop=STOP, warmup=WARMUP, seed=seed, eval_at=EVAL_AT)
        store.put(key, dict(val=log["val"], final_val=log["final_val"], loss=log["loss"], lr=log["lr"]))
        print(f"  {key}: val@{STOP} {log['final_val']:.4f}", flush=True)
    return store[key]


def wsd_cooldown(lr, seed):
    """Rebuild the WSD run to step 200 (same seed -> same trajectory up to MPS noise), then decay to 0 over 40 steps."""
    key = f"wsd+cooldown{COOLDOWN}|lr={lr}|seed={seed}"
    if key not in store:
        log, model, opt, g = train(CFG, lr, sched="wsd", total=TOTAL, stop=STOP, warmup=WARMUP, seed=seed,
                                   return_model=True)
        for i in range(COOLDOWN):
            for pg in opt.param_groups:
                pg["lr"] = lr * (COOLDOWN - i) / COOLDOWN
            x, y = get_batch(corpus()[0], 32, CFG.ctx, g)
            loss = F.cross_entropy(model(x).view(-1, CFG.vocab), y.view(-1))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        store.put(key, dict(at200=log["final_val"], after=evaluate(model)))
        print(f"  {key}: val@200 {store[key]['at200']:.4f} -> val@{STOP + COOLDOWN} {store[key]['after']:.4f}", flush=True)
    return store[key]


def mean(xs):
    xs = [x for x in xs if math.isfinite(x)]
    return sum(xs) / len(xs) if xs else float("nan")


if __name__ == "__main__":
    print(f"LR at step {STOP} of a {TOTAL}-step plan, as a fraction of peak: "
          f"cosine {lr_at(STOP, 'cosine', 1, TOTAL, WARMUP):.3f}, wsd {lr_at(STOP, 'wsd', 1, TOTAL, WARMUP):.3f} "
          f"(WSD decays over steps {TOTAL - int(0.2 * TOTAL)}-{TOTAL}, which a stop at {STOP} never reaches)\n")

    print("== 1. tune both sides: val loss at step 200, mean of seeds", TUNE_SEEDS)
    best = {}
    for sched in ["cosine", "wsd"]:
        row = {lr: mean([run(sched, lr, s)["final_val"] for s in TUNE_SEEDS]) for lr in LRS}
        best[sched] = min(row, key=lambda k: row[k] if math.isfinite(row[k]) else 9e9)
        print(f"  {sched:<6}: " + "  ".join(f"{lr:g}:{v:.4f}" for lr, v in row.items()) + f"   -> best {best[sched]:g}")
        edge = best[sched] in (LRS[0], LRS[-1])
        print(f"          best LR is {'AT THE EDGE of the grid - widen it' if edge else 'inside the grid (bracketed)'}")

    print(f"\n== 2. each at its own best LR, seeds {FINAL_SEEDS}: val loss at step {STOP}")
    res = {s: [run(s, best[s], seed)["final_val"] for seed in FINAL_SEEDS] for s in ["cosine", "wsd"]}
    for s, v in res.items():
        print(f"  {s:<6} (lr {best[s]:g}): mean {mean(v):.4f}  sd {statistics.stdev(v):.4f}  per seed {[round(x, 4) for x in v]}")
    diff = [w - c for c, w in zip(res["cosine"], res["wsd"])]
    print(f"  paired wsd - cosine: mean {mean(diff):+.4f}, per seed {[round(d, 4) for d in diff]}, "
          f"cosine better on {sum(d > 0 for d in diff)}/{len(diff)} seeds")

    print(f"\n== 3. untuned comparison for contrast: both at the same LR 2e-3")
    for s in ["cosine", "wsd"]:
        print(f"  {s:<6}: {mean([run(s, 2e-3, seed)['final_val'] for seed in TUNE_SEEDS]):.4f}")

    print(f"\n== extra (not the 300/200 comparison): WSD step-200 checkpoint + {COOLDOWN}-step cooldown to 0")
    cd = [wsd_cooldown(best["wsd"], seed) for seed in FINAL_SEEDS]
    print(f"  val@200 {mean([c['at200'] for c in cd]):.4f} -> val@{STOP + COOLDOWN} {mean([c['after'] for c in cd]):.4f}   "
          f"(cosine@200 {mean(res['cosine']):.4f})")
