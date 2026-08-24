"""Measure nanoSubQ attention complexity: masked (O(n^2)) vs flex (O(n*kappa)).

This is the near-linearity experiment: as n grows, the masked-dense path should scale
~quadratically in wall-clock and memory, while the flex kernel should scale ~linearly
(attention O(n*kappa) + routing O((n/block)^2), the subquadratic flat-router regime).

Run on the T4 box:  python bench_scaling.py   -> table + bench_scaling.csv
"""
import os, sys, time, csv, argparse
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import nanoSubQConfig, SubQAttention


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="512,1024,2048,4096,8192,16384")
    ap.add_argument("--block", type=int, default=128)  # must be a multiple of 128 for the CUDA flex kernel
    ap.add_argument("--top_c", type=int, default=8)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("bench_scaling needs CUDA (run on the 2xT4 box). Skipping timing.")
        return

    torch.backends.cudnn.benchmark = True
    cfg = nanoSubQConfig(d_model=384, num_layers=1, num_q_heads=6, num_kv_heads=2,
                         block=args.block, top_c=args.top_c, local=1)
    ns = [int(x) for x in args.n.split(",")]
    rows = []
    print(f"{'n':>6} | {'masked(ms)':>10} | {'flex(ms)':>9} | {'speedup':>8} | {'masked_mem':>10} | {'flex_mem':>9}")
    for n in ns:
        if n % args.block:
            print(f"{n:6d} skipped (not a multiple of block={args.block})")
            continue
        m = SubQAttention(cfg).cuda().eval()
        x = torch.randn(1, n, 384, device="cuda", dtype=torch.float16)
        cos = torch.randn(1, 1, n, 64, device="cuda", dtype=torch.float16)
        sin = torch.randn(1, 1, n, 64, device="cuda", dtype=torch.float16)
        res = {}
        for impl in ("masked", "flex"):
            m.attn_impl = impl
            torch.cuda.empty_cache()
            with torch.no_grad():
                for _ in range(2):
                    m(x, cos, sin)                        # warmup (+ flex compile on first call)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()      # measure only the timed reps
                t0 = time.time()
                for _ in range(args.reps):
                    m(x, cos, sin)
                torch.cuda.synchronize()
                dt = (time.time() - t0) / args.reps * 1000
            res[impl] = (dt, torch.cuda.max_memory_allocated() / 1e9)
        speed = res["masked"][0] / res["flex"][0]
        rows.append([n, round(res["masked"][0], 2), round(res["flex"][0], 2), round(speed, 2),
                     round(res["masked"][1], 3), round(res["flex"][1], 3)])
        print(f"{n:6d} | {res['masked'][0]:10.2f} | {res['flex'][0]:9.2f} | {speed:8.2f}x | "
              f"{res['masked'][1]:10.3f} | {res['flex'][1]:9.3f}", flush=True)
        del m, x, cos, sin
        torch.cuda.empty_cache()

    with open(os.path.join(HERE, "bench_scaling.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "masked_ms", "flex_ms", "speedup", "masked_mem_GB", "flex_mem_GB"])
        w.writerows(rows)
    print("wrote bench_scaling.csv")


if __name__ == "__main__":
    main()
