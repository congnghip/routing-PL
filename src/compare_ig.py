"""Compare a GPU solver CSV against the IG paper results (and BKS = best known across g/sa/ig bests).

Usage: python3 src/compare_ig.py <gpu.csv> <ig_result.csv>
"""
import csv
import sys

def num(x):
    try:
        return float(x)
    except ValueError:
        return None

gpu = {r['instance_name'].replace('.txt', '').rsplit('_co_', 1)[0]: r for r in csv.DictReader(open(sys.argv[1]))}
pub = {r['instance']: r for r in csv.DictReader(open(sys.argv[2]))}

rows = []
for inst, g in gpu.items():
    p = pub[inst]
    bks = min(v for v in (num(p['g_best_cost']), num(p['sa_best_cost']), num(p['ig_best_cost'])) if v)
    rows.append((inst, bks, num(p['ig_avg_cost']), num(p['ig_best_cost']),
                 float(g['AVG Distance']), float(g['Best Distance']),
                 num(p['ig_avg_time']), float(g['AVG Runtime (s)'])))

print(f"{'inst':7}{'BKS':>9}{'IGavg':>9}{'GPUavg':>9}{'gapAvg%':>8}{'GPUbest':>9}{'gapBest%':>9}")
for inst, bks, iga, igb, ga, gb, _, _ in rows:
    flag = ' *NEW BKS*' if gb < bks - 1e-3 else ''
    print(f"{inst:7}{bks:9.1f}{iga:9.1f}{ga:9.1f}{(ga-bks)/bks*100:8.2f}{gb:9.1f}{(gb-bks)/bks*100:9.2f}{flag}")

n = len(rows)
tol = 1e-3
# BKS = best known over Gurobi / SA / IG / this solver; "best avg" = lowest avg cost among IG and GPU+IG
bks_all = [min(r[1], r[5]) for r in rows]
best_avg = [min(r[2], r[4]) for r in rows]
ig_avg = sum(r[2] <= b + tol for r, b in zip(rows, best_avg))
gpu_avg = sum(r[4] <= b + tol for r, b in zip(rows, best_avg))
ig_bks = sum(r[3] <= b + tol for r, b in zip(rows, bks_all))
gpu_bks = sum(r[5] <= b + tol for r, b in zip(rows, bks_all))
ig_t = sum(r[6] < r[7] for r in rows)
gpu_t = sum(r[7] < r[6] for r in rows)
print(f"\nIG      (avg_cost: {ig_avg}/{n}, bks: {ig_bks}/{n}, avg_time: {ig_t}/{n})")
print(f"GPU+IG  (avg_cost: {gpu_avg}/{n}, bks: {gpu_bks}/{n}, avg_time: {gpu_t}/{n})")
print(f"mean avg_time: IG {sum(r[6] for r in rows)/n:.3f}s | GPU+IG {sum(r[7] for r in rows)/n:.3f}s")
print(f"new BKS found by GPU+IG (strictly below Gurobi/SA/IG): {sum(r[5] < r[1] - tol for r in rows)}")
