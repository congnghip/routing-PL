"""Independent feasibility checker for PACO/IG route dumps ('SOL <inst> <cost> : C1 P0(C2,C3) | ...').

Usage: python3 src/check_solution.py <data_dir> <dump.log>
Model (same as the IG paper code): speed 1, arc lengths truncated to 0.1, wait until `early`,
service time paid once per stop, arrival must be <= `late` of the stop node.
"""
import math
import re
import sys

def read_instance(path):
    t = [l.split() for l in open(path).read().replace('\r', '').split('\n') if l.strip()]
    n, k = int(t[0][0]), int(t[0][1])
    cap = float(t[1][1])
    dem = [float(t[2 + i][0]) for i in range(n)]
    rows = [list(map(float, t[2 + n + i])) for i in range(1 + n + k)]
    pref = t[2 + n + 1 + n + k:]
    return n, k, cap, dem, rows, pref

def check(data_dir, inst, cost, text):
    n, k, cap, dem, rows, pref = read_instance(f"{data_dir}/{inst}")
    # rows[0]=depot, 1..n customers, n+1..n+k lockers
    d = lambda a, b: math.floor(math.hypot(rows[a][0] - rows[b][0], rows[a][1] - rows[b][1]) * 10) / 10  # IG truncates arcs to 0.1
    ctype = [int(rows[i][5]) for i in range(1, n + 1)]
    seen, total, errs, warns = set(), 0.0, [], []
    for ri, route in enumerate(r for r in text.split('|') if r.strip()):
        cur, t, load = 0, 0.0, 0.0
        lockers = set()
        for tok in re.findall(r'P\d+\([^)]*\)|C\d+', route):
            if tok[0] == 'C':
                c = int(tok[1:]) + 1
                node, custs = c, [c]
                if ctype[c - 1] == 2:
                    errs.append(f"{tok} type2 served at home")
            else:
                lk = int(re.match(r'P(\d+)', tok).group(1))
                node = n + 1 + lk
                custs = [int(x[1:]) + 1 for x in re.findall(r'C\d+', tok)]
                if node in lockers:
                    warns.append(f"route {ri}: locker {tok[:2]} visited twice")
                lockers.add(node)
                for c in custs:
                    if ctype[c - 1] == 1:
                        errs.append(f"C{c-1} type1 at locker")
                    if pref[c - 1][lk] != '1':
                        errs.append(f"C{c-1} not assigned to locker P{lk}")
            for c in custs:
                if c in seen: errs.append(f"C{c-1} served twice")
                seen.add(c)
                load += dem[c - 1]
            total += d(cur, node)
            t += d(cur, node)
            t = max(t, rows[node][2])
            if t > rows[node][3] + 1e-6:
                errs.append(f"route {ri}: {tok} arrives {t:.1f} > late {rows[node][3]}")
            t += rows[node][4]
            cur = node
        total += d(cur, 0)
        if load > cap + 1e-9: errs.append(f"route {ri}: load {load} > {cap}")
    if len(seen) != n: errs.append(f"served {len(seen)}/{n}")
    if abs(total - cost) > 0.05: errs.append(f"cost mismatch reported {cost} vs recomputed {total:.3f}")
    return total, errs, warns

bad = 0
for line in open(sys.argv[2]):
    m = re.match(r'SOL (\S+) (\S+) : (.*)', line)
    if not m: continue
    total, errs, warns = check(sys.argv[1], m.group(1), float(m.group(2)), m.group(3))
    print(f"{m.group(1):18} reported {float(m.group(2)):9.2f} recomputed {total:9.2f}  {'OK' if not errs else 'INFEASIBLE: ' + '; '.join(errs[:3])}{'  (warn: multi-visit locker)' if warns else ''}")
    bad += bool(errs)
print(f"infeasible: {bad}")
