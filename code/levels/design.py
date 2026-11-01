# Design the certificates at one N (mod p ranks): choose quadrics, hyperplane, cubics, degrees.
import pickle, itertools, sys, random
from fractions import Fraction as Fr
p = 2**61 - 1
D = pickle.load(open(sys.argv[1], 'rb')); N = D['N']
n = 5; sym = [(a, b) for a in range(n) for b in range(a, n)]
def unit(i): return tuple(int(k == i) for k in range(n))
def add(a, b): return tuple(x + y for x, y in zip(a, b))
def monos(d, nv=n): return sorted((m for m in itertools.product(range(d + 1), repeat=nv) if sum(m) == d), reverse=True)
def md(x): return x.numerator * pow(x.denominator, -1, p) % p
def rank_rows(rows, cols):     # rows: dict mono->int mod p
    ci = {m: i for i, m in enumerate(cols)}; piv = {}; chosen = []
    for idx, r in enumerate(rows):
        v = {ci[m]: x % p for m, x in r.items() if x % p}
        while v:
            c = min(v)
            if c not in piv: break
            f = v[c]; pr = piv[c]
            for k, y in pr.items():
                v[k] = (v.get(k, 0) - f * y) % p
                if v[k] == 0: del v[k]
        if v:
            c = min(v); inv = pow(v[c], -1, p); piv[c] = {k: y * inv % p for k, y in v.items()}; chosen.append(idx)
    return len(piv), chosen
# quadrics as polys
qkeys = sorted(D['quad'], key=str)
def qpoly(vec):
    q = {}
    for x, (a, b) in zip(vec, sym):
        if x: m = add(unit(a), unit(b)); q[m] = (q.get(m, 0) + md(x)) % p
    return q
allq = [qpoly(D['quad'][k]) for k in qkeys]
r, ch = rank_rows(allq, monos(2)); Qsel = [qkeys[i] for i in ch]
print(f'N={N}: quadric rank {r}; chosen keys {len(Qsel)}')
# (C2): hyperplane c_h = 0, Macaulay in remaining vars
for h in range(5):
    keep = [i for i in range(5) if i != h]
    qs = [{tuple(m[i] for i in keep): x for m, x in allq[i].items() if m[h] == 0} for i in ch]
    for d in range(2, 9):
        rows = [{add(mm, k): x for k, x in q.items()} for q in qs for mm in monos(d - 2, 4)]
        rk, _ = rank_rows(rows, monos(d, 4))
        if rk == len(monos(d, 4)):
            print(f'  (C2) hyperplane c{h}=0: empty at degree {d} (minor size {rk})'); break
    else: print(f'  (C2) hyperplane c{h}=0: not empty up to degree 8')
# (C3): closure cubics
cl = D['clos']; xi_basis = D['vecs']
crP = sorted({P for v in xi_basis for P in v}, key=str)
cubics = []; ckeys = []
for key in sorted(cl, key=str):
    out = cl[key]
    Ps = sorted(set(out) | set(crP), key=str)
    for Pa, Pb in itertools.combinations(Ps, 2):
        # o_Pa * xi_Pb - o_Pb * xi_Pa ; xi_P = sum_b c_b vecs[b][P]
        cub = {}
        for (oP, xP, sg) in [(Pa, Pb, 1), (Pb, Pa, -1)]:
            if oP not in out: continue
            for x, (a, b) in zip(out[oP], sym):
                if not x: continue
                for e in range(5):
                    y = xi_basis[e].get(xP, 0)
                    if y:
                        m = add(add(unit(a), unit(b)), unit(e)); cub[m] = (cub.get(m, 0) + sg * md(x) * md(Fr(y))) % p
        cub = {m: x for m, x in cub.items() if x}
        if cub: cubics.append(cub); ckeys.append((key, Pa, Pb))
    if len(cubics) > 3000: break
r3, ch3 = rank_rows(cubics, monos(3)); print(f'  closure cubics sampled: {len(cubics)}, rank in R_3: {r3} of {len(monos(3))}')
Csel = [cubics[i] for i in ch3]
qs = [allq[i] for i in ch]
for d in range(3, 8):
    rows = [{add(mm, k): x for k, x in q.items()} for q in qs for mm in monos(d - 2)] + \
           [{add(mm, k): x for k, x in c.items()} for c in Csel for mm in monos(d - 3)]
    rk, chosen = rank_rows(rows, monos(d))
    print(f'  (C3) degree {d}: rank {rk} of {len(monos(d))}')
    if rk == len(monos(d)): break
# can cubics alone + quadrics do it at degree 3?
rows = qs and [{add(mm, k): x for k, x in q.items()} for q in qs for mm in monos(1)] + Csel
print('  (C3) at degree 3 directly: rank', rank_rows(rows, monos(3))[0], 'of', len(monos(3)))
pickle.dump({'Qsel': Qsel, 'C3keys': [ckeys[i] for i in ch3]}, open('design.pkl', 'wb'))
