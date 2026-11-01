# Level-6 survival locus: values of Norm_j(N) = det m_j(T) on (R/I)_4, for exact integer N.
# I = ideal of the 9 selected quadrics (polynomial in N), m_j = fixed random combinations of closure cubics.
import pickle, itertools, random, sys, time
from fractions import Fraction as Fr
import poly as P

S = pickle.load(open('step1sym.pkl', 'rb')); cr6, rec = S['cr6'], S['rec']
Dz = pickle.load(open('design.pkl', 'rb'))
C = pickle.load(open('cert.pkl', 'rb')); O = C['O']
n = 5; sym = [(a, b) for a in range(n) for b in range(a, n)]
def unit(i): return tuple(int(t == i) for t in range(n))
def madd(a, b): return tuple(x + y for x, y in zip(a, b))
def monos(d): return sorted((m for m in itertools.product(range(d + 1), repeat=n) if sum(m) == d), reverse=True)

# quadrics as {mono: poly in N}
Qp = []
for key in Dz['Qsel']:
    vec = O[key[:5]][key[5]]; q = {}
    for pol, (a, b) in zip(vec, sym):
        if pol: m = madd(unit(a), unit(b)); q[m] = P.add(q.get(m, []), pol)
    Qp.append(q)
# closure cubics (all minors against a fixed reference coordinate), as {mono: poly}
cr_index = {Pp: a for a, Pp in enumerate(cr6)}
def xi_coef(Pp, e): return rec[e][cr_index[Pp]] if Pp in cr_index else []
cubs = []
closconfs = sorted({c for (c, _, _) in Dz['C3keys']})
for cf in closconfs:
    outP = sorted(O[cf], key=str)
    Ps = sorted(set(outP) | set(cr6), key=str)
    ref = Ps[0]
    for Pb in Ps[1:]:
        cub = {}
        for oP, xP, sg in [(ref, Pb, 1), (Pb, ref, -1)]:
            if oP not in O[cf]: continue
            for pol, (a, b) in zip(O[cf][oP], sym):
                if not pol: continue
                for e in range(5):
                    w = xi_coef(xP, e)
                    if w:
                        m = madd(madd(unit(a), unit(b)), unit(e))
                        cub[m] = P.add(cub.get(m, []), P.scal(Fr(sg), P.mul(pol, w)))
        cub = {m: p_ for m, p_ in cub.items() if p_}
        if cub: cubs.append(cub)
random.seed(2026)
combos = []
for _ in range(2):
    co = [random.randint(-5, 5) for _ in cubs]; m = {}
    for c_, cub in zip(co, cubs):
        if c_:
            for mo, p_ in cub.items(): m[mo] = P.add(m.get(mo, []), P.scal(Fr(c_), p_))
    combos.append({mo: p_ for mo, p_ in m.items() if p_})
print(f'{len(cubs)} closure cubics, 2 random combinations', flush=True)
ell = [Fr(x) for x in (2, -7, 1, 8, -3)]

def rref(rows, cols):
    order = {m: i for i, m in enumerate(cols)}; piv = {}
    for r in rows:
        r = dict(r)
        while True:
            hits = [m for m in r if m in piv]
            if not hits: break
            m = min(hits, key=order.get); f = r[m]
            for k2, v in piv[m].items():
                nv = r.get(k2, 0) - f * v
                if nv == 0: r.pop(k2, None)
                else: r[k2] = nv
        if not r: continue
        m = min(r, key=order.get); f = r[m]; r = {k2: v / f for k2, v in r.items()}
        for pm, pr in piv.items():
            if m in pr:
                g = pr[m]
                for k2, v in r.items():
                    nv = pr.get(k2, 0) - g * v
                    if nv == 0: pr.pop(k2, None)
                    else: pr[k2] = nv
        piv[m] = r
    return piv
def NF(p_, piv):
    p_ = dict(p_)
    for m in [m for m in p_ if m in piv]:
        f = p_.pop(m)
        for k2, v in piv[m].items():
            if k2 != m: p_[k2] = p_.get(k2, 0) - f * v
    return {k2: v for k2, v in p_.items() if v != 0}
def matmul(A, B): return [[sum(A[i][t] * B[t][j] for t in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]
def inv(A):
    m = len(A); M = [list(r) + [Fr(int(i == j)) for j in range(m)] for i, r in enumerate(A)]
    for c in range(m):
        pv = next(r for r in range(c, m) if M[r][c] != 0); M[c], M[pv] = M[pv], M[c]
        v = M[c][c]; M[c] = [x / v for x in M[c]]
        for r in range(m):
            if r != c and M[r][c] != 0:
                f = M[r][c]; M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [r[m:] for r in M]
def det(A):
    M = [list(r) for r in A]; m = len(M); D = Fr(1)
    for c in range(m):
        pv = next((r for r in range(c, m) if M[r][c] != 0), None)
        if pv is None: return Fr(0)
        if pv != c: M[c], M[pv] = M[pv], M[c]; D = -D
        D *= M[c][c]
        for r in range(c + 1, m):
            f = M[r][c] / M[c][c]; M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return D

def norms_at(N):
    Q = [{m: P.ev(p_, N) for m, p_ in q.items()} for q in Qp]
    Q = [{m: v for m, v in q.items() if v != 0} for q in Q]
    piv4 = rref([{madd(m, mm): v for m, v in q.items()} for q in Q for mm in monos(2)], monos(4))
    piv5 = rref([{madd(m, mm): v for m, v in q.items()} for q in Q for mm in monos(3)], monos(5))
    B4 = [m for m in monos(4) if m not in piv4]; B5 = [m for m in monos(5) if m not in piv5]
    if len(B4) != 4 or len(B5) != 4: return None
    def mult(lin):
        A = [[Fr(0)] * 4 for _ in range(4)]
        for j, m in enumerate(B4):
            p_ = {}
            for i, a in enumerate(lin):
                if a: k2 = madd(m, unit(i)); p_[k2] = p_.get(k2, 0) + a
            nf = NF(p_, piv5)
            for i, b in enumerate(B5): A[i][j] = nf.get(b, Fr(0))
        return A
    Al = mult(ell); Ali = inv(Al)
    T = [matmul(Ali, mult([Fr(int(i == t)) for t in range(n)])) for i in range(n)]
    I4 = [[Fr(int(i == j)) for j in range(4)] for i in range(4)]
    out = []
    for cm in combos:
        Mtot = [[Fr(0)] * 4 for _ in range(4)]
        for mo, p_ in cm.items():
            v = P.ev(p_, N)
            if v == 0: continue
            X = I4
            for i in range(n):
                for _ in range(mo[i]): X = matmul(X, T[i])
            Mtot = [[a + v * b for a, b in zip(r1, r2)] for r1, r2 in zip(Mtot, X)]
        out.append(det(Mtot))
    return out

if __name__ == '__main__':
    lo, hi = int(sys.argv[1]), int(sys.argv[2]); res = {}
    for N in range(lo, hi):
        t = time.time(); v = norms_at(N); res[N] = v
        print(N, round(time.time() - t, 2), 'ok' if v else 'degenerate', flush=True)
    pickle.dump(res, open(f'locus_{lo}_{hi}.pkl'.replace('0','o').replace('1','i').replace('2','z').replace('3','e').replace('4','a').replace('5','s').replace('6','x').replace('7','v').replace('8','b').replace('9','n'), 'wb'))
