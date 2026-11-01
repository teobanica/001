# Rotation eigenbasis of K_k(N) mod p = 1048433 (p = 1 mod 8). Saves Wmod, E, chars for level k.
import sys, numpy as np
from engine import kernel_basis, crossing_parts, canon
p = 1048433; k = int(sys.argv[1]); N = int(sys.argv[2])
if k == 8:
    W = np.load('W8_mod.npy'); free = list(np.load('free8.npy'))
else:
    crk, W, free = kernel_basis(k, N, p); W = W.astype(np.int64)
d = W.shape[1]; crk = crossing_parts(k); idx = {P: a for a, P in enumerate(crk)}
perm = np.array([idx[canon([set((x + 1) % k for x in B) for B in P])] for P in crk])
PW = np.zeros_like(W); PW[perm] = W; Rot = PW[free] % p
if k != 8:
    tag = 'x' * k; np.save(f'Wmod{tag}.npy', W); np.save(f'E{tag}.npy', np.eye(d, dtype=np.int64)); np.save(f'chars{tag}.npy', np.zeros(d, dtype=np.int64)); np.save(f'free{tag}.npy', np.array(free)); print('level', k, 'd =', d, '(identity basis)'); raise SystemExit
g = next(x for x in range(2, p) if pow(x, (p - 1) // 2, p) != 1); om = pow(g, (p - 1) // k, p) if (p - 1) % k == 0 else None
assert om is not None
def nullspace_mod(M):
    M = M.copy() % p; rows, cols = M.shape; r = 0; piv = []
    for c in range(cols):
        nz = np.flatnonzero(M[r:, c])
        if len(nz) == 0: continue
        pr = r + nz[0]; M[[r, pr]] = M[[pr, r]]; M[r] = (M[r] * pow(int(M[r, c]), p - 2, p)) % p
        oth = np.flatnonzero(M[:, c]); oth = oth[oth != r]
        if len(oth): M[oth] = (M[oth] - (M[oth, c][:, None] * M[r][None, :]) % p) % p
        piv.append(c); r += 1
        if r == rows: break
    fr = [c for c in range(cols) if c not in piv]; Nm = np.zeros((cols, len(fr)), dtype=np.int64)
    for b, f in enumerate(fr):
        Nm[f, b] = 1
        for i, pc in enumerate(piv): Nm[pc, b] = (-M[i, f]) % p
    return Nm
Es = []; chars = []
for j in range(k):
    Ej = nullspace_mod((Rot - pow(om, j, p) * np.eye(d, dtype=np.int64)) % p); Es.append(Ej); chars += [j] * Ej.shape[1]
E = np.hstack(Es) % p; assert E.shape[1] == d, [e.shape[1] for e in Es]
tag = 'x' * k
np.save(f'Wmod{tag}.npy', W); np.save(f'E{tag}.npy', E); np.save(f'chars{tag}.npy', np.array(chars)); np.save(f'free{tag}.npy', np.array(free))
print(f'level {k}: d={d}, eigenspace dims {[e.shape[1] for e in Es]}')
