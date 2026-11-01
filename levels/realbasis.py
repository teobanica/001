# Real orthonormal basis of K_8(11) (crossing coordinates, sorted crossing partitions), via SVD.
import numpy as np
from engine import crossing_parts, delete_leg, merge
k = 8; N = 11
crk = crossing_parts(k); crl = crossing_parts(k - 1); il = {q: a for a, q in enumerate(crl)}
M = np.zeros((2 * k * len(crl), len(crk))); r0 = 0
for typ, i in [('d', i) for i in range(k)] + [('m', i) for i in range(k)]:
    for a, P in enumerate(crk):
        if typ == 'd':
            sing, q = delete_leg(P, i, k)
            if q in il: M[r0 + il[q], a] += N if sing else 1
        else:
            q = merge(P, i, (i + 1) % k, k)
            if q in il: M[r0 + il[q], a] += 1
    r0 += len(crl)
U, s, Vt = np.linalg.svd(M, full_matrices=True)
print('smallest singular values:', np.round(s[-70:][:8], 8), '...', 'gap:', s[2643], s[2644] if len(s) > 2644 else None)
null = Vt[np.sum(s > 1e-9 * s[0]):]
print('kernel dimension:', null.shape[0])
np.save('W8_real.npy', null.T)
