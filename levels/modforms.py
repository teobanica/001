# Level-8 quadric span modulo p (p = 1 mod 8, p < 2^20), at N = 11: reduced basis saved for the cubic test.
import sys, time, numpy as np
from engine import kernel_basis, labels, forms, BatchRank, upper, config_outputs
k = 8; N = 11; p = int(sys.argv[1]); part, nparts = int(sys.argv[2]), int(sys.argv[3])
crk, W, free = kernel_basis(k, N, p); d = W.shape[1]; L, nb = labels(crk, k)
np.save('W8_mod.npy', W.astype(np.int64)); np.save('free8.npy', np.array(free))
confs = [(s, t, j, o1, o2) for s in range(k) for t in range(k) for j in range(4, k + 1)
         for o1 in (0, 1) for o2 in (0, 1) if 2 * k - 2 * j + o1 + o2 <= k - 1 and len(config_outputs(k, (s, t, j, o1, o2))[2]) >= 4]
confs = confs[part::nparts]
BR = BatchRank(d * (d + 1) // 2, p); cache = {}
for ci, conf in enumerate(confs):
    t = time.time(); acc, nout = forms(L, nb, k, conf, W, N, p=p, crossing_cache=cache)
    if acc: BR.add(np.array([upper(Q) for Q in acc.values()]))
    print(f'{ci+1}/{len(confs)} rank {BR.rank} ({time.time()-t:.1f}s)', flush=True)
np.save(f'modspan_{"ab"[part]}.npy', BR.B.astype(np.int64))
