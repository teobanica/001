import sys, time, numpy as np
from engine import *
k = int(sys.argv[1]); N = int(sys.argv[2]); smax = int(sys.argv[3])
t = time.time(); crk, W, free = kernel_basis(k, N); d = W.shape[1]
L, nb = labels(crk, k)
print(f'level {k}, N={N}: dim K = {d}, target {d*(d+1)//2}, basis in {time.time()-t:.0f}s', flush=True)
confs = [(s, t_, j, o1, o2) for s in range(smax) for t_ in range(k) for j in range(4, k + 1)
         for o1 in (0, 1) for o2 in (0, 1) if 2 * k - 2 * j + o1 + o2 <= k - 1]
BR = BatchRank(d * (d + 1) // 2); cache = {}
for conf in confs:
    t = time.time(); acc, nout = forms(L, nb, k, conf, W, N, crossing_cache=cache)
    if acc: BR.add(np.array([upper(Q) for Q in acc.values()]))
    print(f'  {conf}: {len(acc)} forms, {time.time()-t:.1f}s, rank {BR.rank}', flush=True)
    if BR.rank == d * (d + 1) // 2: break
