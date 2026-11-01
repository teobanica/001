import sys, pickle, json
from dmod import load, poly_mod
def primes_below(start, count):
    out = []; n = start
    while len(out) < count:
        n -= 1
        if n % 2 and all(n % q for q in range(3, int(n ** .5) + 1, 2)): out.append(n)
    return out
ps = primes_below(2 ** 30, 140)
part, nparts, tag = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
rows, den = load(); DB = sum(max(len(pol) - 1 for pol in row) for row in rows)
res = {}
for i, p in enumerate(ps):
    if i % nparts != part: continue
    res[p] = poly_mod(rows, p, DB)
    print(p, flush=True)
json.dump({str(p): v for p, v in res.items()}, open(f'primes_{tag}.json', 'w'))
