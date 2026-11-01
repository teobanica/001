import json, math
from crt import crt_poly, check_unused, analyse
from dmod import load, hadamard_bits
rows, den = load(); hb = hadamard_bits(rows)
res = {}
for f in ('primes_a.json', 'primes_b.json'):
    res.update({int(p): v for p, v in json.load(open(f)).items()})
print('primes available:', len(res), ' bound bits:', round(hb))
coeffs, used = crt_poly(res, math.ceil(hb))
print('primes used:', len(used), ' consistency with the', len(res) - len(used), 'unused primes:', check_unused(coeffs, res, used))
facs, rem, bad = analyse(coeffs, 7, 'D7(N) (105x105 quadric determinant)')
import pickle; pickle.dump({'coeffs': coeffs, 'facs': facs}, open('D7.pkl', 'wb'))
