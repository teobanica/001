import pickle, numpy as np
from fractions import Fraction as Fr
import poly as P
C = pickle.load(open('cert.pkl','rb'))
rem = C['C3'][1]
def shift(p, a):
    sh = [Fr(0)]
    for c in reversed(p): sh = P.add(P.mul(sh, [Fr(a), Fr(1)]), [c])
    return sh
def changes(p):
    s = [x > 0 for x in p if x != 0]; return sum(1 for a, b in zip(s, s[1:]) if a != b)
M = 8
while changes(shift(rem, M)) > 0: M *= 2
print('no roots beyond', M)
r = np.roots([float(x) for x in reversed(rem)])
print('numerical real roots > 5:', sorted(x.real for x in r if abs(x.imag) < 1e-6 and x.real > 5))
bad = [N for N in range(6, M + 1) if P.ev(rem, N) == 0]
print('integer roots in [6, %d]:' % M, bad)
# sign pattern at integers
print('sign of factor at N=6..%d:' % min(M, 40), ''.join('+' if P.ev(rem, N) > 0 else '-' for N in range(6, min(M, 40) + 1)))
