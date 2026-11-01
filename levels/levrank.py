import sys, itertools
from fractions import Fraction
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from levquad import Kbasis, contract
from levlin import noncrossing
def run(N):
    vecs,crP,K=Kbasis(N); n=len(vecs)
    pairs=[(a,b) for a in range(n) for b in range(n)]
    sym_idx={}; 
    for a in range(n):
        for b in range(a,n): sym_idx[(a,b)]=len(sym_idx)
    rows=[]
    for s in range(6):
        for t in range(6):
            for j in (4,5,6):
                for o1 in (0,1):
                    for o2 in (0,1):
                        L=12-2*j+o1+o2
                        if L>5: continue
                        coeff={}   # crossing partition -> 15-vector
                        for a,b in pairs:
                            _,r=contract(vecs[a],vecs[b],s,t,j,o1,o2,N)
                            for P,c in r.items():
                                if noncrossing(P): continue
                                v=coeff.setdefault(P,[Fraction(0)]*len(sym_idx))
                                v[sym_idx[(min(a,b),max(a,b))]]+=c
                        rows+=list(coeff.values())
    # rank over Q
    M=[r[:] for r in rows]; rk=0; m=len(sym_idx)
    for c in range(m):
        pr=next((i for i in range(rk,len(M)) if M[i][c]!=0),None)
        if pr is None: continue
        M[rk],M[pr]=M[pr],M[rk]; v=M[rk][c]; M[rk]=[x/v for x in M[rk]]
        for i in range(len(M)):
            if i!=rk and M[i][c]!=0:
                f=M[i][c]; M[i]=[x-f*y for x,y in zip(M[i],M[rk])]
        rk+=1
    return rk, m, len(rows)
if __name__=='__main__':
    for N in [int(x) for x in sys.argv[1:]]:
        rk,m,nr=run(N)
        print(f'N={N}: quadratic constraints {nr}, rank {rk} of {m} possible  ->',
              'NO nonzero level-6 candidate survives' if rk==m else f'{m-rk}-dim space of quadratic forms unconstrained',flush=True)
