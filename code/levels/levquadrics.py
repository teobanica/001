import sys, io, contextlib, pickle
from fractions import Fraction
with contextlib.redirect_stdout(io.StringIO()):
    from levquad import Kbasis, contract
from levlin import noncrossing
import numpy as np
def quadrics(N):
    vecs,crP,K=Kbasis(N); n=len(vecs)
    sym=[(a,b) for a in range(n) for b in range(a,n)]; si={p:i for i,p in enumerate(sym)}
    rows=[]
    for s in range(6):
        for t in range(6):
            for j in (4,5,6):
                for o1 in (0,1):
                    for o2 in (0,1):
                        if 12-2*j+o1+o2>5: continue
                        coeff={}
                        for a in range(n):
                            for b in range(n):
                                _,r=contract(vecs[a],vecs[b],s,t,j,o1,o2,N)
                                for P,c in r.items():
                                    if noncrossing(P): continue
                                    v=coeff.setdefault(P,[Fraction(0)]*len(sym))
                                    v[si[(min(a,b),max(a,b))]]+=c
                        rows+=list(coeff.values())
    # independent rows (RREF)
    M=[r[:] for r in rows]; rk=0
    for c in range(len(sym)):
        pr=next((i for i in range(rk,len(M)) if M[i][c]!=0),None)
        if pr is None: continue
        M[rk],M[pr]=M[pr],M[rk]; v=M[rk][c]; M[rk]=[x/v for x in M[rk]]
        for i in range(len(M)):
            if i!=rk and M[i][c]!=0:
                f=M[i][c]; M[i]=[x-f*y for x,y in zip(M[i],M[rk])]
        rk+=1
    return M[:rk], sym, vecs
N=int(sys.argv[1]) if len(sys.argv)>1 else 7
Q,sym,vecs=quadrics(N)
pickle.dump((Q,sym,vecs),open(f'quad_N{N}.pkl','wb'))
print('independent quadrics:',len(Q))
for row in Q:
    print('  ',' + '.join(f'({x})*c{a}c{b}' for x,(a,b) in zip(row,sym) if x!=0))
