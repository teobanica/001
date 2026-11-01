from levlin import *
from fractions import Fraction
def nullspace_exact(k,N):
    Pk=parts(k); Pl=parts(k-1); cross=[q for q in Pl if not noncrossing(q)]; cidx={q:a for a,q in enumerate(cross)}
    rows=[]
    for typ,i in [('d',i) for i in range(k)]+[('m',i) for i in range(k)]:
        R=[[Fraction(0)]*len(Pk) for _ in cross]
        for a,P in enumerate(Pk):
            if typ=='d': sing,q=delete_leg(P,i,k); f=N if sing else 1
            else: q=merge(P,i,(i+1)%k,k); f=1
            if q in cidx: R[cidx[q]][a]+=f
        rows+=R
    # RREF
    M=[r[:] for r in rows]; piv=[]; r=0; n=len(Pk)
    for c in range(n):
        pr=next((i for i in range(r,len(M)) if M[i][c]!=0),None)
        if pr is None: continue
        M[r],M[pr]=M[pr],M[r]; v=M[r][c]; M[r]=[x/v for x in M[r]]
        for i in range(len(M)):
            if i!=r and M[i][c]!=0:
                f=M[i][c]; M[i]=[x-f*y for x,y in zip(M[i],M[r])]
        piv.append(c); r+=1
    free=[c for c in range(n) if c not in piv]
    basis=[]
    for fcol in free:
        v=[Fraction(0)]*n; v[fcol]=Fraction(1)
        for i,pc in enumerate(piv): v[pc]=-M[i][fcol]
        basis.append(v)
    return Pk,basis
if __name__=='__main__':
    for N in (7,11):
        Pk,B=nullspace_exact(6,N)
        nc=[noncrossing(P) for P in Pk]
        # project out NC part: keep crossing coordinates only, then reduce
        cr=[a for a in range(len(Pk)) if not nc[a]]
        vecs=[[v[a] for a in cr] for v in B]
        # row reduce vecs to get basis of K (mod NC)
        M=[v[:] for v in vecs]; r=0; out=[]
        for c in range(len(cr)):
            pr=next((i for i in range(r,len(M)) if M[i][c]!=0),None)
            if pr is None: continue
            M[r],M[pr]=M[pr],M[r]; v=M[r][c]; M[r]=[x/v for x in M[r]]
            for i in range(len(M)):
                if i!=r and M[i][c]!=0:
                    f=M[i][c]; M[i]=[x-f*y for x,y in zip(M[i],M[r])]
            r+=1
        K=[row for row in M[:r]]
        print(f'N={N}: dim K={len(K)}; supports on crossing partitions:',[sum(1 for x in row if x!=0) for row in K])
        for row in K:
            print('  ',{Pk[cr[a]]:str(x) for a,x in enumerate(row) if x!=0})

