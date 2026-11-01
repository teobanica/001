import itertools, sys
from fractions import Fraction
from levlin import parts, noncrossing, canon
from levnull import nullspace_exact
def Kbasis(N):
    Pk,B=nullspace_exact(6,N)
    cr=[a for a,P in enumerate(Pk) if not noncrossing(P)]
    M=[[v[a] for a in cr] for v in B]; r=0
    for c in range(len(cr)):
        pr=next((i for i in range(r,len(M)) if M[i][c]!=0),None)
        if pr is None: continue
        M[r],M[pr]=M[pr],M[r]; v=M[r][c]; M[r]=[x/v for x in M[r]]
        for i in range(len(M)):
            if i!=r and M[i][c]!=0:
                f=M[i][c]; M[i]=[x-f*y for x,y in zip(M[i],M[r])]
        r+=1
    K=M[:r]; crP=[Pk[a] for a in cr]
    vecs=[{crP[a]:x for a,x in enumerate(row) if x!=0} for row in K]
    return vecs, crP, K
class UF:
    def __init__(s): s.p={}
    def f(s,x):
        s.p.setdefault(x,x)
        while s.p[x]!=x: s.p[x]=s.p[s.p[x]]; x=s.p[x]
        return x
    def u(s,a,b): s.p[s.f(a)]=s.f(b)
def contract(xi,eta,s,t,j,o1,o2,N):
    # xi legs ('x',i), eta legs ('y',i); strings x_{s+i} <-> y_{t+j-1-i}
    xs=[(s+i)%6 for i in range(j)]; ys=[(t+j-1-i)%6 for i in range(j)]
    outs=[]                                     # output order
    for i in range(6-j): outs.append(('x',(s+j+i)%6))
    if o1: outs.append(('x',xs[0]))
    for i in range(6-j): outs.append(('y',(t+j+i)%6))
    if o2 and j>1: outs.append(('x',xs[-1]))
    L=len(outs); res={}
    for p,cp in xi.items():
        for q,cq in eta.items():
            U=UF()
            for B in p:
                for a in B[1:]: U.u(('x',a),('x',B[0]))
            for B in q:
                for a in B[1:]: U.u(('y',a),('y',B[0]))
            for a,b in zip(xs,ys): U.u(('x',a),('y',b))
            nodes=[('x',i) for i in range(6)]+[('y',i) for i in range(6)]
            roots={U.f(n) for n in nodes}
            outroots=[U.f(o) for o in outs]
            closed=len(roots-set(outroots))
            blocks={}
            for pos,r in enumerate(outroots): blocks.setdefault(r,set()).add(pos)
            key=canon(list(blocks.values()))
            res[key]=res.get(key,0)+cp*cq*N**closed
    return L,{k:v for k,v in res.items() if v!=0}
def run(N):
    vecs,crP,K=Kbasis(N); cidx={P:a for a,P in enumerate(crP)}
    def inK(vec):   # crossing part of level-6 vector in span K ?
        v=[Fraction(0)]*len(crP)
        for P,c in vec.items():
            if P in cidx: v[cidx[P]]+=c
        # subtract K rows using pivot structure
        for row in K:
            pc=next(i for i,x in enumerate(row) if x!=0)
            if v[pc]!=0:
                f=v[pc]/row[pc]; v=[a-f*b for a,b in zip(v,row)]
        return all(x==0 for x in v)
    bad=[]; checked=0
    for a in range(len(vecs)):
        for b in range(len(vecs)):
            for s in range(6):
                for t in range(6):
                    for j in (3,4,5,6):
                        for o1 in (0,1):
                            for o2 in (0,1):
                                if 12-2*j+o1+o2>6: continue
                                L,r=contract(vecs[a],vecs[b],s,t,j,o1,o2,N); checked+=1
                                if L<=5:
                                    crossing=any(not noncrossing(P) for P in r)
                                    if crossing: bad.append((a,b,s,t,j,o1,o2,L))
                                else:
                                    if not inK(r): bad.append((a,b,s,t,j,o1,o2,L))
    return checked,bad
if __name__=='__main__':
    N=int(sys.argv[1]) if len(sys.argv)>1 else 7
    checked,bad=run(N)
    print(f'N={N}: contractions checked {checked}, violations {len(bad)}')
    from collections import Counter
    print(' violations by (j,o1+o2,level):',Counter((x[4],x[5]+x[6],x[7]) for x in bad))
    print(' violations by basis pair:',Counter((x[0],x[1]) for x in bad))
