import itertools, sys
from hm import set_partitions
def canon(blocks):
    # blocks: list of sets over positions 0..m-1 ; return canonical tuple of sorted tuples
    return tuple(sorted(tuple(sorted(b)) for b in blocks if b))
def noncrossing(P):
    for A,B in itertools.permutations(P,2):
        for a1,a2 in itertools.combinations(A,2):
            for b1,b2 in itertools.combinations(B,2):
                if a1<b1<a2<b2: return False
    return True
def parts(k): return [canon([set(b) for b in p]) for p in set_partitions(k)]
def delete_leg(P,i,k):   # sum over leg i: returns (factor_is_N, new partition on k-1 legs)
    sing=any(b==(i,) for b in P)
    nb=[set(x if x<i else x-1 for x in b if x!=i) for b in P]
    return sing, canon(nb)
def merge(P,i,j,k):      # identify legs i and j (j=i+1 mod k) -> k-1 legs; new leg placed at position of min, cyclic handled by relabel
    # relabel so that merged leg sits at position i (if j==i+1) ; for cyclic pair (k-1,0) put merged leg at position 0
    bi=[b for b in P if i in b][0]; bj=[b for b in P if j in b][0]
    blocks=[set(b) for b in P if b not in (bi,bj)]+[set(bi)|set(bj)]
    # remove j, map i/j -> merged position
    if j==i+1:
        f=lambda x: x if x<j else x-1
        nb=[set(f(x) for x in b if x!=j) for b in blocks]
    else: # i=k-1, j=0 : drop leg k-1, keep leg 0
        nb=[set(x for x in b if x!=i) for b in blocks]
    return canon(nb)
def rank_mod(rows,ncols,p):
    M=[r[:] for r in rows]; r=0
    for c in range(ncols):
        piv=next((i for i in range(r,len(M)) if M[i][c]%p),None)
        if piv is None: continue
        M[r],M[piv]=M[piv],M[r]; inv=pow(M[r][c],p-2,p); M[r]=[x*inv%p for x in M[r]]
        for i in range(len(M)):
            if i!=r and M[i][c]%p:
                f=M[i][c]; M[i]=[(x-f*y)%p for x,y in zip(M[i],M[r])]
        r+=1
    return r
def dimK(k,N,p=2**61-1):
    Pk=parts(k); Pl=parts(k-1); idx={q:a for a,q in enumerate(Pl)}
    cross=[q for q in Pl if not noncrossing(q)]; cidx={q:a for a,q in enumerate(cross)}
    ncK=sum(noncrossing(q) for q in Pk)
    # constraint rows: for each map and each crossing partition of level k-1, coefficient functional on P(k)
    rows=[]
    maps=[('d',i) for i in range(k)]+[('m',i) for i in range(k)]
    for typ,i in maps:
        R=[[0]*len(Pk) for _ in cross]
        for a,P in enumerate(Pk):
            if typ=='d':
                sing,q=delete_leg(P,i,k); f=N if sing else 1
            else:
                q=merge(P,i,(i+1)%k,k); f=1
            if q in cidx: R[cidx[q]][a]=(R[cidx[q]][a]+f)%p
        rows+=R
    r=rank_mod(rows,len(Pk),p)
    return len(Pk)-r-ncK, len(Pk), ncK
if __name__=='__main__':
    k=int(sys.argv[1]) if len(sys.argv)>1 else 6
    for k_ in [4,5]:
        print('sanity level',k_,'dim K for N=7,10,100:',[dimK(k_,N)[0] for N in (7,10,100)],flush=True)
    res={}
    for N in list(range(6,41))+[97,1009,10**6+3]:
        res[N]=dimK(k,N)[0]
    print('level',k,': dim K(N) =',res,flush=True)
