import numpy as np, itertools, sys
from math import comb

def petrescu():
    w = np.exp(2j*np.pi/3)
    idx=[t for t in itertools.product([0,1],repeat=3) if t!=(0,0,0)]
    H=np.zeros((7,7),complex)
    for a,x in enumerate(idx):
        for b,y in enumerate(idx):
            s=sum(p*q for p,q in zip(x,y))%2
            v=(-1)**s
            if a==b and v==1: H[a,b]=-w
            elif a!=b and v==-1: H[a,b]=w
            else: H[a,b]=v
    return H
def fourier(N):
    return np.array([[np.exp(2j*np.pi*i*j/N) for j in range(N)] for i in range(N)])

def model(H):
    N=H.shape[0]
    U=np.zeros((N,N,N,N),complex)
    for i in range(N):
        for j in range(N):
            x=H[i]/H[j]/np.sqrt(N)
            U[i,j]=np.outer(x,x.conj())
    return U

def set_partitions(k):
    def rec(l):
        if l==0: yield []; return
        for p in rec(l-1):
            for b in range(len(p)):
                yield p[:b]+[p[b]+[l-1]]+p[b+1:]
            yield p+[[l-1]]
    return list(rec(k))
def noncrossing(p):
    for A,B in itertools.permutations(p,2):
        for a1,a2 in itertools.combinations(A,2):
            for b1,b2 in itertools.combinations(B,2):
                if a1<b1<a2<b2: return False
    return True

def delta(p,N,k):
    v=np.zeros((N,)*k)
    lab=[0]*k
    for bi,B in enumerate(p):
        for x in B: lab[x]=bi
    for vals in itertools.product(range(N),repeat=len(p)):
        v[tuple(vals[lab[s]] for s in range(k))]=1
    return v

def apply(U,xi,k):
    N=U.shape[0]
    # X indexed (j1..jk) scalars -> matrices
    X=xi.astype(complex)[...,None,None]*np.eye(N)
    # apply legs from last to first; leg s: X[..., j_s, (i_{s+1}..i_k), d,d] -> u_{i_s j_s} @ X
    for s in reversed(range(k)):
        X=np.moveaxis(X,s,0)            # j_s first
        X=np.einsum('ijab,j...bc->i...ac',U,X)
        X=np.moveaxis(X,0,s)
    return X

def joint_dim(H,k):
    N=H.shape[0]; U=model(H)
    parts=[p for p in set_partitions(k) if len(p)<=N]
    vecs=[]
    for p in parts:
        xi=delta(p,N,k)
        v=apply(U,xi,k)-xi[...,None,None]*np.eye(N)
        vecs.append(v.ravel())
    A=np.array(vecs).T
    s=np.linalg.svd(A,compute_uv=False)
    # also basis of S_N-invariants: rank of deltas
    nul=sum(s<1e-8*max(1,s.max()))
    return nul, len(parts), s

def catalan(k): return comb(2*k,k)//(k+1)
