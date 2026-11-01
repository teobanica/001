import pickle, io, contextlib, sys
import numpy as np
from fractions import Fraction
with contextlib.redirect_stdout(io.StringIO()):
    from levquad import Kbasis, contract
from levlin import noncrossing, canon
N=int(sys.argv[1]) if len(sys.argv)>1 else 7
Q,sym,vecs=pickle.load(open(f'quad_N{N}.pkl','rb'))
_,crP,K=Kbasis(N); cidx={P:a for a,P in enumerate(crP)}
piv=[next(i for i,x in enumerate(row) if x!=0) for row in K]
def coordsK(vec):          # crossing part -> (coords in K basis, residual norm)
    v=np.zeros(len(crP),complex)
    for P,c in vec.items():
        if P in cidx: v[cidx[P]]+=complex(c)
    Kf=np.array([[float(x) for x in r] for r in K])
    x=np.array([v[p] for p in piv]); res=v-x@Kf
    return x, np.linalg.norm(res)/(np.linalg.norm(v)+1e-300)
rot=lambda P: canon([set((x+1)%6 for x in b) for b in P])
refl=lambda P: canon([set((-x)%6 for x in b) for b in P])
R=np.array([coordsK({rot(P):c for P,c in v.items()})[0] for v in vecs]).T
S=np.array([coordsK({refl(P):c for P,c in v.items()})[0] for v in vecs]).T
A=np.array([[float(x) for x in r] for r in Q])
def F(c): return np.array([sum(A[k,i]*c[a]*c[b] for i,(a,b) in enumerate(sym)) for k in range(len(A))])
def J(c):
    Jm=np.zeros((len(A),5),complex)
    for k in range(len(A)):
        for i,(a,b) in enumerate(sym): Jm[k,a]+=A[k,i]*c[b]; Jm[k,b]+=A[k,i]*c[a]
    return Jm
rng=np.random.default_rng(1); sols=[]
for trial in range(int(sys.argv[2]) if len(sys.argv)>2 else 300):
    a=rng.normal(size=5)+1j*rng.normal(size=5); c=rng.normal(size=5)+1j*rng.normal(size=5); c/=a@c
    for it in range(100):
        r=np.concatenate([F(c),[a@c-1]]); dc=np.linalg.lstsq(np.vstack([J(c),a]),-r,rcond=None)[0]; c=c+dc
        if np.linalg.norm(dc)<1e-14: break
    if np.linalg.norm(F(c))<1e-10:
        c=c/np.linalg.norm(c); k=np.argmax(abs(c)); c=c*abs(c[k])/c[k]
        if not any(np.linalg.norm(c-s)<1e-6 for s in sols): sols.append(c)
print(f'N={N}: {len(sols)} isolated solutions of the level<=5 quadrics')
def vec_of(c): 
    out={}
    for ci,v in zip(c,vecs):
        for P,x in v.items(): out[P]=out.get(P,0)+ci*float(x)
    return out
for si,s in enumerate(sols):
    Rs=R@s; lam=(np.conj(s)@Rs); Ss=S@s; mu=np.conj(s)@Ss
    print(f' sol {si}: rotation eigen? {np.linalg.norm(Rs-lam*s)<1e-8} (eigenvalue {np.round(lam,4)}), reflection eigen? {np.linalg.norm(Ss-mu*s)<1e-8} ({np.round(mu,4)})')
    # W = span of D6-orbit of s
    orb=[s]; M=np.eye(5)
    for g in [R,S]:
        pass
    span=[np.linalg.matrix_power(R,i)@s for i in range(6)]+[np.linalg.matrix_power(R,i)@S@s for i in range(6)]
    U,sv,_=np.linalg.svd(np.array(span).T); W=U[:,:int((sv>1e-8).sum())]
    print(f'   dim W (orbit span) = {W.shape[1]}')
    # level-6 test: contractions of xi x eta, xi,eta in W basis, land in NC + W ?
    Wv=[vec_of(W[:,i]) for i in range(W.shape[1])]
    worst=0; tests=0; scale=0
    for a in range(len(Wv)):
        for b in range(len(Wv)):
            for st in range(6):
                for t in range(6):
                    for j,o1,o2 in [(3,0,0),(4,1,1)]:
                        _,r=contract(Wv[a],Wv[b],st,t,j,o1,o2,N); tests+=1
                        v=np.zeros(len(crP),complex)
                        for P,c in r.items():
                            if P in cidx: v[cidx[P]]+=c
                        Kf=np.array([[float(x) for x in rr] for rr in K])
                        x=np.array([v[p] for p in piv]); outK=np.linalg.norm(v-x@Kf)
                        proj=W@(np.conj(W.T)@x); outW=np.linalg.norm(x-proj)
                        worst=max(worst,outK+outW); scale=max(scale,np.linalg.norm(v))
    print(f'   level-6 closure: {tests} contractions, absolute violation {worst:.3e}, typical output size {scale:.3e}')
