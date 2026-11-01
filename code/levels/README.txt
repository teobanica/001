Sections 7-9: the uniform-in-N chain (levels 5 to 8), server side.  Python 3, numpy, scipy; exact rational and modular
arithmetic.  Generated file names use the level in unary ("x" * k), e.g. stepone_xxxxxxx.pkl: no digits in file names.
Helpers: levlin.py (partitions, leg-sums, merges), levquad.py / levquadrics.py (level-6 kernel and ribbon contractions,
exact), levnull.py (exact nullspace), hm.py (partition vectors), poly.py (polynomials in N over Q), ribbon.py and
engine.py (ribbon contractions, modular), triple8.py / triple.py (triple contractions).

Level 5     level5.py          K_5(N) = 0: det M_5(N) = -(N-4)(N^2-3N+1)^2
Level 6     step1sym.py        the kernel K_6(N) with N symbolic (5 polynomial vectors, the 66 x 66 minor)
            levquadrics.py N   the 792 quadrics at integer N, exact (quad_N{N}.pkl)
            elim.py N          Hilbert function, Gotzmann, characteristic polynomial (the 4 points)
            irr.py N           irreducibility of the characteristic polynomial
            design.py, step2sel.py, cert.py     the certificates (C2) finiteness and (C3) closure, N symbolic
            closex.py N        closure in the quartic field (the 4 Galois-conjugate lines)
            locus.py, locustwo.py, runtwo.py, pade.py, padeq.py, roots.py, zeroset.py   the survival locus {0, 2, 5}
            triple6.py         triple contractions at level 6, N = 11
Level 7     stepone.py 7, stepone_rec.py 7     the kernel K_7(N), N symbolic (14 vectors, denominator delta(N))
            select.py, exactrows.py            the 105 quadric rows, as polynomials in N
            dmod.py, runprimes.py, mkjob.py    the determinant D_7(N) and Delta_7(N) modulo primes (laptop job in ../levelseven)
            crt.py, dfinal.py                  Chinese remaindering with the Hadamard bound, integer roots
            nsix.py, nsixexact.py              the case N = 6, modulo the Moebius relation
Level 8     testengine.py, dimk.py, dimk8.py   dim K_8 = 66
            modforms.py, realbasis.py, realforms.py    the quadric span (1154 of 2211), modular and real
            cubic.py, cubicplus.py, prep.py, tripbatch.py, batchlap.py, subsearch.py   degree-3 tests by rotation character
            halfturn.py, halfseven.py          the half-turn theorem (see also ../halfturn for the laptop version)
            ncpartner.py                       the redundancy of the noncrossing-partner constraints (Remark 7.x)
