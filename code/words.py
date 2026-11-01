#!/usr/bin/env python3
# words.py -- Theorem 10.x (exotic words of length <= 16) of "On the maximality conjecture for quantum permutation
# groups" (Claude and Teo), Section 10.5, the reduction of <O_3, O_2^+> = O_3^+ to words in Z_2 * Z_2 * Z_2.
#
#   python3 words.py [kmax]       all even lengths k <= kmax (default 16): counts the exotic words beginning with a,
#                                 and checks that each is killed by a rotation relation (exact, pure combinatorics)
#   python3 words.py 20 two       the same, then a second pass on the words not killed by a single relation (31 Oct:
#                                 240 of them at length 20): a word is killed in two steps if some relation at (p, y)
#                                 has all its other pairwise reducible terms v exotic and killed by a single relation
#                                 (then z(v) = 0 for all of them, hence z(w) = 0).  The surviving words of the first
#                                 pass are also saved to wordsleft.txt, one per line, with the two-step outcome.
#
# Gamma = Z_2 * Z_2 * Z_2 with generators a, b, c; a word is a multi-index in {a,b,c}^k.  phi_d deletes the letter d
# and reduces in Z_2 * Z_2.  A word is pairwise reducible if phi_a(w) = phi_b(w) = phi_c(w) = 1, and exotic if it is
# pairwise reducible but w != 1 in Gamma.  Theorem: for every exotic word w of length k <= 16 there are a position p,
# with x = w_p, and a letter y != x, such that none of the words v obtained from u = w^{p -> y} by switching one more
# letter between x and y is pairwise reducible.  Then the rotation relation of Proposition 10.x at (u, x, y) reads
# z(w) = sum of z(v) over the v, which all vanish, so z(w) = 0.
# Reduction in Z_2 * Z_2 is done through the faithful action on Z by the affine maps y: t -> -t, z: t -> 2 - t
# (D_infinity); reduction in Gamma by a vectorized stack.  Results are written to wordsout.txt as well.
import sys, time, itertools, numpy as np

kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 16
two = 'two' in sys.argv[2:]
log = open('wordsout.txt', 'a')
def out(s):
    print(s, flush=True); log.write(s + '\n'); log.flush()

def trivial_in_free(W):
    """W: (n, k) int8 array of letters 0,1,2.  Returns boolean (n,): the word reduces to 1 in Z_2*Z_2*Z_2."""
    n, k = W.shape; stack = np.full((n, k), -1, dtype=np.int8); top = np.zeros(n, dtype=np.int64); ar = np.arange(n)
    for i in range(k):
        let = W[:, i]
        last = stack[ar, np.maximum(top - 1, 0)]
        pop = (top > 0) & (last == let)
        top = np.where(pop, top - 1, top)
        push = ~pop
        stack[ar[push], top[push]] = let[push]; top = np.where(push, top + 1, top)
    return top == 0

def trivial_projection(W, d):
    """phi_d(w) = 1 in Z_2 * Z_2 (letters != d, in order), via the affine representation of D_infinity on Z."""
    n, k = W.shape; sign = np.ones(n, dtype=np.int64); off = np.zeros(n, dtype=np.int64)
    others = [e for e in range(3) if e != d]
    for i in range(k):
        let = W[:, i]
        # apply the letter on the left of the current map t -> sign*t + off:  y: t -> -t,  z: t -> 2 - t
        isy = let == others[0]; isz = let == others[1]
        sign = np.where(isy | isz, -sign, sign)
        off = np.where(isy, -off, np.where(isz, 2 - off, off))
    return (sign == 1) & (off == 0)

def pairwise_reducible(W):
    return trivial_projection(W, 0) & trivial_projection(W, 1) & trivial_projection(W, 2)

def exotic_words(k, chunk_letters=14):
    """all exotic words of length k beginning with a (letter 0), enumerated in chunks of at most 3^chunk_letters words"""
    free = k - 1; fixed = max(0, free - chunk_letters); out_ = []
    for pre in itertools.product(range(3), repeat=fixed):
        n = 3 ** (free - fixed); W = np.zeros((n, k), dtype=np.int8); W[:, 1:1 + fixed] = pre
        idx = np.arange(n)
        for i in range(k - 1, fixed, -1): W[:, i] = idx % 3; idx //= 3
        out_.append(W[pairwise_reducible(W) & ~trivial_in_free(W)])
    return np.vstack(out_)

def killed(w):
    """position p and letter y such that no other term of the rotation relation at (w^{p->y}, w_p, y) is pairwise reducible"""
    k = len(w)
    for p in range(k):
        x = w[p]
        for y in range(3):
            if y == x: continue
            u = w.copy(); u[p] = y; vs = []
            for q in range(k):
                if q == p: continue
                if u[q] == y: v = u.copy(); v[q] = x; vs.append(v)      # left-hand side terms
                if u[q] == x: v = u.copy(); v[q] = y; vs.append(v)      # right-hand side terms
            if not vs or not pairwise_reducible(np.array(vs, dtype=np.int8)).any(): return p, y
    return None

def killed_two(w):
    """position p and letter y such that every other pairwise reducible term v of the relation at (w^{p->y}, w_p, y)
    is exotic and killed by a single relation; returns (p, y, list of these v) or None"""
    k = len(w)
    for p in range(k):
        x = w[p]
        for y in range(3):
            if y == x: continue
            u = w.copy(); u[p] = y; vs = []
            for q in range(k):
                if q == p: continue
                if u[q] == y: v = u.copy(); v[q] = x; vs.append(v)
                if u[q] == x: v = u.copy(); v[q] = y; vs.append(v)
            V = np.array(vs, dtype=np.int8); left = V[pairwise_reducible(V)]
            if len(left) and trivial_in_free(left).any(): continue
            if all(killed(v) is not None for v in left): return p, y, left
    return None

out(time.strftime('%d %b %H:%M') + f'  words.py, lengths up to {kmax}' + (', with the two-step pass' if two else ''))
for k in range(2, kmax + 1, 2):
    t = time.time(); E = exotic_words(k)
    bad = 0; ex = None; left = []
    for w in E:
        r = killed(w)
        if r is None: bad += 1; ex = w; left.append(w)
    names = 'abc'
    first = ''.join(names[i] for i in E[0]) if len(E) else '-'
    out(f'  length {k}: {len(E)} exotic words beginning with a (e.g. {first}); not killed by a single rotation relation: {bad}'
        f'{"" if bad == 0 else "  e.g. " + "".join(names[i] for i in ex)}  ({time.time()-t:.0f}s)')
    if two and bad:
        t = time.time(); still = 0; ex2 = None
        with open('wordsleft.txt', 'a') as f:
            f.write(time.strftime('%d %b %H:%M') + f'  length {k}: words not killed by a single relation, two-step outcome\n')
            for w in left:
                r = killed_two(w); sw = ''.join(names[i] for i in w)
                if r is None: still += 1; ex2 = sw; f.write(f'{sw}  NOT killed in two steps\n')
                else: f.write(f'{sw}  killed in two steps: p = {r[0] + 1}, y = {names[r[1]]}, through '
                              + ' '.join(''.join(names[i] for i in v) for v in r[2]) + '\n')
        out(f'    two-step pass at length {k}: {bad - still} of {bad} killed in two steps, {still} left'
            f'{"" if still == 0 else "  e.g. " + ex2}  ({time.time()-t:.0f}s)')
out('  (paper: 10, 240, 3654, 45360 exotic words at lengths 10, 12, 14, 16, all killed; lengths 18 and 20 are new)')
