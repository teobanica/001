# On the maximality conjecture for quantum permutation groups

**Claude and Teo**, November 2026

The quantum permutation group $S_N^+$ of Wang contains the symmetric group $S_N$, and the maximality conjecture states that there is no quantum group strictly in between. This paper revisits the conjecture with three approaches: quantum subgroups as seeds (group duals, graphs, twists, Hadamard models), the theory of a hypothetical exotic intermediate quantum group, and the level-by-level analysis of the intermediate tensor categories by partitions, which is the main contribution. Among the results:

- the conjecture holds up to level 7, for all $N\ge 6$;
- level 8 is closed computationally for $8\le N\le 37$, and level 9 numerically at $N=9,\dots,13,17,19,23$;
- in the orthogonal case, $\langle O_N,O_N^{-1}\rangle$ agrees with $O_N^+$ up to level $\max(20,2N)$, and every $O_N^*\subset G\subsetneq O_N^+$ differing from $O_N^+$ at some level $\le 10$ is $O_N^*$;
- projective versions: a quantum group $PU_N\subset H\subset PO_N^+$ differing from $PO_N^+$ at some level $\le 10$ coincides with $PU_N$ through 12 points.

## Contents

- `001.pdf` — the paper (68 pages)
- `001.tex` — its LaTeX source (self-contained; compile with `pdflatex`, twice)
- `code/` — every script behind the computational statements, with their output logs. `code/README.txt` maps the scripts to the sections and statements of the paper.

## Running the code

Python 3 with `numpy` and `scipy`. Each script explains its usage in its first lines (run it without arguments, or with `test`), and writes a log file next to itself. Exact computations use rational arithmetic or arithmetic modulo large primes; statements labeled "numerical" in the paper rest on floating-point certificates. The longest jobs (levels 8 and 9) ran 7–8 hours per value of $N$ on a laptop, and are resumable.
