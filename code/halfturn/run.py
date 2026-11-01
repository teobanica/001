#!/usr/bin/env python3
# Level-8 half-turn theorem, one N at a time (Claude and Teo).
#   python3 run.py a      laptop A: N = 12, 14, ..., 30   (two workers)
#   python3 run.py b      laptop B: N = 13, 15, ..., 31   (two workers)
#   python3 run.py a 1    same, but with ONE worker (half the memory; use this while Firefox is open)
#   python3 run.py test   quick setup check (about 2 minutes)
# Needs numpy and scipy. Results go to resulta.txt / resultb.txt (one RESULT line per N). Stop any time with Ctrl-C.
import sys, os, subprocess, time
here = os.path.dirname(os.path.abspath(__file__)); os.chdir(here)
which = sys.argv[1] if len(sys.argv) > 1 else 'test'
if which == 'test':
    import numpy, scipy
    from engine import kernel_basis
    t = time.time(); crk, W, free = kernel_basis(8, 11, 1048433)
    print(f'numpy {numpy.__version__}, scipy {scipy.__version__}; level-8 kernel mod p: dim {W.shape[1]} in {time.time()-t:.0f}s')
    print('Setup OK' if W.shape[1] == 66 else 'UNEXPECTED dimension, tell Claude'); sys.exit()
Ns = list(range(12, 31, 2)) if which == 'a' else list(range(13, 32, 2))
nw = int(sys.argv[2]) if len(sys.argv) > 2 else 2
lists = [Ns[i::nw] for i in range(nw)]
out = open('result' + which + '.txt', 'a')
env = dict(os.environ, OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', VECLIB_MAXIMUM_THREADS='3')
procs = [subprocess.Popen([sys.executable, 'job.py'] + [str(n) for n in L], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, env=env) for L in lists]
import threading
def pump(pr, tag):
    for line in pr.stdout:
        line = f'[worker {tag}] ' + line
        print(line, end='', flush=True); out.write(line); out.flush()
th = [threading.Thread(target=pump, args=(pr, i + 1)) for i, pr in enumerate(procs)]
for t in th: t.start()
for t in th: t.join()
print('all done; results in result' + which + '.txt')
