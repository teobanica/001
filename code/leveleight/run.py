#!/usr/bin/env python3
# Level-8 triple-contraction run (Claude and Teo).  Usage:
#   python3 run.py a     (laptop A: rotation sector 0)
#   python3 run.py b     (laptop B: rotation sector 4)
#   python3 run.py test  (quick check, a few minutes)
# Needs numpy and scipy:  python3 -m pip install --user numpy scipy
# Progress is printed and saved in resulta.txt / resultb.txt. Stop any time with Ctrl-C; the file keeps what was done.
import sys, os, subprocess
which = sys.argv[1] if len(sys.argv) > 1 else 'test'
here = os.path.dirname(os.path.abspath(__file__)); os.chdir(here)
block, seed, maxc = {'a': ('0', '101', '100000'), 'b': ('4', '202', '100000'), 'test': ('4', '303', '2')}[which]
out = open('result' + which + '.txt', 'w')
proc = subprocess.Popen([sys.executable, 'batch.py', block, seed, maxc], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    print(line, end='', flush=True); out.write(line); out.flush()
proc.wait()
print('finished; results in', 'result' + which + '.txt')
