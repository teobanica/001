#!/usr/bin/env python3
# nine.py -- level 9 of S_N by positivity certificates, at a given list of N, on one laptop (Claude and Teo).
#
#   python3 nine.py b [hours]      laptop B: N = 13, then N = 17 (four workers; stops starting new work after `hours`)
#   python3 nine.py c              N = 11, 12, 10, 9      (second round, after a and b were done)
#   python3 nine.py d              N = 14, 15, 16, 18
#   python3 nine.py e              N = 20, 21, 22, 24
#   python3 nine.py test           quick check (level-9 kernel at N = 11, about 4 minutes)
#   python3 nine.py one N          a single value of N
#
# Put nine.py together with engine.py, ninekernel.py, nineforms.py, ninepositivity.py in one folder.  Everything is written
# to  nineout/<N spelled out>/  and a running log is kept in  nineout/progress.txt  (keep that file, and the files
# ninecert_*.npy if you can).  Resumable: run the same command again and it continues where it was.
#
# For each N: (1) ninekernel.py: K_9(N) by rotation characters, with the J-real structure (minutes);
#             (2) nineforms.py: the 90 ribbon configurations, quadrics restricted to the J-real slices of the nine blocks
#                 (the long part, about 2 hours with four workers);
#             (3) ninepositivity.py: one small semidefinite program per block; "POSITIVE DEFINITE: block excluded" is the
#                 result we want for every block, in particular for every character j != 0.
import sys, os, time, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
WORKERS = 4
WORDS = {9: 'nine', 10: 'ten', 11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen', 15: 'fifteen', 16: 'sixteen',
         17: 'seventeen', 18: 'eighteen', 19: 'nineteen', 20: 'twenty', 21: 'twentyone', 22: 'twentytwo', 23: 'twentythree',
         24: 'twentyfour'}
os.makedirs('nineout', exist_ok=True)
def log(line):
    s = time.strftime('%d %b %H:%M  ') + line
    print(s, flush=True); open(os.path.join('nineout', 'progress.txt'), 'a').write(s + '\n')
def run_logged(cmd, logfile):
    with open(logfile, 'a') as lf:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout: print(line, end='', flush=True); lf.write(line); lf.flush()
        p.wait(); return p.returncode
def do_N(N, t_end):
    out = os.path.join('nineout', WORDS.get(N, 'n' + 'x' * N)); os.makedirs(out, exist_ok=True)
    t = time.time()
    if not os.path.exists(os.path.join(out, 'ninekernel.npz')):
        log(f'N={N}: kernel by rotation characters, start')
        run_logged([sys.executable, 'ninekernel.py', str(N), out], os.path.join(out, 'kernel.log'))
        log(f'N={N}: kernel done ({(time.time()-t)/60:.1f} min)')
    if time.time() > t_end: log('time is up before the quadrics; run again to continue'); return
    log(f'N={N}: quadrics, {WORKERS} workers, start')
    procs = [subprocess.Popen([sys.executable, 'nineforms.py', str(w), str(WORKERS), out],
                              stdout=open(os.path.join(out, f'forms_{"abcdefgh"[w]}.log'), 'a'), stderr=subprocess.STDOUT) for w in range(WORKERS)]
    while any(p.poll() is None for p in procs):
        time.sleep(60)
        if time.time() > t_end + 3600:            # an hour of grace for the quadrics in progress, then stop them (resumable)
            for p in procs: p.terminate()
            log('time is up: quadric workers stopped (state saved); run again to continue'); return
    log(f'N={N}: quadrics done ({(time.time()-t)/60:.1f} min)')
    log(f'N={N}: certificates')
    run_logged([sys.executable, 'ninepositivity.py', out], os.path.join(out, 'positivity.log'))
    for line in open(os.path.join(out, 'positivity.log')).read().splitlines()[-12:]:
        if 'character' in line or 'BLOCKS' in line or 'blocks' in line: log(f'N={N}: ' + line.strip())
if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'test'
    if what == 'test':
        import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)
        run_logged([sys.executable, 'ninekernel.py', '11', os.path.join('nineout', 'test')], os.path.join('nineout', 'test.log'))
        print('Setup OK if the line above says dim K_9 = 307 by character [37, 33, 33, 36, 33, 33, 36, 33, 33]'); sys.exit()
    hours = float(sys.argv[2]) if len(sys.argv) > 2 and what != 'one' else 8.0
    Ns = {'b': [13, 17], 'a': [19, 23], 'c': [11, 12, 10, 9], 'd': [14, 15, 16, 18], 'e': [20, 21, 22, 24]}.get(what) or ([int(sys.argv[2])] if what == 'one' else None)
    if Ns is None: print(__doc__); sys.exit()
    t_end = time.time() + hours * 3600
    log(f'=== nine {what}: N in {Ns}, {WORKERS} workers, stops after {hours} h ===')
    for N in Ns:
        if time.time() > t_end: log('time is up'); break
        do_N(N, t_end)
    log(f'=== nine {what}: done ===')
