import pickle
from locustwo import norms_at
res = {N: norms_at(N) for N in range(6, 96)}
pickle.dump(res, open('locustwo.pkl', 'wb')); print('done')
