import numpy as np

n = 10

indices = np.arange(n)

pairs = np.array(list(zip(*np.triu_indices(n, k=1))))

print(pairs)