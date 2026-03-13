import numpy as np

def swish(x):
    x = np.array(x)
    return x * (1 / (1 + np.exp(-x)))

print(swish([0,1,-1,3]))