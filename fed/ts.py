import numpy as np
import math

def frange_cycle_cosine_adjusted(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch / n_cycle
    step = 1 / (period * ratio)  # step in [0,1] range for cosine
    
    for c in range(n_cycle):
        v, i = 0, 0  # v will range from 0 to 1 for the cosine increase phase
        while v <= 1:
            L[int(i + c * period)] = start + (stop - start) * (0.5 - 0.5 * np.cos(v * np.pi))
            v += step
            i += 1
        
        # After cosine increase phase, keep the value at `stop` for the rest of the period
        while i < period:
            L[int(i + c * period)] = stop
            i += 1
    
    return L


L = frange_cycle_cosine_adjusted(start=0.001, stop=0.1, n_epoch = 500)

print( L)