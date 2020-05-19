import numpy as np

def ecdf(dist):
    """ dist is numpy 1D-array containing distance errors """
    N = len(dist)
    percentiles = [p/N for p in range(1, N+1)]
    return np.sort(dist), percentiles