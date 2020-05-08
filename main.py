from collections import namedtuple
import numpy as np

# constants
P0:int = -28
n_p:int = 2
d0:int = 1

# reference lcoations for nodes
#   sensorid : (x, y)
Location = namedtuple('Location', ['x', 'y'])
locations = {
    1: Location(41.99, 9.35),
    2: Location(0.09, -21.14),
    4: Location(-34.12, 16.95),
    5: Location(3.7, 21.4),
    6: Location(2.47, -5.62),
    7: Location(-17.51, -19.68),
    8: Location(-15.91, 5.63),
    9: Location(19.3, -6.89),
}

def estimated_dist(rssi:float, X:float) -> float:
    # rssi = P0 - 10*n*log_10(d) + X_q
    # rssi - P0 - X_q = -10*n*log_10(d)
    # (rssi - P0 - X_q)/(-10*n) = log_10(d)
    # 10^[(rssi - P0 - X_q)/(-10*n)] = d_hat
    # 10^((rssi - P0 - X_q)/-20) = d_hat

    # X_q is gaussian with 0 mean and std of q
    # X = np.random.normal(0, q, len(samples))
    # input X is gaussian sampled for this element-wise call
    return 0.0