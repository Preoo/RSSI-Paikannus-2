import numpy as np

def lateration(references, node_to_locate, est_dists, skip_refs=[]):
    """ return location (x, y) using lateration with all references """
    using_refs = {key:value for key, value in references.items() if key != node_to_locate and key not in skip_refs}

    x = np.array([loc.x for loc in using_refs.values()])
    y = np.array([loc.y for loc in using_refs.values()])
    d = np.array([est_dists[k] for k in using_refs.keys()])

    xs = [x[i] - x[-1] for i in range(len(x)-1)]
    ys = [y[i] - y[-1] for i in range(len(y)-1)]
    #         |x_0 - x_{n-1}    , y_0     - y_{n-1}|
    # A = -2* |x_1 - x_{n-1}    , y_1     - y_{n-1}|
    #         |x_{n-2} - x_{n-1}, y_{n-2} - y_{n-1}|
    A = np.array((xs, ys)).transpose() # shape (6, 2)

    A *= -2
    #     |d_{0}^2   - d_{n-1}^2 -x_{0}^2   + x_{n-1}^2 - y_{0}^2   + y_{n-1}^2|
    # B = |d_{1}^2   - d_{n-1}^2 -x_{1}^2   + x_{n-1}^2 - y_{1}^2   + y_{n-1}^2|
    #     |d_{n-2}^2 - d_{n-1}^2 -x_{n-2}^2 + x_{n-1}^2 - y_{n-2}^2 + y_{n-1}^2|
    B = np.array( [d[i]**2 - d[-1]**2 - x[i]**2 + x[-1]**2 -y[i]**2 + y[-1]**2 for i in range(len(d)-1 )] )

    # Computes x of Ax = B. Equivalent with (A.T * A)^{-1} * A.T * B.
    # Source: Distributed localization in wireless sensor networks:a quantitative comparison
    x, y = np.linalg.lstsq(A, B)[0]
    return x, y

def minmaxbox(references, node_to_locate, est_dists, skip_refs=[]):
    """ return center of bounding box as location (x, y) """
    using_refs = {key:value for key, value in references.items() if key != node_to_locate and key not in skip_refs}
    return 0, 0