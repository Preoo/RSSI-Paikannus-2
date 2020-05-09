import numpy as np

def lateration(references, node_to_locate, est_dists, skip_refs=[]):
    """ return location (x, y) using lateration with all references """
    # filter out measuremnts from refrences specified in skip_refs list.
    # Useful if fading or other causes are contributing to weird estimates.
    anchors = {key:value for key, value in references.items() if key != node_to_locate and key not in skip_refs}

    x = np.array([loc.x for loc in anchors.values()])
    y = np.array([loc.y for loc in anchors.values()])
    d = np.array([est_dists[k] for k in anchors.keys()])

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
    x, y = np.linalg.lstsq(A, B, rcond=None)[0]
    return x, y

def minmaxbox(references, node_to_locate, est_dists, skip_refs=[]):
    """ return center of bounding box as location (x, y) """
    anchors = {key:value for key, value in references.items() if key != node_to_locate and key not in skip_refs}
    
    # dataset isn't too large so naive implementation should be fine
    x_min = []
    x_max = []
    y_min = []
    y_max = []
    # for all anchors in using_refs
    for neighbor, loc in anchors.items():
    #   calculate bounding box
        x_min.append(loc.x - est_dists[neighbor])
        x_max.append(loc.x + est_dists[neighbor])
        y_min.append(loc.y - est_dists[neighbor])
        y_max.append(loc.y + est_dists[neighbor])

    x = (max(x_min) + min(x_max)) / 2
    y = (max(y_min) + min(y_max)) / 2

    return x, y