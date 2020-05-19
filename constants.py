import pandas as pd
from collections import namedtuple
from typing import List
Location = namedtuple('Location', ['x', 'y'])

# constants
P0:float = -28.0
n_p:int = 2
d0:float = 1.0      # reference distance in meters
X_std:float = None  # specify a constant or None to calculate it from rssi-measurements

sensorid_to_locate:int = 6
skip_ref_ids:List[int] = [4,5]

locations = {
    1: Location(41.99, 9.35),   #reference
    2: Location(0.09, -21.14),  #reference
    4: Location(-34.12, 16.95), #reference
    5: Location(3.7, 21.4),     #reference
    6: Location(2.47, -5.62),   #possible/reference
    7: Location(-17.51, -19.68),#reference
    8: Location(-15.91, 5.63),  #possible/reference
    9: Location(19.3, -6.89),   #reference
}

def build_locations_df():
    # just do the dumb thing and be done with it
    nodes:List[int] = []
    X:List[float] = []
    Y:List[float] = []

    for node, loc in locations.items():
        nodes.append(node)
        X.append(loc.x)
        Y.append(loc.y)

    return pd.DataFrame({
        'node' : nodes,
        'x_m' : X,
        'y_m' : Y
    })
