from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# constants
P0:int = -28
n_p:int = 2
d0:int = 1
X_std:float = 0.1

sensorid_to_locate = 6

# reference lcoations for nodes
#   sensorid : (x, y)
Location = namedtuple('Location', ['x', 'y'])
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

# just do the dumb thing and be done with it
nodes = []
X = []
Y = []

for node, loc in locations.items():
    nodes.append(node)
    X.append(loc.x)
    Y.append(loc.y)

location_df = pd.DataFrame({
    'node' : nodes,
    'x_m' : X,
    'y_m' : Y
})

from plot_data import plot_node_loc
plot_node_loc(location_df)

# read csv and drop rows where sensorid is not our chosen node
main_df = pd.read_csv('prosessoitu_mittausdata.csv', parse_dates=['timestamp'])
main_df = main_df[main_df.sensorid == sensorid_to_locate]
main_df = main_df.drop(['sensorid'], axis=1)
# print(main_df.info())
# print(main_df.head())

# groupby neighbor and take hourly mean of rssi1
hourly_df = main_df.groupby(['neighbor'], as_index=False).resample('1H', on='timestamp').mean()

print(hourly_df.info())
# print(hourly_df.head())
# print(hourly_df.tail())
print('=====missing======')
print(hourly_df[hourly_df.isna().any(axis=1)])
print('-----------------')

# backfill missing datapoints based on latest values
hourly_df = hourly_df.fillna(method='bfill')
print(hourly_df.info())

# get sampling of 0 means gaussians dist with some std
X = np.random.normal(0, X_std, len(hourly_df))


def estimated_dist(rssi:float, X:float) -> float:
    # rssi = P0 - 10*n*log_10(d) + X_q
    # rssi - P0 - X_q = -10*n*log_10(d)
    # (rssi - P0 - X_q)/(-10*n) = log_10(d)
    # 10^[(rssi - P0 - X_q)/(-10*n)] = d_hat
    # 10^((rssi - P0 - X_q)/-20) = d_hat

    # X_q is gaussian with 0 mean and std of q
    # X = np.random.normal(0, q, len(samples))
    # input X is gaussian sampled for this element-wise call
    return 10**((rssi - P0 - X)/(-10*n_p))

# apply estimate function to generate new column in dataframe for d_hat
v_estimated_dist = np.vectorize(estimated_dist)

hourly_df['d_hat'] = v_estimated_dist(hourly_df.rssi1, X)

print(hourly_df.head())


def run():
    pass