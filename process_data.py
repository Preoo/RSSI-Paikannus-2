import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# from plot_functions import plot_node_loc
from estimation_methods import lateration, minmaxbox
from constants import P0, n_p, d0, X_std, sensorid_to_locate, skip_ref_ids, Location, locations, build_locations_df

# read csv and drop rows where sensorid is not our chosen node
main_df = pd.read_csv('prosessoitu_mittausdata.csv', parse_dates=['timestamp'])
main_df = main_df[main_df.sensorid == sensorid_to_locate]
main_df = main_df.drop(['sensorid'], axis=1)

# groupby neighbor and take hourly mean of rssi1
hourly_df = main_df.groupby(['neighbor'], as_index=False).resample('1H', on='timestamp').mean()

print(hourly_df.info())
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
print(hourly_df.describe())

print('=====groups=====')
hourly_df = hourly_df.reset_index()
hourly_df = hourly_df.set_index(['timestamp', 'neighbor'])
hourly_df = hourly_df.drop(['level_0'], axis=1)

# calculate location estimates for timestamps
locations_estimates:list = []
for n, g in hourly_df.groupby(['timestamp']):
    # save estimated locations as np.arrays to ease working with saved data in next step
    # locations are defined as coord-paris [x, y]
    neighbors = [d for _, d in g.index.values]
    neighbor_dist_est = dict(zip(neighbors, g.d_hat))
    locations_estimates.append({
        'timestamp' : str(n),
        'lateration' : np.array((lateration(locations, sensorid_to_locate, neighbor_dist_est, skip_refs=skip_ref_ids)), dtype=np.float),
        'minmaxbox' : np.array((minmaxbox(locations, sensorid_to_locate, neighbor_dist_est, skip_refs=skip_ref_ids)), dtype=np.float)
    })

# columns lateration and minmaxbox contains tuple of estimated Location from said method for sensorid_to_locate
position_df = pd.DataFrame(locations_estimates)
position_df.timestamp = pd.to_datetime(position_df.timestamp) # need to convert to datetime after creation

# calculate error for each method
# d(P, Q) = sqrt((x2 − x1)**2 + (y2 − y1)**2)
def error_distance(a:Location, b:Location) -> float:
    x1, y1 = a
    x2, y2 = b
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

position_df['lateration_error'] = position_df['lateration'].apply(error_distance, args=(locations[sensorid_to_locate],))
position_df['minmaxbox_error'] = position_df['minmaxbox'].apply(error_distance, args=(locations[sensorid_to_locate],))

# just brute force this...
# we need X, Y in separate columns for plotting
lat_X = np.empty(len(position_df.lateration))
lat_Y = np.empty(len(position_df.lateration))
for i, v in enumerate(position_df.lateration):
    lat_X[i] = v[0]
    lat_Y[i] = v[1]
position_df['lateration_x'] = lat_X
position_df['lateration_y'] = lat_Y

box_X = np.empty(len(position_df.minmaxbox))
box_Y = np.empty(len(position_df.minmaxbox))
for i, v in enumerate(position_df.minmaxbox):
    box_X[i] = v[0]
    box_Y[i] = v[1]
position_df['minmaxbox_x'] = box_X
position_df['minmaxbox_y'] = box_Y

print(position_df.info())
print(position_df.head())

# position_df.to_csv(f'paikannus_estimaatit_sensorille_{sensorid_to_locate}.csv', index=False)
position_df.to_pickle(f'paikannus_estimaatit_sensorille_{sensorid_to_locate}.plk')
