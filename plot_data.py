import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from constants import P0, n_p, d0, X_std, sensorid_to_locate, skip_ref_ids, Location, locations, build_locations_df

from plot_functions import plot_node_loc

def run():
    try:
        # df = pd.read_csv(f'paikannus_estimaatit_sensorille_{sensorid_to_locate}.csv', parse_dates=['timestamp'])
        df = pd.read_pickle(f'paikannus_estimaatit_sensorille_{sensorid_to_locate}.plk')
        
    except Exception as e:
        print(f'Failed to load .csv file. It might be missing, check that file paikannus_estimaatit_sensorille')
        print(f'Exception: {e}')
    else:
        print(df.info())
        print(df.head())

if __name__ == "__main__":
    run()