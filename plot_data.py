import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from constants import P0, n_p, d0, X_std, sensorid_to_locate, skip_ref_ids, Location, locations, build_locations_df

from plot_functions import ecdf

sns.set_style('white')

def run():
    try:
        df = pd.read_pickle(f'paikannus_estimaatit_sensorille_{sensorid_to_locate}.plk')
        anchors_df = build_locations_df()
        # all node locations
        list_of_references = [r for r in locations.keys() if r not in [sensorid_to_locate, *skip_ref_ids]]
        fig, (ax_lat, ax_box) = plt.subplots(2, 1, sharex=True, sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0})

        for ax in (ax_lat, ax_box):
            ax.plot(anchors_df[anchors_df['node'].isin(skip_ref_ids)].x_m, anchors_df[anchors_df['node'].isin(skip_ref_ids)].y_m, '.', label='ei käytetty')
            ax.plot(anchors_df[anchors_df['node'].isin(list_of_references)].x_m, anchors_df[anchors_df['node'].isin(list_of_references)].y_m, 'd', label='referenssi')
            ax.plot(anchors_df[anchors_df['node'] == sensorid_to_locate].x_m, anchors_df[anchors_df['node'] == sensorid_to_locate].y_m, '*', label='arvioitava')
            ax.label_outer()


        ax_lat.plot(df.lateration_x, df.lateration_y, 'xc', label='arvioitu sijainti (lateraatio)', alpha=.3)
        ax_box.plot(df.minmaxbox_x, df.minmaxbox_y, '+c', label='arvioitu sijainti (minmax)', alpha=.3)
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        handles, labels = [sum(lol, []) for lol in zip(*lines_labels)] # thanks stackoverflow

        handles, labels = handles[3:], labels[3:]
        handles.insert(-1, handles.pop(0))
        labels.insert(-1, labels.pop(0))
        fig.legend(handles, labels, loc='upper center', ncol=3)
        plt.show()

        # error for each methods over time
        sns.lineplot(x='timestamp', y='lateration_error', data=df, label='Lateration')
        sns.lineplot(x='timestamp', y='minmaxbox_error', data=df, label='MinMax (Bounding Box)')
        plt.xlabel('Aika')
        plt.ylabel('Paikannusvirhe (m)')
        plt.show()

        # ecdf for both methods
        fig, ax = plt.subplots()
        values1, percentiles1 = ecdf(df.lateration_error)
        ax.plot(values1, percentiles1, label='Lateration')
        values2, percentiles2 = ecdf(df.minmaxbox_error)
        ax.plot(values2, percentiles2, label='MinMax (Bounding Box)')
        # persentiilit
        ax.hlines(y=0.5, xmin=0, xmax=max(values1.max(), values2.max()), color='r', linestyle='--', alpha=.15, label=f'50% persentiili')
        ax.hlines(y=0.9, xmin=0, xmax=max(values1.max(), values2.max()), color='g', linestyle='--', alpha=.15, label=f'90% persentiili')
        plt.title('ECDF')
        plt.xlabel('Paikannusvirhe (m)')
        plt.ylabel('Persentiilit')
        plt.legend()
        plt.show()

    except Exception as e:
        print(f'Failed to load .csv file. It might be missing, check that file paikannus_estimaatit_sensorille')
        print(f'Exception: {e}')
        raise e
    else:
        print(df.info())
        print('======== describe location errors ========')
        print(df.lateration_error.describe(percentiles=[.5, .9]))
        print(df.minmaxbox_error.describe(percentiles=[.5, .9]))
        # print(df.head())

if __name__ == "__main__":
    run()