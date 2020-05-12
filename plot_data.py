import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from math import ceil

from constants import sensorid_to_locate, skip_ref_ids, locations, build_locations_df
from plot_functions import ecdf

sns.set_style('ticks')
# sns.set_palette('Blues')
try:
    df = pd.read_pickle(f'paikannus_estimaatit_sensorille_{sensorid_to_locate}.plk')
    anchors_df = build_locations_df()
    loc_coords_abs = [abs(coord) for coord in [*anchors_df.x_m, *anchors_df.y_m]]
    xy_limits = ceil(round(max(loc_coords_abs))/10.0)*10

    # all node locations
    list_of_references = [r for r in locations.keys() if r not in [sensorid_to_locate, *skip_ref_ids]]
    fig, (ax_lat, ax_box) = plt.subplots(2, 1, sharex=True, sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0}, subplot_kw={'xlim':[-xy_limits, xy_limits], 'ylim':[-xy_limits, xy_limits], 'aspect':'equal'})

    # mark common nodes in both axes
    for ax in (ax_lat, ax_box):
        ax.plot(anchors_df[anchors_df['node'].isin(skip_ref_ids)].x_m, anchors_df[anchors_df['node'].isin(skip_ref_ids)].y_m, '.', label='ei käytetty')
        ax.plot(anchors_df[anchors_df['node'].isin(list_of_references)].x_m, anchors_df[anchors_df['node'].isin(list_of_references)].y_m, 'd', label='referenssi')
        ax.plot(anchors_df[anchors_df['node'] == sensorid_to_locate].x_m, anchors_df[anchors_df['node'] == sensorid_to_locate].y_m, '*', label='arvioitava')
        ax.label_outer()

    ax_lat.plot(df.lateration_x, df.lateration_y, 'xk', label='arvioitu sijainti (lateraatio)', alpha=.3)
    ax_box.plot(df.minmaxbox_x, df.minmaxbox_y, '+k', label='arvioitu sijainti (minmax)', alpha=.3)

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    handles, labels = [sum(lol, []) for lol in zip(*lines_labels)] # thanks stackoverflow
    # remove dumplicate legend entries. This is project dependent manipulation.
    handles, labels = handles[3:], labels[3:]
    handles.insert(-1, handles.pop(0))
    labels.insert(-1, labels.pop(0))
    fig.legend(handles, labels, loc='upper center', ncol=3)
    plt.show()

    # error for each methods over time
    fig, ax = plt.subplots()
    ax.plot(df.timestamp, df.lateration_error, '-', label='Lateration')
    ax.plot(df.timestamp, df.minmaxbox_error, '-.', label='MinMax (Bounding Box)')
    fig.autofmt_xdate()
    # sns.lineplot(x='timestamp', y='lateration_error', data=df, label='Lateration')
    # sns.lineplot(x='timestamp', y='minmaxbox_error', data=df, label='MinMax (Bounding Box)')
    plt.xlabel('Aika')
    plt.ylabel('Paikannusvirhe (m)')
    plt.legend()
    plt.show()

    # ecdf for both methods
    fig, ax = plt.subplots()
    values1, percentiles1 = ecdf(df.lateration_error)
    ax.plot(values1, percentiles1, '-', label='Lateration')
    values2, percentiles2 = ecdf(df.minmaxbox_error)
    ax.plot(values2, percentiles2, '-.', label='MinMax (Bounding Box)')
    # persentiilit
    ax.hlines(y=0.5, xmin=0, xmax=max(values1.max(), values2.max()), color='r', linestyle=':', alpha=.3, label=f'50% persentiili')
    ax.hlines(y=0.9, xmin=0, xmax=max(values1.max(), values2.max()), color='g', linestyle=':', alpha=.3, label=f'90% persentiili')
    plt.title('ECDF')
    plt.xlabel('Paikannusvirhe (m)')
    plt.ylabel('Persentiilit')
    plt.legend()
    plt.show()

except IOError:
    print(f'Failed to load .csv file. It might be missing, check that file paikannus_estimaatit_sensorille')

else:
    print(df.info())
    print('======== describe location errors ========')
    lateration_error_table = df.lateration_error.describe(percentiles=[.5, .9])
    minmaxbox_error_table = df.minmaxbox_error.describe(percentiles=[.5, .9])
    print(lateration_error_table)
    print(minmaxbox_error_table)
