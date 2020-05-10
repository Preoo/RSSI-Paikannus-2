import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_node_loc(plot_df):
    sns.set_style('dark')
    # sns.set_context('notebook')
    sns.relplot(x='x_m', y='y_m', hue='node', style='node', palette='muted', legend='full', data=plot_df).set(xlim=(-50,50), ylim=(-50,50))
    plt.show()

def plot_error_distance_over_time(df):
    #plt.plot(df.timestamp, df.lateration_error)
    #plt.plot(df.timestamp, df.minmaxbox_error)
    pass

def plot_error_distance_ecdf(df):
    pass

def describe_error_distance(df):
    # grab them the easy way by calling
    # table_df = df.lateration_error.describe(percentiles=[.5, .9])
    pass