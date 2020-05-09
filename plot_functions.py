import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_node_loc(plot_df):
    sns.set_style('dark')
    # sns.set_context('notebook')
    sns.relplot(x='x_m', y='y_m', hue='node', style='node', palette='muted', legend='full', data=plot_df).set(xlim=(-50,50), ylim=(-50,50))
    plt.show()