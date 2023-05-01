import numpy as np
import matplotlib.pyplot as plt
from fnal import Dataset

def plot_crate(datasets, labels, metric='rawrms', component='WW19') -> None:
    """
    Plots the desired metric (channel-to-channel for the specified
    component) for each of the input Datasets.

    Parameters
    ----------
    datasets: list[Dataset]
        The Dataset objects to be plotted against each other.
    labels: list[str]
        The labels for each Dataset's entry in the legend.
    metric: str
        The key in the Dataset that specifies the metric.
    component: str or list[str]
        The name of the component(s) to select and plot.

    Returns
    -------
    None.
    """
    figure = plt.figure(figsize=(8,6))
    ax = figure.add_subplot()
    for di, d in enumerate(datasets):
        if isinstance(component, list) and len(component) == len(datasets):
            c = component[di]
            title = ' vs. '.join(component)
        else:
            c = component
            title = component
        selected = d['flange'] == c
        x = 64*d['board'][selected] + d['ch'][selected]
        ax.scatter(x, d['rawrms'][selected], label=labels[di])
    ax.set_xlim(0,576)
    ax.set_ylim(0, 10.0)
    ax.set_xticks([64*i for i in range(10)])
    ax.set_xlabel('Channel Number')
    ax.set_ylabel('RMS [ADC]')
    ax.legend()
    figure.suptitle(title)