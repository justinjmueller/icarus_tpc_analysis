import numpy as np
import matplotlib.pyplot as plt
from fnal import Dataset

def plot_tpc(datasets, labels, metric='rawrms', tpc=0) -> None:
    """
    Plots the desired metric (channel-to-channel by wire plane) for
    the specified TPC.

    Parameters
    ----------
    datasets: list[Dataset]
        The Dataset objects to be plotted against each other.
    labels: list[str]
        The labels for each Dataset's entry in the legend.
    metric: str
        The key in the Dataset that specifies the metric.
    tpc: int or list[int]
        The index of the tpc(s) to select and plot.
    """
    plt.style.use('plot_style.mplstyle')
    figure = plt.figure(figsize=(26,16))
    gspec = figure.add_gridspec(3,8)
    saxs = [figure.add_subplot(gspec[i,0:6]) for i in [0,1,2]]
    haxs = [figure.add_subplot(gspec[i,6:8]) for i in [0,1,2]]
    planes = ['Induction 1', 'Induction 2', 'Collection']
    nchannels = [2304, 5760, 5760]
    ndivs = [288, 576, 576]

    for pi, p in enumerate(planes):
        for di, d in enumerate(datasets):
            mask = ((d['tpc'] == tpc) & (d['plane'] == pi))

            saxs[pi].scatter(d['channel_id'][mask], d[metric][mask], label=labels[di])
            xlow = 13824*tpc + sum(nchannels[:pi])
            xhigh = xlow + nchannels[pi]
            saxs[pi].set_xlim(xlow,xhigh)
            saxs[pi].set_ylim(0,10)
            saxs[pi].set_xticks(np.arange(xlow, xhigh+ndivs[pi], ndivs[pi]))
            saxs[pi].set_xlabel('Channel ID')
            saxs[pi].set_ylabel('RMS [ADC]')
            saxs[pi].set_title(p)

            haxs[pi].hist(d[metric][mask], range=(0,10), bins=50,
                          label=labels[di], histtype='step')
            haxs[pi].legend()
            haxs[pi].set_xlim(0,10)
            haxs[pi].set_xlabel('RMS [ADC]')
            haxs[pi].set_ylabel('Entries')

def plot_crate(datasets, labels, metric='rawrms', component='WW19') -> None:
    """
    Plots the desired metric (channel-to-channel for the specified
    component) for each of the input Datasets. A Dataset can also
    be an INFNDataset.

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
    plt.style.use('plot_style.mplstyle')
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

def plot_waveform(waveform, title) -> None:
    """
    Plot a single waveform over its full range.

    Parameters
    ----------
    waveform: np.array
        The input waveform to plot. Assumed shape of (4096,).
    title: str
        The title to be displayed at the top of the plot.

    Returns
    -------
    None.
    """
    plt.style.use('plot_style.mplstyle')
    figure = plt.figure(figsize=(14,6))
    ax = figure.add_subplot()
    ax.plot(np.arange(4096), waveform, linestyle='-', linewidth=1)
    ax.set_xlim(0,4096)
    ax.set_ylim(-20, 20)
    ax.set_xlabel('Time [ticks]')
    ax.set_ylabel('Waveform Height [ADC]')
    figure.suptitle(title)