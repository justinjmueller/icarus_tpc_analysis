import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as colors
from matplotlib import cm as cm
from fnal import Dataset
import sqlite3
import sys

sys.path.append('..')
from globals import *

def plot_ffts(datasets, labels, fft_type='raw', tpc=None) -> None:
    """
    Plots the desired FFTs for the specifed TPC as a set of 1D scatterplots.

    Parameters
    ----------
    datasets: list(Dataset)
        The Datasets to be plotted.
    labels: list(str)
        The labels for each Dataset.
    fft_type: str
        The type of noise FFT to plot. Options: 'raw', 'int', 'coh'
    tpc: int
        The index of the TPC to select and plot.

    Returns
    -------
    None.
    """
    plt.style.use(PLOT_STYLE)
    figure = plt.figure(figsize=(9,7.5))
    planes = ['Induction 1', 'Induction 2', 'Collection']
    frequency = 0.610351563*np.arange(2049)    
    gspec = figure.add_gridspec(22,22)
    axs = [figure.add_subplot(gspec[7*p:7*(p+1),1:22]) for p in [0,1,2]]
    fft_type = {'raw': 0, 'int': 1, 'coh': 2}[fft_type]
    fft_label = ['Raw', 'Intrinsic', 'Coherent'][fft_type]
    for p in [0,1,2]:
        for di, d in enumerate(datasets):
            fft = d.get_ffts_plane(p)[fft_type, :]
            mask = ~np.isnan(fft)
            axs[p].scatter(frequency[mask], fft[mask]/1000, s=3, label=labels[di])
            axs[p].set_title(planes[p])
            if p < 2: axs[p].tick_params('x', labelbottom=False)
            if p == 0: axs[p].legend(markerscale=4)
            axs[p].set_xlim(0, 500)
            axs[p].set_ylim(0, 1.2)
            axs[p].set_yticks(0.3*np.arange(5))
        
    figure.text(0.55, 0.01, 'Frequency [kHz]', ha='center', fontsize=18)
    figure.text(0.01, 0.5, 'Power [$\mathrm{ADC^2}$/kHz]', va='center', rotation='vertical', fontsize=18)

def plot_planes_new(datasets, labels, metrics, title, tpc=None, normalize=False) -> None:
    """
    Plots the desired metric(s) for the specified TPC and dataset(s)
    as a set of three 1D histograms.

    Parameters
    ----------
    datasets: list[Dataset]
        The Dataset objects to be plotted.
    labels: str
        The key in the Dataset that specifies the metric.
    metrics: list[str]
        The metrics to be plotted.
    title: str
        The title of the plot.
    tpc: list[int]
        The TPC(s) to be plotted.
    normalize: bool
        Toggles the normalization of each histogram.
    
    Returns
    -------
    None.
    """
    plt.style.use(PLOT_STYLE)
    if isinstance(datasets, list):
        datasets = [(d, metrics, tpc) for d in datasets]
    elif isinstance(metrics, list):
        datasets = [(datasets, m, tpc) for m in metrics]
    elif isinstance(tpc, list):
        datasets = [(datasets, metrics, t) for t in tpc]
    else:
        print('Misconfigured parameters for plot_planes.')
    
    figure = plt.figure(figsize=(14,6))
    gspec = figure.add_gridspec(16,3)
    haxs = [figure.add_subplot(gspec[2:13, p]) for p in [0,1,2]]
    planes = ['Induction 1', 'Induction 2', 'Collection']
    for pi, p in enumerate(planes):
        for di, d in enumerate(datasets):
            mask = d[0].get_mask(d[1], tpc=d[2], plane=pi)
            style = d[0].get_styling(d[1])
            haxs[pi].hist(d[0][d[1]][mask], range=style[1], bins=style[2],
                          histtype='step', label=labels[di], density=normalize)
            median = np.median(d[0][d[1]][mask])
            color = plt.rcParams["axes.prop_cycle"].by_key()["color"][di]
            haxs[pi].text(0.09+0.45*di, -0.25, f'{median:.2f} ADC', transform=haxs[pi].transAxes, verticalalignment='top', horizontalalignment='left', c=color, size=20)
        haxs[pi].set_xlim(style[1])
        haxs[pi].set_xlabel(style[0])
        haxs[pi].set_ylabel('Entries (Arb.)')
        haxs[pi].set_title(p)

    h, l = plt.gca().get_legend_handles_labels()
    bl = dict(zip(l,h))
    figure.suptitle(title)
    figure.legend(bl.values(), bl.keys())

def plot_planes(dataset, metric, mtype='e2e', tpc=None) -> None:
    """
    Plots the desired metric (channel-to-channel by wire plane) for
    the specified TPC as a set of 1D histograms.

    Parameters
    ----------
    dataset: Dataset
        The Dataset object to be plotted.
    metric: str
        The key in the Dataset that specifies the metric.
    mtype: str or list[str]
        The sub-type(s) of the metric (e2e, c2c).
    tpc: int
        The index of the TPC to select and plot.

    Returns
    -------
    None.
    """
    plt.style.use(PLOT_STYLE)
    figure = plt.figure(figsize=(14,6))
    gspec = figure.add_gridspec(16,3)
    haxs = [figure.add_subplot(gspec[2:13, p]) for p in [0,1,2]]
    planes = ['Induction 1', 'Induction 2', 'Collection']
    if isinstance(mtype, str):
        mtype = [mtype,]
    xaxis = 'Difference [ADC]' if 'abs' in mtype[0] else 'Relative Difference'
    scale = 1.25 if 'abs' in mtype[0] else 0.25
    for pi, p in enumerate(planes):
        for m in mtype:
            if tpc != None:
                mask = ((dataset.noise_data['tpc'] == tpc) & (dataset.noise_data['plane'] == pi))
            else:
                mask = (dataset.noise_data['plane'] == pi)
            label = {'e2e': 'Event-to-Event', 'c2c': 'Channel-to-Channel'}[m[:3]]
            haxs[pi].hist(dataset[f'{metric}_{m}'][mask], range=(-scale,scale), bins=50,
                          histtype='step', density=True, label=label)
        haxs[pi].set_xlim(-scale,scale)
        haxs[pi].set_xlabel(xaxis)
        haxs[pi].set_ylabel('Entries (Arb.)')
        haxs[pi].set_title(p)
    if tpc != None:
        tpc_name = {0: 'EE', 1: 'EW', 2: 'WE', 3: 'WW'}[tpc]
    else:
        tpc_name = 'All TPCs'
    diff = {'e2e': 'Event-to-Event', 'c2c': 'Channel-to-Channel'}[mtype[0][:3]]
    figure.suptitle(f'{diff} Difference in RMS: {tpc_name}')
    h, l = plt.gca().get_legend_handles_labels()
    bl = dict(zip(l,h))
    figure.legend(bl.values(), bl.keys())

def plot_tpc(datasets, labels, metric='raw_rms', tpc=0) -> None:
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

    Returns
    -------
    None.
    """
    plt.style.use(PLOT_STYLE)
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
    if isinstance(tpc, int):
        tpc_name = {0: 'EE', 1: 'EW', 2: 'WE', 3: 'WW'}[tpc]
        figure.suptitle(f'{tpc_name} TPC')

def plot_crate(datasets, labels, metric='raw_rms', component='WW19', label_mean=False) -> None:
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
    label_mean: bool
        Toggles the display of the arithmetic mean in the legend
        label for each dataset/metric.

    Returns
    -------
    None.
    """
    plt.style.use(PLOT_STYLE)
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
        mean = np.mean(d[metric][selected])
        x = 64*d['slot_id'][selected] + d['local_id'][selected]
        label_annotated = f'{labels[di]} ($\mu$: {mean:.2f})' if label_mean else labels[di]
        ax.scatter(x, d[metric][selected], label=label_annotated)
    ax.set_xlim(0,576)
    ax.set_ylim(0, 10.0)
    ax.set_xticks([64*i for i in range(10)])
    ax.set_xlabel('Channel Number')
    ax.set_ylabel('RMS [ADC]')
    ax.legend()
    figure.suptitle(title)

def plot_wire_planes(data, metric, metric_label, tpc=0):
    """
    Create and save a plot displaying a per-plane view with each wire
    in its physical location colored by the specified metric.

    Parameters
    ----------
    data: Dataset
        The dataset containing the data.
    metric: str
        The name of the metric to be plotted.
    metric_label: str
        The label for the metric.
    tpc: int
        The desired TPC to display (0, 1, 2, 3 = EE, EW, WE, WW).

    Returns
    -------
    None.
    """
    plt.style.use(PLOT_STYLE)
    figure = plt.figure(figsize=(20,12))
    gspec = figure.add_gridspec(ncols=10, nrows=3)
    axs = [figure.add_subplot(gspec[i, :9]) for i in [0,1,2]]
    cmap = plt.cm.ScalarMappable(cmap=cm.viridis, norm=colors.Normalize(0,1))
    cmap.set_clim(0,10)
    plane_label = {0: 'Induction 1', 1: 'Induction 2', 2: 'Collection'}

    conn = sqlite3.connect(SQLITE_CHANNEL_MAP_PATH)
    chmap = pd.read_sql_query("SELECT channel_id, z0, y0, z1, y1 FROM physicalwires", conn)
    conn.close()

    for p in [0,1,2]:
        mask = chmap['channel_id'] // 13824 == tpc
        wires = chmap.loc[mask][['channel_id', 'z0', 'y0', 'z1', 'y1']]
        tmp = pd.DataFrame({'channel_id': data['channel_id'], metric: data[metric]}).loc[data['plane'] == p]
        wires = wires.merge(tmp, how='inner', left_on='channel_id', right_on='channel_id')
        cs = cmap.to_rgba(wires[metric])
        wires = [[(x[1], x[2]), (x[3], x[4])] for x in wires.to_numpy()]
        axs[p].add_collection(LineCollection(wires, colors=cs))
        axs[p].set_xlim(-895.95, 895.95)
        axs[p].set_ylim(-181.70, 134.80)
        axs[p].set_xlabel('Z Position [cm]')
        axs[p].set_ylabel('Y Position [cm]')
        axs[p].set_title(plane_label[p])
    cax = figure.add_axes([0.92, 0.05, 0.035, 0.89])
    axcb = figure.colorbar(cmap, ax=axs, cax=cax, use_gridspec=True)
    axcb.set_label(metric_label)
    axcb.solids.set_edgecolor('face')
    tpc_name = {0: 'EE', 1: 'EW', 2: 'WE', 3: 'WW'}
    figure.suptitle(f'{metric_label.split(" [")[0]} by Plane for {tpc_name[tpc]}')

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
    plt.style.use(PLOT_STYLE)
    figure = plt.figure(figsize=(14,6))
    ax = figure.add_subplot()
    ax.plot(np.arange(4096), waveform, linestyle='-', linewidth=1)
    ax.set_xlim(0,4096)
    ax.set_ylim(-20, 20)
    ax.set_xlabel('Time [ticks]')
    ax.set_ylabel('Waveform Height [ADC]')
    figure.suptitle(title)