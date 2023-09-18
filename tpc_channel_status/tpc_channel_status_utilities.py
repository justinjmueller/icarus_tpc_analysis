import numpy as np
import uproot
import matplotlib.pyplot as plt
import pandas as pd
import os
from shutil import copyfile
import sqlite3

def load_histogram(rf, key, normalize=True):
    """
    Loads requested histogram from a ROOT file.

    Parameters
    ----------
    rf: ROOT file (uproot)
        The opened uproot ROOT file that contains the histogram.
    key: str
        The name (key) of the histogram.

    Returns
    -------
    contents: np.array
        The bin contents of the histogram normalized by the number of events.
    edges: np.array
        The bin edges of the histogram.
    centers: np.array
        The bin centers of the histogram.
    """
    contents, edges = rf[key].to_numpy()
    centers = (edges[1:] + edges[:-1]) / 2.0
    nevents = int(rf['event_count']) if normalize else 1.0
    return contents / nevents, edges, centers

def calculate_difference(rf0, rf1, key):
    """
    Calculates the difference between two histograms.

    Parameters
    ----------
    rf0: ROOT file (uproot)
        The opened uproot ROOT file that contains the first histogram.
    rf1: ROOT file (uproot)
        The opened uproot ROOT file that contains the second histogram.

    Returns
    -------
    contents: np.array
        The bin-by-bin difference between the two histograms. Each bin
        which would be negative is set to zero instead.
    """
    contents0, _, _ = load_histogram(rf0, key)
    contents1, _, _ = load_histogram(rf1, key)
    contents = contents0 - contents1
    contents[contents < 0] = 0
    return contents

def plot_max_signal_height(rf0, rf1):
    """
    Plots the max signal height distributions for the two input ROOT
    files as well as the difference between the two.

    Parameters
    ----------
    rf0: ROOT file (uproot)
        The opened uproot ROOT file that contains the first histogram.
    rf1: ROOT file (uproot)
        The opened uproot ROOT file that contains the second histogram.

    Returns
    -------
    None.
    """
    figure = plt.figure(figsize=(18,9))
    gspec = figure.add_gridspec(16,3)
    main_axis = figure.add_subplot(gspec[2:13, 0])
    sub_axis = [figure.add_subplot(gspec[2:13, p], sharey=main_axis, sharex=main_axis) for p in [1,2]]
    haxs = [main_axis, *sub_axis]
    planes = ['Induction 1', 'Induction 2', 'Collection']
    for pi, p in enumerate(planes):
        content0, edges0, centers0 = load_histogram(rf0, f'mhit_height_plane{pi}')
        haxs[pi].hist(centers0, weights=content0, bins=len(centers0), range=(edges0[0], edges0[-1]), histtype='step', label='Noise+Signal')

        content1, edges1, centers1 = load_histogram(rf1, f'mhit_height_plane{pi}')
        haxs[pi].hist(centers1, weights=content1, bins=len(centers1), range=(edges1[0], edges1[-1]), histtype='step', label='Noise-only')

        content2 = calculate_difference(rf0, rf1, f'mhit_height_plane{pi}')
        haxs[pi].hist(centers1, weights=content2, bins=len(centers1), range=(edges1[0], edges1[-1]), histtype='step', label='Signal-only')

        haxs[pi].set_title(p)

    haxs[0].set_xlim(0,100)
    haxs[0].set_ylim(0,700)
    haxs[0].set_ylabel('Entries')
    h, l = plt.gca().get_legend_handles_labels()
    bl = dict(zip(l,h))
    figure.legend(bl.values(), bl.keys())
    figure.suptitle('Distribution of Max Hit Height for Signal and Noise')


def plot_acceptance_rejection_curves(rf0, rf1):
    """
    Plots the acceptance/rejection curves as a function of a simple cut
    on the largest hit height (per waveform). The noise acceptance
    reflects the percentage of noise-only waveforms which are accepted
    by the cut and the signal rejection reflects the complement of the
    percentage of signal-containing waveforms accepted by the cut.

    Parameters
    ----------
    rf0: ROOT file (uproot)
        The opened uproot ROOT file that contains the first histogram.
    rf1: ROOT file (uproot)
        The opened uproot ROOT file that contains the second histogram.

    Returns
    -------
    None.
    """
    figure = plt.figure(figsize=(18,9))
    gspec = figure.add_gridspec(16,3)
    main_axis = figure.add_subplot(gspec[2:13, 0])
    sub_axis = [figure.add_subplot(gspec[2:13, p], sharey=main_axis, sharex=main_axis) for p in [1,2]]
    haxs = [main_axis, *sub_axis]
    planes = ['Induction 1', 'Induction 2', 'Collection']

    for pi, p in enumerate(planes):

        content_noise, _, centers = load_histogram(rf1, f'mhit_height_plane{pi}')
        content_signal = calculate_difference(rf0, rf1, f'mhit_height_plane{pi}')
        underflow_signoise = rf0[f'mhit_height_plane{pi}'].counts(flow=True)[0] / float(rf0['event_count'])
        underflow_noise = rf1[f'mhit_height_plane{pi}'].counts(flow=True)[0] / float(rf1['event_count'])

        noise_acceptance = (np.cumsum(content_noise) + underflow_noise) / (np.sum(content_noise) + underflow_noise)
        signal_acceptance = 1.0 - np.cumsum(content_signal) / (np.sum(content_signal) + underflow_signoise)
        haxs[pi].scatter(centers, 100*noise_acceptance, label='Noise Acceptance', s=3)
        haxs[pi].scatter(centers, 100*signal_acceptance, label='Signal Rejection', s=3)
        haxs[pi].set_xlim(0,100)
        haxs[pi].set_xlabel('Largest Hit Height [ADC]')
        haxs[pi].set_title(p)
        if pi > 0:
            plt.setp(haxs[pi].get_yticklabels(), visible=False)

    haxs[0].set_ylim(0,100)
    haxs[0].set_ylabel('Performance [%]')
    h, l = plt.gca().get_legend_handles_labels()
    bl = dict(zip(l,h))
    figure.legend(bl.values(), bl.keys())
    figure.suptitle('Signal/Noise Discrimination')

def plot_signal_occupancy(rf, norm=False):
    """
    Plots the signal occupancy (percentage of waveforms containing
    signal) for each plane. Each entry is a single channel.

    Parameters
    ----------
    rf: ROOT file (uproot)
        The opened uproot file that contains the input histograms.
    norm: bool
        Boolean flag for turning on normalization by wire length.

    Returns
    -------
    None.
    """
    figure = plt.figure(figsize=(12,9))
    ax = figure.add_subplot()

    for pi, p in enumerate(['Induction 1', 'Induction 2', 'Collection']):
        contents, edges, centers = load_histogram(rf, f'occupancy{"_norm" if norm else ""}_plane{pi}', normalize=False)
        ax.hist(centers, weights=contents, range=(edges[0], edges[-1]), bins=len(contents), histtype='step', label=p)
    xlabel = 'Signal Occupancy ' + ('[% Signal Events / meter]' if norm else '[% Signal Events]')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Entries')
    ax.set_xlim(edges[0], edges[-1])
    ax.legend()
    figure.suptitle('Signal Occupancy')

def plot_noise(rf, norm=False):
    """
    Plots the noise for each plane. Each entry corresponds to a single
    waveform (total = #events * #channels).

    Parameters
    ----------
    rf: ROOT file (uproot)
        The opened uproot file that contains the input histograms.
    norm: bool
        Boolean flag for turning on normalization by total capacitance.

    Returns
    -------
    None.
    """
    figure = plt.figure(figsize=(12,9))
    ax = figure.add_subplot()

    for pi, p in enumerate(['Induction 1', 'Induction 2', 'Collection']):
        contents, edges, centers = load_histogram(rf, f'raw_rms_{"norm" if norm else "sigf"}_plane{pi}', normalize=True)
        ax.hist(centers, weights=contents, range=(edges[0], edges[-1]), bins=len(contents), histtype='step', label=p)
    xlabel = 'RMS ' + ('[ADC / pF]' if norm else '[ADC]')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Entries')
    ax.set_xlim(edges[0], edges[-1])
    ax.legend()
    figure.suptitle('Noise')

def plot_low_noise(rf):
    """
    Plots the fraction of low-noise waveforms per channel.

    Parameters
    ----------
    rf: ROOT file (uproot)
        The opened uproot file that contains the input.

    Returns
    -------
    None.
    """
    figure = plt.figure(figsize=(12,9))
    ax = figure.add_subplot()
    data = rf['channels'].arrays(['channel_id', 'low_noise_normed'], library='pd')
    data['plane'] = np.digitize(data['channel_id'] % 13824, [2304, 8064, 13824])
    for pi, p in enumerate(['Induction 1', 'Induction 2', 'Collection']):
        mask = data['plane'] == pi
        ax.hist(100*data.loc[mask]['low_noise_normed'], range=(0,100), bins=100, histtype='step', label=p)
    ax.set_xlabel('Low Noise Rate [%]')
    ax.set_ylabel('Entries')
    ax.set_xlim(0, 100)
    ax.legend()
    figure.suptitle('High Noise Rate per Plane')

def plot_high_noise(rf):
    """
    Plots the fraction of high-noise waveforms per channel.

    Parameters
    ----------
    rf: ROOT file (uproot)
        The opened uproot file that contains the input.

    Returns
    -------
    None.
    """
    figure = plt.figure(figsize=(12,9))
    ax = figure.add_subplot()
    data = rf['channels'].arrays(['channel_id', 'high_noise'], library='pd')
    data['plane'] = np.digitize(data['channel_id'] % 13824, [2304, 8064, 13824])
    for pi, p in enumerate(['Induction 1', 'Induction 2', 'Collection']):
        mask = data['plane'] == pi
        ax.hist(100*data.loc[mask]['high_noise'], range=(0,10), bins=100, histtype='step', label=p)
    ax.set_xlabel('High Noise Rate [%]')
    ax.set_ylabel('Entries')
    ax.set_xlim(0, 10)
    ax.legend()
    figure.suptitle('High Noise Rate per Plane')

def summarize_low_occupancy(rf, cut=0.025):
    """
    Returns a list of channels which fail the low signal occupancy
    cut.

    Parameters
    ----------
    rf: ROOT file (uproot)
        The opened uproot file that contains the input.
    cut: float
        The cut value that delineates low and nominal signal occupancy.
    
    Returns
    -------
    result: pd.DataFrame
        The resulting Dataframe that lists the channels that do not
        pass the signal occupancy cut.
    """
    all = rf['channels'].arrays(['channel_id', 'occupancy_normed'], library='pd')
    mask = all['occupancy_normed'] < cut
    result = {'channel_id': all.loc[mask]['channel_id'], 'occupancy': all.loc[mask]['occupancy_normed'], 'reason': np.repeat('Low signal occupancy', np.sum(mask))}
    result = pd.DataFrame(result)
    return result

def summarize_low_noise(rf, cut=0.025):
    """
    Returns a list of channels which fail the low raw noise cut.

    Parameters
    ----------
    rf: ROOT file (uproot)
        The opened uproot file that contains the input.
    cut: float
        The cut value that delineates low and nominal raw noise.
    
    Returns
    -------
    result: pd.DataFrame
        The resulting Dataframe that lists the channels that do not
        pass the low raw noise cut.
    """
    all = rf['channels'].arrays(['channel_id', 'low_noise'], library='pd')
    mask = all['low_noise'] > cut
    result = {'channel_id': all.loc[mask]['channel_id'], 'low_noise': all.loc[mask]['low_noise'], 'reason': np.repeat('Low raw noise', np.sum(mask))}
    result = pd.DataFrame(result)
    return result

def summarize_high_noise(rf, cut=0.025):
    """
    Returns a list of channels which fail the high raw noise cut.

    Parameters
    ----------
    rf: ROOT file (uproot)
        The opened uproot file that contains the input.
    cut: float
        The cut value that delineates high and nominal raw noise.
    
    Returns
    -------
    result: pd.DataFrame
        The resulting Dataframe that lists the channels that do not
        pass the high raw noise cut.
    """
    all = rf['channels'].arrays(['channel_id', 'high_noise'], library='pd')
    mask = all['high_noise'] > cut
    result = {'channel_id': all.loc[mask]['channel_id'], 'high_noise': all.loc[mask]['high_noise'], 'reason': np.repeat('High raw noise', np.sum(mask))}
    result = pd.DataFrame(result)
    return result

def summarize_hardware(hardware_list):
    """
    Returns a list of channels which have been identified to have
    hardware reasons for not functioning properly.

    Parameters
    ----------
    hardware_list: str
        The full path of the CSV file containing the known channels
        with hardware issues.
    
    Returns
    -------
    result: pd.DataFrame
        The resulting DataFrame that lists the channels that have
        known hardware issues.
    """
    return pd.read_csv(hardware_list)

def summarize_all(dfs):
    """
    Concatenates a list of DataFrames that each separately list the
    channels failing some channel health cut. Each 'reason' is
    preserved, but exact metric values of the failing cut are not.

    Parameters
    ----------
    dfs: list[pd.DataFrame]
        The list of pandas DataFrames that each contain a summary of
        channels failing some cut.
    
    Returns
    -------
    result: pd.DataFrame
        The concatenated DataFrame summarizing the joint information
        about channels failing channel health cuts.
    """
    bad_channels = dict()
    for df in dfs:
        for c, r in df[['channel_id', 'reason']].to_numpy():
            if c in bad_channels:
                bad_channels[c] = bad_channels[c] + f' & {r.capitalize()}'
            else:
                bad_channels[c] = r.capitalize()
    result = {'channel_id': [k for k in bad_channels.keys()], 'reason': [v for v in bad_channels.values()]}
    result = pd.DataFrame(result)
    reasons, counts = np.unique(result['reason'], return_counts=True)
    for r, c in zip(reasons, counts):
        print(f'Reason: {r}, Count: {c}')
    return result

def export_channel_status(export_name, icarus_db, status_df):
    """
    Writes a new table containing channel status information to a copy
    of the existing database containing the ICARUS channel information
    (as prepared by the 'tpc_database' project).

    Parameters
    ----------
    export_name: str
        The name to be assigned to the output .db file.
    icarus_db: str
        The full path of the input ICARUS channels database.
    status_df: pd.DataFrame
        The pandas DataFrame that reflects the total list of non-healthy
        channels.

    Returns
    -------
    None.
    """
    base = '/'.join(export_name.split('/')[:-1])
    if not os.path.exists(base):
        os.makedirs(base)
    copyfile(icarus_db, export_name)

    conn = sqlite3.connect(export_name)
    curs = conn.cursor()
    command = 'CREATE TABLE IF NOT EXISTS channelstatus(channel_id integer NOT NULL PRIMARY KEY, reason text NOT NULL, diagnose_status text NOT NULL);'
    curs.execute(command)
    insert = 'INSERT INTO channelstatus(channel_id, reason, diagnose_status) VALUES(?,?,?);'
    add_channels = np.arange(55296, dtype=int)[~np.isin(np.arange(55296, dtype=int), status_df['channel_id'])]
    add_reasons = np.repeat('', len(add_channels))
    add_df = pd.DataFrame({'channel_id': add_channels, 'reason': add_reasons})
    status_df = pd.concat([status_df, add_df]).sort_values('channel_id')

    reasons = np.unique(status_df['reason'])
    reason_map = {r: 'kDEAD' if 'Low signal occupancy' in r else 'kGOOD' for r in reasons}
    for r in reason_map.keys():
        if 'Connectivity' in r or 'Isolated' in r:
            reason_map[r] = 'kDEAD'
        elif 'High raw noise' in r:
            reason_map[r] = 'kNOISY'
    status_df['diagnose_status'] = [reason_map[r] for r in status_df['reason']]
    curs.executemany(insert, [(c, r, d)  for c, r, d in status_df[['channel_id', 'reason', 'diagnose_status']].to_numpy()])
    conn.commit()
    conn.close()