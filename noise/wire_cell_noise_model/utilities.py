import numpy as np
import sqlite3
import pandas as pd
import json
import sys

sys.path.append('/Users/mueller/ROOT/install/lib/')
sys.path.append('../../')

from ROOT import TH2F
from globals import *

def get_channel_from_histogram(histogram, channel_id):
    """
    Loads the spectra corresponding to the channel (or rather its
    group) from the input histogram.

    Parameters
    ----------
    histogram: ROOT.TH2F
        The input histogram.
    channel_id: int
        The channel ID of a channel within the desired group.

    Returns
    -------
    amplitudes: np.array
        The spectra corresponding to the channel.
    rms: float
        The RMS calculated using the power spectrum.
    """
    conn = sqlite3.connect(SQLITE_CHANNEL_MAP_PATH)
    group_id = pd.read_sql_query(f"SELECT group_id FROM channelinfo WHERE channel_id={channel_id};", conn)['group_id'].iloc[0]
    conn.close()

    amplitudes = np.array([histogram.GetBinContent(x, int(group_id)) for x in range(1,2049)])
    counts = histogram.GetBinContent(2050, int(group_id))
    amplitudes /= counts if counts != 0 else 1.0
    rms = (1.0 / 0.8891829146258154) * np.sqrt((2.0 / (4096.0**2)) * np.sum(np.square(amplitudes)))
    amplitudes *= (1e-9 * 3.3 * 1e3 * (1/4095.0))
    amplitudes = np.concatenate([amplitudes, amplitudes[::-1]])
    return amplitudes, rms

def get_group_from_histogram(histogram, group_id):
    """
    Loads the spectra corresponding to the channel (or rather its
    group) from the input histogram.

    Parameters
    ----------
    histogram: ROOT.TH2F
        The input histogram.
    group_id: int
        The group ID of the desired group.

    Returns
    -------
    amplitudes: np.array
        The spectra corresponding to the channel.
    rms: float
        The RMS calculated using the power spectrum.
    """

    amplitudes = np.array([histogram.GetBinContent(x, int(group_id)) for x in range(1,2049)])
    counts = histogram.GetBinContent(2050, int(group_id))
    amplitudes /= counts if counts != 0 else 1.0
    rms = (1.0 / 0.8891829146258154) * np.sqrt((2.0 / (4096.0**2)) * np.sum(np.square(amplitudes)))
    amplitudes *= (1e-9 * 3.3 * 1e3 * (1/4095.0))
    amplitudes = np.concatenate([amplitudes, amplitudes[::-1]])
    return amplitudes, rms

def get_block_channel(histogram, channel_id, noise_val=None):
    """
    Produces a configuration block for the requested channel using the
    histogram as the input for the spectra.

    Parameters
    ----------
    histogram: ROOT.TH2F
        The input histogram.
    channel_id: int
        The channel ID of a channel within the desired group.
    noise_val: float or None
        If not None, force the integral of the spectra to match
        this RMS value.
    
    Returns
    -------
    block: dict
        The dictionary containing the key/value pairs for the JSON block.
    """
    f = np.linspace(0, 0.00125, 2048)
    frequencies = np.concatenate([f, f[::-1]]) 
    amplitudes, rms = get_channel_from_histogram(histogram, channel_id)
    if noise_val is not None:
        amplitudes *= noise_val / rms
    tpc = int(channel_id / 13824)
    tpc_name = {0: 'EE', 1: 'EW', 2: 'WE', 3: 'WW'}[tpc]
    block = {'freqs': frequencies.tolist(), 'amps': amplitudes.tolist()}
    return block

def get_block_groups(histogram, groups, group_id_assignment):
    """
    Produces a configuration block for the requested groups using the
    histogram as the input for the spectra.

    Parameters
    ----------
    histogram: ROOT.TH2F
        The input histogram.
    groups: list[int]
        The groups to be represented by this block.
    group_id_assignment: int
        The group ID to assign to this block.
    
    Returns
    -------
    block: dict
        The dictionary containing the key/value pairs for the JSON block.
    """
    f = np.linspace(0, 0.00125, 2048)
    frequencies = np.concatenate([f, f[::-1]])

    amplitudes = [get_group_from_histogram(histogram, g)[0] for g in groups]
    mask = [np.any(a != 0) for a in amplitudes]
    if sum(mask) != 0:
        amps = np.mean([a for ai, a in enumerate(amplitudes) if mask[ai]], axis=0)
    else:
        amps = np.mean([a for ai, a in enumerate(amplitudes)], axis=0)
    tpc = int(groups[0] / 432)
    tpc_name = {0: 'EE', 1: 'EW', 2: 'WE', 3: 'WW'}[tpc]
    block = {'group': group_id_assignment, 'nsamples': 4096, 'period': 400, 'freqs': frequencies.tolist(), 'amps': amps.tolist()}
    return block

def get_blocks_tpc(histogram, tpc, noise_vals=None):
    """
    Produces a configuration block for the requested TPC using the
    histogram as the input for the spectra (coherent or intrinsic).

    Parameters
    ----------
    histogram: ROOT.TH2F
        The input histogram.
    tpc: int
        The number of the TPC to the configuration blocks to be
        produced.
    noise_vals: dict or None.
        If not None, use the (channel_id, rms) pairs to force the
        integral of the spectra to match the RMS value.

    Returns
    -------
    blocks: list[dict]
        The list of blocks for the TPC.
    maps: list[dict]
        The list of channel group configuration blocks for the TPC.
    """
    conn = sqlite3.connect(SQLITE_CHANNEL_MAP_PATH)
    res = pd.read_sql_query(f"SELECT group_id, channel_type FROM channelinfo WHERE tpc_number={tpc} GROUP BY group_id ORDER BY fragment_id, slot_id;", conn).to_numpy()

    conn.close()
    pairs = np.split(res[:,0], res.shape[0]/2)
    iswired = [np.any(x=='wired') for x in np.split(res[:,1], res.shape[0]/2)]
    pairs = [np.array([x[0], x[1]])[[res[2*xi, 1]=='wired', res[2*xi+1, 1]=='wired']] for xi, x in enumerate(pairs)]

    blocks = [get_block_groups(histogram, g, gi+tpc*216) for gi, g in enumerate(pairs) if iswired[gi]]
    maps = [get_group_to_channel_block(g, gi+tpc*216) for gi, g in enumerate(pairs) if iswired[gi]]
    return blocks, maps

def get_group_to_channel_block(groups, group_id_assignment):
    """
    Produces a list of blocks that configure the channels that belong
    to each group set.
    
    Parameters
    ----------
    groups: list[int]
        The group_ids describing the block.
    group_id_assignment: int
        The group ID to assign to this block.

    Returns
    -------
    block: dict
        The configuration block.
    """
    conn = sqlite3.connect(SQLITE_CHANNEL_MAP_PATH)
    channels = pd.read_sql_query(f"SELECT channel_id FROM channelinfo WHERE {' OR '.join(['group_id='+str(g) for g in groups])};", conn).to_numpy()[:,0]
    conn.close()
    block = {'groupID': group_id_assignment, 'channels': channels.tolist()}
    return block

def write_blocks(blocks, path, compress=False):
    """
    Writes a list of blocks a JSON file.

    Parameters
    ----------
    blocks: list[dict]
        The list of blocks to be written to a JSON file.
    path: str
        The full path for the JSON output

    Returns
    -------
    None.
    """
    json_format = json.dumps(blocks, indent=4)
    noise_file = open(path,'w')
    noise_file.write(json_format)
    noise_file.close()
    if compress:
        os.system(f'bzip2 -zf {path}')