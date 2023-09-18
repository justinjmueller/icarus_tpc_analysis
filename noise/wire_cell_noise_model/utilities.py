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
    """
    conn = sqlite3.connect(SQLITE_CHANNEL_MAP_PATH)
    group_id = pd.read_sql_query(f"SELECT group_id FROM channelinfo WHERE channel_id={channel_id};", conn)['group_id'].iloc[0]
    conn.close()

    amplitudes = np.array([histogram.GetBinContent(x, int(group_id)) for x in range(1,2049)])
    counts = histogram.GetBinContent(2050, int(group_id))
    amplitudes /= counts if counts != 0 else 1.0
    amplitudes *= (1e-9 * 3.3 * 1e3 * (1/4095.0))
    amplitudes = np.concatenate([amplitudes, amplitudes[::-1]])
    return amplitudes

def get_block_channel(histogram, channel_id):
    """
    Produces a configuration block for the requested channel using the
    histogram as the input for the spectra.

    Parameters
    ----------
    histogram: ROOT.TH2F
        The input histogram.
    channel_id: int
        The channel ID of a channel within the desired group.
    
    Returns
    -------
    block: dict
        The dictionary containing the key/value pairs for the JSON block.
    """
    f = np.linspace(0,0.00125,2048)
    frequencies = np.concatenate([f, f[::-1]]) 
    amplitudes = get_channel_from_histogram(histogram, channel_id)
    tpc = int(channel_id / 13824)
    tpc_name = {0: 'EE', 1: 'EW', 2: 'WE', 3: 'WW'}[tpc]
    block = {'period': 400.0, 'nsamples': 4096, 'gain': 8.811970678500002e-10, 'shaping': 1.3, 'wire-delta': 32.0,
            'const': 0, 'tpcname': tpc_name, 'freqs': frequencies.tolist(), 'amps': amplitudes.tolist()}
    return block

def get_blocks_tpc(histogram, tpc):
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

    Returns
    -------
    blocks: list[dict]
        The list of blocks for the TPC.
    """
    conn = sqlite3.connect(SQLITE_CHANNEL_MAP_PATH)
    channels = pd.read_sql_query(f"SELECT DISTINCT FLOOR(channel_id / 32) FROM channelinfo WHERE tpc_number={tpc} AND channel_type='wired';", conn).to_numpy()[:,0]
    conn.close()
    blocks = [get_block_channel(histogram, 32*x) for x in channels]
    return blocks

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