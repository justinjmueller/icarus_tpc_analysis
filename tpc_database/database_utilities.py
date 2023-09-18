import numpy as np
import sqlite3

def command(curs, comm, vals=None) -> None:
    """
    Execute a command defined in an SQL file.

    Parameters
    ----------
    curs: The SQLite cursor handle.
    comm: The file name and path for the sql command.
    vals: Values to use as arguments for the sql command (tuple).

    Returns
    -------
    None.
    """
    sql_command = open(comm, 'r').read()
    try:
        if isinstance(vals, list):
            curs.executemany(sql_command, vals)
        elif vals:
            curs.execute(sql_command, vals)
        else:
            curs.execute(sql_command)
    except Exception as e:
        print(e)
        print(f'Caught exception with command {comm} and values {vals}.')

def fix_cable_label(l, tpc) -> str:
    """
    Fixes the cable label if the cable is affected by the S <-> V swap.

    Parameters
    ----------
    l: str
        The original label
    tpc: int
        The TPC (0,1,2,3) where the cable resides.

    Returns
    -------
    label: str
        The fixed (or original if not affected) label. 
    """
    label = f'{l[0]}{l[1:]:0>2}'
    if tpc == 1:
        label = label.replace('V', 'S')
    elif tpc == 2:
        label = label.replace('S', 'V')
    return label

def map_channels_to_fragment(df) -> tuple[np.array, np.array]:
    """
    Maps the channels in the input DataFrame to its corresponding
    fragment ID.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame containing the channel information to be mapped.

    Returns
    -------
    fragment_id: np.array
        The fragment ID corresponding to each channel.
    crate_names: np.array
        The name of the crate corresponding to each channel.
    """
    channel_id = df['channel_id'].to_numpy()
    chimney_number = df['flange_number'].to_numpy()
    group_id = channel_id // 32
    group_bin = np.digitize(group_id % 432, [18, 36, 54, 72])
    offsets = np.array([-1, -2, 1, 0, 3])[group_bin]
    offsets[(chimney_number != 1) & (chimney_number != 20)] = 0
    offsets[(offsets == 3) & (chimney_number == 20)] = 2
    offsets[(offsets == 3) & (chimney_number == 1)] = 0
    crate = 24 * df['tpc_number'] + chimney_number + 1 + offsets
    fragment_id = 2048 + 2 * (crate + 96 + 116 * (8 + 2 * np.floor_divide(crate, 24) + np.floor_divide(crate % 24, 12)))

    subdesignation = np.array(['M', 'T', 'M', 'T', 'B'])[group_bin]
    subdesignation[(chimney_number != 1) & (chimney_number != 20)] = ''
    tpcnames = np.array(['EE', 'EW', 'WE', 'WW'])[df['tpc_number']]
    crate_names = np.array([tpcnames[i] + f'{c:02}' + subdesignation[i] for i, c in enumerate(chimney_number)])

    return fragment_id, crate_names

def parse_wire_entry(line) -> tuple:
    """
    Parses a line entry from the wire geometry text dump.

    Parameters
    ----------
    line: str
        A single entry from the wire geometry.

    Returns
    -------
    c: int
        The cryostat containing the wire.
    t: int
        The TPC containing the wire.
    p: int
        The plane containing the wire.
    w: int
        The wire number.
    length: float
        The length of the wire.
    x: float
        The x-coordinate of the wire.
    y0: float
        The y-coordinate of the first endpoint.
    z0: float
        The z-coordinate of the first endpoint.
    y1: float
        The y-coordinate of the second endpoint.
    z1: float
        The z-coordinate of the second endpoint.
    """
    slices = line.split(':')
    c = slices[1][:-2]
    t = slices[2][:-2]
    p = slices[3][:-2]
    w = slices[4].split(' ')[0]
    slices = line.split('(')
    coordinates0 = [float(x) for x in slices[1].split(')')[0].split(',')]
    coordinates1 = [float(x) for x in slices[2].split(')')[0].split(',')]
    length = float(slices[3].split(' ')[0])
    x, y0, z0 = coordinates0
    _, y1, z1 = coordinates1
    return c, t, p, w, length, x, y0, z0, y1, z1

def parse_map_entry(line) -> tuple:
    """
    Parses a line entry from the ICARUS channel map text dump.
    
    Parameters
    ----------
    line: str
        A single entry from the channel map.

    Returns
    -------
    channel_id: int
        The channel ID.
    c: int
        The cryostat containing the wire.
    t: int
        The TPC containing the wire.
    p: int
        The plane containing the wire.
    w: int
        The wire number.
    """
    slices = line.split(':')
    c = slices[1][:-2]
    t = slices[2][:-2]
    p = slices[3][:-2]
    w = slices[4].split(' ')[0]
    slices = line.split('>')
    channel_id = slices[-1].strip('\n')[1:]
    return channel_id, c, t, p, w

def get_physical_wire_for_channel(g) -> tuple[float]:
    """
    Merges the logical wire information (assuming the input belongs to
    a single channel) into a single physical wire.

    Parameters
    ----------
    g: pd.DataFrame
        The DataFrame storing logical wire entries for the single
        channel.
    
    Returns
    -------
    x: float
        The x-coordinate of the physical wire.
    y0: float
        The y-coordinate of the first endpoint.
    z0: float
        The z-coordinate of the first endpoint.
    y1: float
        The y-coordinate of the second endpoint.
    z1: float
        The z-coordinate of the second endpoint.
    length: float
        The total length of the physical wire.
    capacitance: float
        The total capacitance of the physical wire.
    """
    endpoints = np.vstack([g[['y0', 'z0']].to_numpy(), g[['y1', 'z1']].to_numpy()]).astype(float)
    x = g['x'].astype(float).iloc[0]
    y0, z0 = endpoints[0,:]
    y1, z1 = endpoints[-1,:]
    length = np.sum(g['length'].astype(float))
    if float(g['p'].iloc[0]) == 0:
        length = 942.0
        if z0 != 0:
            z0 = 942.0 * np.sign(z0)
        if z1 != 0:
            z1 = 942.0 * np.sign(z1)
    capacitance = length * (0.20 if g['p'].iloc[0] != 1 else 0.21)
    return x, y0, z0, y1, z1, length, capacitance