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
    group_id = df['group_id'].to_numpy()
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