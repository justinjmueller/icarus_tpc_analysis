import numpy as np
import pandas as pd
import sqlite3
from database_utilities import *

# Configuration
DB_PATH = '/Users/mueller/Projects/channel_status/db/icarus_channels_dev.db'
SQL_PATH = '/Users/mueller/Projects/GitRepos/ICARUSNoiseAnalysis/tpc_database/sql/'
INPUT_PATH = '/Users/mueller/Projects/channel_status/inputs/'

def main():
    # Open a connection to the database.
    conn = sqlite3.connect(DB_PATH)
    curs = conn.cursor()

    # Create the tables (if they do not exist).
    command(curs, f'{SQL_PATH}create_channelinfo.sql')
    command(curs, f'{SQL_PATH}create_flatcables.sql')
    command(curs, f'{SQL_PATH}create_logicalwires.sql')
    command(curs, f'{SQL_PATH}create_physicalwires.sql')


    # Table: channelinfo
    daq_channels = pd.read_csv(f'{INPUT_PATH}daq_channels.csv')
    channelinfo = daq_channels[['channel_id', 'readout_board_slot', 'channel_number', 'chimney_number',
                                'wire_number', 'cable_label_number', 'channel_type']]
    channelinfo = channelinfo.rename(columns={'readout_board_slot': 'slot_id', 'channel_number': 'local_id',
                                              'chimney_number': 'flange_number', 'cable_label_number': 'cable_number'})
    channelinfo['tpc_number'] = (channelinfo['channel_id'] / 13824).astype(int)
    channelinfo['plane_number'] = np.digitize(channelinfo['channel_id'] % 13824, [2304, 8064, 13824])
    channelinfo['group_id'] = (channelinfo['channel_id'] / 32).astype(int)
    channelinfo['fragment_id'], channelinfo['flange_name'] = map_channels_to_fragment(channelinfo)
    channelinfo['cable_number'] = [fix_cable_label(x[0], x[1]) for x in channelinfo[['cable_number', 'tpc_number']].to_numpy()]
    channelinfo = channelinfo[['channel_id', 'tpc_number', 'plane_number', 'wire_number', 'slot_id', 'local_id',
                               'group_id', 'fragment_id', 'flange_number', 'flange_name', 'cable_number', 'channel_type']]
    vals = [tuple(x) for x in channelinfo.to_numpy()]
    command(curs, f'{SQL_PATH}insert_channelinfo.sql', vals=vals)
    conn.commit()

    # Table: flatcables
    flatcables = pd.read_csv(f'{INPUT_PATH}cables.csv')
    vals = [tuple(x) for x in flatcables.to_numpy()]
    command(curs, f'{SQL_PATH}insert_flatcables.sql', vals=vals)
    conn.commit()

    # Table: logicalwires
    lines = open(f'{INPUT_PATH}ICARUS-channelmap.txt').readlines()
    line_has_channel = ['C:' in x and 'T:' in x and 'P:' in x and 'W:' and '=>' in x in x for x in lines]
    channel_entries = np.array([parse_map_entry(x) for xi, x in enumerate(lines) if line_has_channel[xi]])
    logicalwires = pd.DataFrame(channel_entries, columns=['channel_id', 'c', 't', 'p', 'w'])
    vals = [tuple(x) for x in logicalwires[['channel_id', 'w', 'c', 't', 'p']].to_numpy(dtype=float)][:5]
    command(curs, f'{SQL_PATH}insert_logicalwires.sql', vals=vals)
    conn.commit()

    # Table: physicalwires
    lines = open(f'{INPUT_PATH}ICARUS-geometry.txt').readlines()
    line_has_wire = ['C:' in x and 'T:' in x and 'P:' in x and 'W:' in x for x in lines]
    wire_entries = np.array([parse_wire_entry(x) for xi, x in enumerate(lines) if line_has_wire[xi]])
    wires = pd.DataFrame(wire_entries, columns=['c', 't', 'p', 'w', 'length', 'x', 'y0', 'z0', 'y1', 'z1'])
    wires = wires.merge(logicalwires, left_on=['c', 't', 'p', 'w'], right_on=['c', 't', 'p', 'w'])
    vals = [(float(n), *get_physical_wire_for_channel(g)) for n, g in wires.groupby('channel_id')]
    command(curs, f'{SQL_PATH}insert_physicalwires.sql', vals=vals)
    conn.commit()

    conn.close()

if __name__ == '__main__':
    main()