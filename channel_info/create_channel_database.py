import numpy as np
import pandas as pd
import sqlite3
from database_utilities import command, fix_cable_label, map_channels_to_fragment

# Configuration
DB_PATH = '/Users/mueller/Projects/channel_status/db/icarus_channels_dev.db'
SQL_PATH = '/Users/mueller/Projects/GitRepos/ICARUSNoiseAnalysis/channel_info/sql/'
INPUT_PATH = '/Users/mueller/Projects/channel_status/inputs/'

def main():
    # Open a connection to the database.
    conn = sqlite3.connect(DB_PATH)
    curs = conn.cursor()

    # Create the tables (if they do not exist).
    command(curs, f'{SQL_PATH}create_channelinfo.sql')

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

    conn.close()

if __name__ == '__main__':
    main()