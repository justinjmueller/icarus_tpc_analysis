import os

# Current directory
CWD = os.getcwd()

# Base path
BASE_PATH = os.path.dirname(os.path.realpath(__file__))

# Plot style
PLOT_STYLE = f'{BASE_PATH}/plot_style.mplstyle'

# TPC channel map paths
SQLITE_CHANNEL_MAP_PATH = '/Users/mueller/Projects/channel_status/db/icarus_channels.db'
CHANNEL_MAP_INPUT = '/Users/mueller/Projects/channel_status/inputs/'