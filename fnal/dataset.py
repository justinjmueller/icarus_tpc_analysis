import numpy as np
import math
import pandas as pd
import uproot
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, path, run) -> None:
        """
        Parameters
        ----------
        path: str
            Path to input data file.
        run: int
            Run number for the data to load.

        Returns
        -------
        None.
        """
        plt.style.use('plot_style.mplstyle')
        data = uproot.open(f'{path}/run{run}.root')
        self.chmap = pd.read_csv('channel_map.csv')
        self.correlations = data['tpcnoiseartdaq/tpccorrelation'].arrays(library='pd')
        self._get_noise(data['tpcnoiseartdaq/tpcnoise'].arrays(library='pd'))
        self.indexer = {int(i*(i+1)/2) + j: (i, j) for i in range(576) for j in range(i+1)}
    
    def __getitem__(self, key) -> np.array:
        """
        Provides key-value access to the median noise DataFrame.

        Parameters
        ----------
        key: str
            The name of the column to return.

        Returns
        -------
        The column corresponding to the key as a numpy array.
        """
        return self.median_noise_data[key].to_numpy()

    def _get_noise(self, input_df, signal_threshold=[40,25,25]) -> None:
        """
        Loads the noise data from the input dataframe while performing
        averaging per channel and basic signal rejection using a range-
        based (max - min of waveform) threshold.

        Parameters
        ----------
        input_df: pandas.DataFrame
            The input noise data in a Pandas DataFrame.
        signal_threshold: list(float)
            Per-plane thresholds on the range to be used for basic
            signal rejection.
        
        Returns
        -------
        None.
        """
        data = input_df.astype({'run': int, 'event': int, 'time': int,
                                'ch': int,'frag': int, 'board': int,
                                'slot_id': int})
        data = data.rename(columns={'frag': 'fragment', 'rms': 'rawrms', 'ped': 'pedestal'})
        columns = list(data.columns)
        data = data.merge(self.chmap,
                          left_on=['fragment', 'board', 'ch'],
                          right_on=['fragment', 'readout_board_slot', 'channel_number'])
        data = data[columns + ['channel_id', 'flange']]
        flange_map = dict(zip(data['fragment'], data['flange']))
        data['plane'] = np.digitize(data['channel_id'] % 13824, [2304, 8064, 13824])
        data['tpc'] = data['channel_id'].to_numpy() // 13824
        mask = data['range'] < np.array([signal_threshold[x] for x in data['plane']])
        self.noise_data = data.loc[mask]
        self.median_noise_data = data.loc[mask].groupby('channel_id').median().reset_index()
        self.median_noise_data['flange'] = [flange_map[x] for x in self.median_noise_data['fragment']]

    def _get_correlation_matrix(self, crate=0) -> np.array:
        """
        Loads the correlation matrix containing channel-to-channel
        correlations for the desired crate.

        Parameters
        ----------
        crate: int
            The crate number to load from the correlations dataset.

        Returns
        -------
        cov: np.array
            A covariance matrix of shape (576,576).
        """
        size = int(576 * 577 / 2)
        cov = np.zeros((576,576))
        sel = self.correlations['rho'].to_numpy()[crate*size:(crate+1)*size]
        for si, s in enumerate(sel):
            i, j = self.indexer[si]
            if math.isnan(s):
                s = 0
            cov[i,j] = s
            cov[j,i] = s
        return cov
    
    @staticmethod
    def _map_crate(crate_name='WW19') -> int:
        """
        Maps the mini-crate name to a unique integer.

        Parameters
        ----------
        crate_name: str
            The name of the mini-crate.

        Returns
        -------
        offset: int
            The crate number of the requested mini-crate.
        """
        tpc = {'EE': 0, 'EW': 1, 'WE': 2, 'WW': 3}[crate_name.upper()[:2]]
        offset = 24*tpc + int(crate_name[2:4]) + 1
        if len(crate_name) == 5 and crate_name[2:4] == '01':
            offset += {'T': -2, 'M': -1, 'B': 0}[crate_name.upper()[4]]
        elif len(crate_name) == 5 and crate_name[2:4] == '20':
            offset += {'T': 0, 'M': 1, 'B': 2}[crate_name.upper()[4]]
        return offset
    
    def plot_correlation_matrix(self, crate='WW19') -> None:
        """
        Plot the channel-to-channel correlation matrix of either a
        single mini-crate or the average over all mini-crates.

        Parameters
        ----------
        crate: str
            The name of the single mini-crate, or the keyword 'all'
            if the average is desired.

        Returns
        -------
        None.
        """
        figure = plt.figure(figsize=(8,6))
        ax = figure.add_subplot()

        if crate.upper() == 'ALL':
            title = 'Average Mini-Crate Correlation Matrix'
            cov = np.mean([self._get_correlation_matrix(i) for i in range(96)], axis=0)
        else:
            title = f'{crate.upper()} Correlation Matrix'
            crate = self._map_crate(crate)
            cov = self._get_correlation_matrix(crate)

        im = ax.imshow(cov, cmap='RdBu', vmin=-1.0, vmax=1.0)
        figure.colorbar(im, ax=ax)
        ax.set_xticks([64*i for i in range(10)])
        ax.set_yticks([64*i for i in range(10)])
        ax.set_xlabel('Channel Number')
        ax.set_ylabel('Channel Number')
        figure.suptitle(title)

