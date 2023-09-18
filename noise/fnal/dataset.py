import numpy as np
import math
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import sqlite3
import warnings
import sys

sys.path.append('..')
from globals import *

class Dataset:
    def __init__(self, path) -> None:
        """
        Parameters
        ----------
        path: str
            Full path to input data file.

        Returns
        -------
        None.
        """
        plt.style.use(PLOT_STYLE)
        pd.options.mode.chained_assignment = None
        data = uproot.open(path)
        trimmed_keys = [x.split(';')[0] for x in data.keys()]
        if 'tpccorrelation' in trimmed_keys:
            self.correlations = data['tpccorrelation'].arrays(library='pd')
        if 'tpcnoise' in trimmed_keys:
            self._get_noise(data['tpcnoise'].arrays(library='pd'))
        if 'raw_ffts' in trimmed_keys:
            self.rawffts = data['raw_ffts'].values()
            self.intffts = data['int_ffts'].values()
            self.cohffts = data['coh_ffts'].values()
            norm = self.rawffts[-1,:]
            norm[norm == 0] = 1.0
            self.rawffts = self.rawffts[:2049] / norm
            self.intffts = self.intffts[:2049] / norm
            self.cohffts = self.cohffts[:2049] / norm
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
        if key in self.median_noise_data.columns:
            return self.median_noise_data[key].to_numpy()
        else:
            return self.noise_data[key].to_numpy()
        
    def get_mask(self, metric='raw_rms', tpc=None, plane=None) -> np.array:
        """
        Creates a mask for the designated tpc/plane using the shape
        of the provided metric.

        Parameters
        ----------
        metric: str
            The name of the metric to use for creating the mask.
        tpc: int
            The number of the TPC to select in the mask.
        plane: int
            The number of the plane to select in the mask.

        Returns
        -------
        plane_mask: np.array
            A boolean mask corresponding to the metric dimensions
            and the requested tpc/plane.
        """
        if metric in self.median_noise_data.columns:
            blank_mask = np.repeat(True, len(self.median_noise_data))
            tpc_mask = (self.median_noise_data['tpc'] == tpc if tpc is not None else blank_mask)
            plane_mask = (self.median_noise_data['plane'] == plane if plane is not None else blank_mask)
        else:
            blank_mask = np.repeat(True, len(self.noise_data))
            tpc_mask = (self.noise_data['tpc'] == tpc if tpc is not None else blank_mask)
            plane_mask = (self.noise_data['plane'] == plane if plane is not None else blank_mask)
        return tpc_mask & plane_mask
    
    def get_styling(self, metric):
        """
        Retrieves default bin sizes, labels, and ranges for the
        specified metric

        Parameters
        ----------
        metric: str
            The metric name.

        Returns
        -------
        style: list
            List containing as elements (in order) the x-axis label,
            x-axis range, and the bin count.
        """
        return {'raw_rms': ['RMS [ADC]', (0, 10), 50],
                'int_rms': ['RMS [ADC]', (0, 10), 50],
                'coh_rms': ['RMS [ADC]', (0, 10), 50],
                'raw_rms_e2eabs': ['Difference [ADC]', (-1.25, 1.25), 50],
                'raw_rms_c2cabs': ['Difference [ADC]', (-1.25, 1.25), 50],
                'raw_rms_e2erel': ['Relative Difference', (-0.25, 0.25), 50],
                'raw_rms_c2crel': ['Relative Difference', (-0.25, 0.25), 50],
                'hit_occupancy': ['Hit Occupancy', (0, 5), 25],
                'mhit_sadc': ['Max Hit Summed ADC [ADC]', (0,1000), 50],
                'mhit_height': ['Max Hit Height [ADC]', (0,100), 50]}[metric]
    
    def get_ffts(self, group) -> np.array:
        """
        Retrieves the FFTs for the corresponding group

        Parameters
        ----------
        group: int
            The group number.

        Returns
        -------
        ffts: np.array
            The FFTs of the group in a numpy array with shape (3, 2049)
            where the second axis is the frequency bins and the first
            axis contains the raw, intrinsic, and coherent components
            of the noise respectively.
        """
        return np.vstack([self.rawffts[:, group], self.intffts[:, group], self.cohffts[:, group]])
    
    def get_ffts_plane(self, plane=0) -> np.array:
        """
        Retrieves the average FFTs for the requested plane.

        Parameters
        ----------
        plane: int
            The plane number.

        Returns
        -------
        ffts: np.array
            The average FFTs of all groups in the requested plane. This
            has shape (3, 2049) where the second axis is the frequency
            bins and the first axis contains the raw, intrinsic, and
            coherent components of the noise respectively.
        """
        conn = sqlite3.connect(SQLITE_CHANNEL_MAP_PATH)
        groups = pd.read_sql_query(f"SELECT DISTINCT group_id FROM channelinfo WHERE tpc_number={plane};", conn)['group_id']
        conn.close()
        all_ffts = np.dstack([self.get_ffts(x) for x in groups])
        return np.mean(all_ffts, axis=-1)

    def _get_noise(self, input_df) -> None:
        """
        Loads the noise data from the input dataframe while performing
        averaging per channel and basic signal rejection using a range-
        based (max - min of waveform) threshold.

        Parameters
        ----------
        input_df: pandas.DataFrame
            The input noise data in a Pandas DataFrame.
        
        Returns
        -------
        None.
        """
        columns = list(input_df.columns)
        conn = sqlite3.connect(SQLITE_CHANNEL_MAP_PATH)
        chmap = pd.read_sql_query("SELECT channel_id, flange_name FROM channelinfo;", conn)
        conn.close()
        data = input_df.merge(chmap[['channel_id', 'flange_name']], left_on='channel_id', right_on='channel_id')
        flange_map = dict(zip(data['fragment'], data['flange_name']))
        data['plane'] = np.digitize(data['channel_id'] % 13824, [2304, 8064, 13824])
        data['tpc'] = data['channel_id'].to_numpy() // 13824
        mask = (data['mhit_height'] <= 25)
        self.noise_data = data.loc[mask]
        self.median_noise_data = data.loc[mask].groupby('channel_id').median().reset_index()
        self.median_noise_data['flange'] = [flange_map[x] for x in self.median_noise_data['fragment']]
        self.median_noise_data['hit_occupancy'] = data.groupby('channel_id').mean()['hits']
        group_e2e = self.noise_data.groupby('channel_id')
        group_c2c = self.noise_data.groupby(['fragment', 'slot_id'])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.warn("runtime", RuntimeWarning)
            for k in ['raw_rms', 'int_rms', 'coh_rms', 'pedestal']:
                median_e2e = group_e2e[k].transform('median').to_numpy()
                median_c2c = group_c2c[k].transform('median').to_numpy()
                self.noise_data[f'{k}_e2eabs'] = (self.noise_data[k].to_numpy() - median_e2e)
                self.noise_data[f'{k}_c2cabs'] = (self.noise_data[k].to_numpy() - median_c2c)
                self.noise_data[f'{k}_e2erel'] = (self.noise_data[k].to_numpy() - median_e2e) / median_e2e
                self.noise_data[f'{k}_c2crel'] = (self.noise_data[k].to_numpy() - median_c2c) / median_c2c

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
        sel = (self.correlations['rho'] / self.correlations['count']).to_numpy()[crate*size:(crate+1)*size]
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

