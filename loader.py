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
        self.correlations = data['tpcnoiseartdaq/tpccorrelation'].arrays(library='pd')
        self.indexer = {int(i*(i+1)/2) + j: (i, j) for i in range(576) for j in range(i+1)}
        
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
        figure.suptitle(title)

