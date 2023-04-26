import numpy as np
import math
import pandas as pd
import uproot
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, path, run) -> None:
        plt.style.use('plot_style.mplstyle')
        data = uproot.open(f'{path}/run{run}.root')
        self.correlations = data['tpcnoiseartdaq/tpccorrelation'].arrays(library='pd')
        self.indexer = {int(i*(i+1)/2) + j: (i, j) for i in range(576) for j in range(i+1)}
        
    def _get_correlation_matrix(self, crate=0) -> np.array:
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
    
    def plot_correlation_matrix(self, crate=0) -> None:
        figure = plt.figure(figsize=(8,6))
        ax = figure.add_subplot()
        cov = self._get_correlation_matrix(crate)
        im = ax.imshow(cov, cmap='RdBu', vmin=-1.0, vmax=1.0)
        figure.colorbar(im, ax=ax)
        ax.set_xticks([64*i for i in range(9)])
        ax.set_yticks([64*i for i in range(9)])

    def plot_average_correlation_matrix(self) -> None:
        figure = plt.figure(figsize=(8,6))
        ax = figure.add_subplot()
        cov = np.mean([self._get_correlation_matrix(i) for i in range(96)], axis=0)
        im = ax.imshow(cov, cmap='RdBu', vmin=-1.0, vmax=1.0)
        cbar = figure.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient')
        ax.set_xticks([64*i for i in range(10)])
        ax.set_yticks([64*i for i in range(10)])
        figure.suptitle('Average Mini-Crate Correlation Matrix')
