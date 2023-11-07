import numpy as np
import pandas as pd
import struct
from glob import glob
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from scipy.signal import periodogram

sys.path.append('..')
from globals import *

class INFNDataset:
    def __init__(self, path, nevts=10) -> None:
        """
        Parameters
        ----------
        path: str
            The path to the folder containing the events.
        nevts: int
            The maximum number of events to load.

        Returns
        -------
        None.
        """
        plt.style.use(PLOT_STYLE)
        if '.npz' in path:
            self._read(path)
        else:
            inputs = glob(f'{path}/*')
            if len(inputs) > nevts:
                inputs = inputs[:nevts]
            self.events = [INFNEvent(x) for x in tqdm(inputs)]
            self.covariance = np.mean([x.cov for x in self.events], axis=0)
            self._get_noise()
            self._get_ffts()

    def write(self, path) -> None:
        """
        Saves the dataset to a npz file.

        Parameters
        ----------
        path: str
            The full path to save the dataset to.

        Results
        -------
        None.
        """
        noise_data = self.median_noise_data[['slot_id', 'local_id', 'pedestal', 'raw_rms']].to_numpy()
        np.savez(path, median_noise_data=noise_data, covariance=self.covariance)

    def _read(self, path) -> None:
        """
        Initializes the dataset by reading in the results from an npz
        file.

        Parameters
        ----------
        path: str
            The full path of the input npz file containing the previous
            analysis results

        Returns
        -------
        None.
        """
        input_file = np.load(path)
        self.median_noise_data = pd.DataFrame(input_file['median_noise_data'], columns=['slot_id', 'local_id', 'pedestal', 'raw_rms'])
        self.median_noise_data['flange'] = 'infn'
        self.covariance = input_file['covariance']
    
    def _get_noise(self) -> None:
        """
        Loads the noise data from the list of INFN events and calculates
        the median over the number of events.

        Parameters
        ----------
        None.
        
        Returns
        -------
        None.
        """
        data = {'flange': np.repeat('INFN', self.events[0].rms.shape[0]),
                'slot_id': np.array([int(x/64) for x in range(len(self.events[0].rms))]),
                'local_id': np.array([x%64 for x in range(len(self.events[0].rms))]),
                'pedestal': np.median(np.vstack([x.pedestals for x in self.events]), axis=0),
                'raw_rms': np.median(np.vstack([x.rms for x in self.events]), axis=0)}
        self.median_noise_data = pd.DataFrame(data)

    def _get_ffts(self) -> None:
        """
        Calculates the average FFTs per board from the list of INFN
        events and stores it as a class attribute.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        self.ffts = np.mean(np.stack([x.ffts for x in self.events], axis=0), axis=0)
    
    def __getitem__(self, key) -> np.array:
        """
        Provides key-value access to the noise metrics.

        Parameters
        ----------
        key: str
            The name of the metric to return.

        Returns
        -------
        The metric corresponding to the key as a numpy array.
        """
        return self.median_noise_data[key].to_numpy()

    def plot_correlation_matrix(self, override_title='', boards=9) -> None:
        """
        Plot the channel-to-channel correlation matrix of the test-
        stand data.

        Parameters
        ----------
        override_title: str
            Overrides the default title of the plot if set.
        boards: int
            The number of boards to include in the plot.

        Returns
        -------
        None.
        """
        figure = plt.figure(figsize=(8,6))
        ax = figure.add_subplot()
        im = ax.imshow(self.covariance, cmap='RdBu', vmin=-1.0, vmax=1.0)
        figure.colorbar(im, ax=ax)
        ax.set_xticks([64*i for i in range(boards+1)])
        ax.set_yticks([64*i for i in range(boards+1)])
        figure.suptitle(override_title if override_title else 'INFN Test Stand Correlation Matrix')
        plt.show()

class INFNEvent:
    def __init__(self, path) -> None:
        """
        Parameters
        ----------
        path: str
            The full path/name of the event.
        
        Returns
        -------
        None.
        """
        raw = open(path, mode='rb').read()
        header = struct.unpack('>cccciiiibbhi', raw[:28])
        self.run_number = header[4]
        self.event_number = header[5]
        self.time = header[6]
        self.total_crates = header[10]
        self.tokens = {'EVEN': self.get_tile_locations(raw, 'EVEN'),
                       'DATA': self.get_tile_locations(raw, 'DATA'),
                       'STAT': self.get_tile_locations(raw, 'STAT'),
                       'CONF': self.get_tile_locations(raw, 'CONF')}
        self.waveforms = np.vstack([self.read_data_tile(raw, x+28+8) for x in self.tokens['DATA']])
        self.pedestals = np.median(self.waveforms, axis=1)
        self.waveforms = self.waveforms - self.pedestals[:, None]
        self._calc_rms()
        self._calc_correlations()
        self.frequencies, self.ffts = periodogram(self.waveforms, fs=2.5e3, axis=1)
        nboards = int(self.waveforms.shape[0] / 64)
        self.ffts = np.array([np.mean(self.ffts[b*64:(b+1)*64, :], axis=0) for b in range(nboards)])
        
    @staticmethod
    def get_tile_locations(evt, token) -> list[int]:
        """"
        Locates the specified tile within the event.

        Parameters
        ----------
        evt: str
            The buffer data of the event.
        token: str
            The type of token to find in the buffer.

        Returns
        -------
        tile_locations: list[int]
            The positions in the buffer that begin a tile of the
            requested token type.
        """
        tokenized = tuple([bytes(x, 'utf-8') for x in token])
        tile_locations = list()
        for pos in range(len(evt)-4):
            if struct.unpack('>cccc', evt[pos:pos+4]) == tokenized:
                tile_locations.append(pos)
        return tile_locations
    
    @staticmethod
    def read_data_tile(evt, start) -> np.array:
        """
        Read the data tile for a single event and return the waveform
        data.

        Parameters
        ----------
        evt: str
            The full buffer of the event.
        start: int
            The position corresponding to the start of the waveform
            data within the data tile.

        Returns
        -------
        waveforms: np.array
            A numpy array of shape (N, 4096) where N is the total
            number of channels in the readout.
        """
        waveforms = np.zeros((64,4096))
        form = '<'+'h'*64
        for t in range(4096):
            waveforms[:,t] = struct.unpack(form,evt[start+t*128:start+(t+1)*128])
        return waveforms
    
    def _calc_rms(self) -> None:
        """
        Calculate the RMS of each waveform. These are stored internal
        to the class.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        self.rms = np.sqrt(np.mean(np.square(self.waveforms), axis=1))

    def _calc_correlations(self) -> None:
        """
        Calculate the channel-to-channel correlation matrix for the
        event. This is stored internal to the class.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        self.cov = np.zeros((self.waveforms.shape[0], self.waveforms.shape[0]))
        for i, j in [(k, l) for k in range(self.cov.shape[0]) for l in range(k+1)]:
            norm = self.waveforms.shape[1] * self.rms[i] * self.rms[j]
            self.cov[i,j] = np.inner(self.waveforms[i,:], self.waveforms[j,:]) / norm
            self.cov[j,i] = self.cov[i,j]
