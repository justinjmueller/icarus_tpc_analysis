import numpy as np
import struct
from glob import glob
import matplotlib.pyplot as plt

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
        plt.style.use('plot_style.mplstyle')
        inputs = glob(f'{path}/*')
        if len(inputs) > nevts:
            inputs = inputs[:nevts]
        self.events = [INFNEvent(x) for x in inputs]
        self.covariance = np.mean([x.cov for x in self.events], axis=0)

    def plot_correlation_matrix(self) -> None:
        """
        Plot the channel-to-channel correlation matrix of the test-
        stand data.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        figure = plt.figure(figsize=(8,6))
        ax = figure.add_subplot()
        im = ax.imshow(self.covariance, cmap='RdBu', vmin=-1.0, vmax=1.0)
        figure.colorbar(im, ax=ax)
        ax.set_xticks([64*i for i in range(10)])
        ax.set_yticks([64*i for i in range(10)])
        figure.suptitle('INFN Test Stand Correlation Matrix')

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
        self.calc_rms()
        self.calc_correlations()
        
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
    
    def calc_rms(self) -> None:
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

    def calc_correlations(self) -> None:
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
