import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot
import sqlite3

class TPDataset:
    """
    Class container for an ICARUS test pulse injection dataset. Also
    contains helper methods for extracting metrics, storing, and
    producing plots of the results

    Attributes
    ----------
    ploc: np.array
        The array of all positive lobe centers.
    mloc: np.array
        The array of all negative lobe centers.
    pulses: np.array
        The array of all pulses with shape (N = #channels, 150).
    is_pulsed: np.array
        Boolean mask tagging channels as pulsed with shape
        (N = #channels,).
    chmap_join_index: np.array
        An array that contains the index to be used when matching
        against information from the channel map table.
    channel_id: np.array
        The global channel number that uniquely identifies a channel.
    flange_name: np.array
        The name of the flange that contains the channel.
    slot_id: np.array
        The position of the readout board containing the channel.
    local_id: np.array
        The position of the channel within the readout board.
    metrics: dict
        The dictonary containing the metrics (callables) keyed by name.
    analysis_data: pd.DataFrame
        The pandas DataFrame containing the metrics for each channel.

    Methods
    -------
    add_metric(func, name) -> None:
        Adds a metric (callable) to the list of metrics.
    run_analysis() -> pd.DataFrame:
        Runs each metric on each channel and returns the result as a
        pandas DataFrame.
    plot_metric_board(name, yrange, metric_label, title) -> None.
        Plots a single metric as a 2D histogram across the common
        readout board grouping.
    plot_metric_crate(name, yrange, metric_label, title) -> None.
        Plots a single metric as a 2D histogram across the common
        readout mini-crate grouping.
    _is_pulsed_channel(w, thr=400) -> bool:
        Checks the channel to see if it has been pulsed.
    _read_channel_map() -> None:
        Reads the channel map from the SQLite database and stores the
        relevant information.
    __repr__() -> str:
        Returns a string summarizing the TPDataset.
    __sub__(other) -> TPDataset:
        Creates a new TPDataset from the difference of two TPDatasets.
    __add__(other) -> TPDataset:
        Creates a new TPDataset from the sum of two TPDatasets.
    join(other) -> TPDataset:
        Creates a new TPDataset by joining two TPDatasets.
    """
    
    def __init__(self, path, dbpath, which='average', tail=75, pulses=None):
        """
        Constructor for the TPDataset.

        Parameters
        ----------
        path: str
            The full path to the input histogram.
        dbpath: str
            The full path to the channel map SQLite file.
        which: str
            Which type of average pulse to return ('positive', 'negative', 'average').
        tail: int
            The time extent beyond the peak to save.
        pulses: np.array
            The array of pulses to copy into this object. For copy constructing.
        """
        if pulses is None:

            if '.root' in path:
                data = uproot.open(path)['tpctestpulseartdaq/output'].values()
                waveforms, counts = data[:-1, :].astype(float), data[-1, :]
                waveforms -= np.median(waveforms, axis=0)
                waveforms[:, counts > 0] /= counts[counts > 0]

                ppeaks = np.argmax(waveforms, axis=0)
                ppeaks[ppeaks < 75] = 75
                ppeaks[ppeaks > waveforms.shape[0] - tail] = waveforms.shape[0] - tail
                mpeaks = np.argmin(waveforms, axis=0)
                mpeaks[mpeaks < 75] = 75
                mpeaks[mpeaks > waveforms.shape[0] - tail] = waveforms.shape[0] - tail
                self.ploc = ppeaks
                self.mloc = mpeaks
                pwaveform = np.vstack([waveforms[x-75:x+tail, xi] for xi, x in enumerate(ppeaks)])
                mwaveform = -1 * np.vstack([waveforms[x-75:x+tail, xi] for xi, x in enumerate(mpeaks)])
                if which == 'average':
                    self.pulses = (pwaveform+mwaveform)[:, :75+tail] / 2.0
                elif which == 'positive':
                    self.pulses = pwaveform[:, :75+tail]
                else:
                    self.pulses = mwaveform[:, :75+tail]
                self.is_pulsed = np.array([self._is_pulsed_channel(x) for x in self.pulses])
            else:
                data = np.load(path)['arr_0']
                waveforms, counts = data[:-1, :].astype(float), data[-1, :]
                waveforms -= np.median(waveforms[:500], axis=0)
                waveforms[:, counts > 0] /= counts[counts > 0]
                self.ploc = np.argmax(waveforms, axis=0)
                self.mloc = np.argmin(waveforms, axis=0)
                self.pulses = waveforms.transpose()
                self.is_pulsed = np.array([self._is_pulsed_channel(x, thr=0.05) for x in self.pulses])

            self.chmap_join_index = np.arange(self.pulses.shape[0])
            self._read_channel_map(dbpath)
        else:
            self.pulses = pulses
            self.is_pulsed = np.array([self._is_pulsed_channel(x) for x in self.pulses])
        
        self.metrics = dict()

    def add_metric(self, func, name, **kwargs) -> None:
        """
        Adds a metric (callable) to the list of metrics to be run on
        each waveform.

        Parameters
        ----------
        func: callable
            The callable that produces the metric. Must map a waveform
            to a single value.
        name: str
            The name to be assigned to the metric.
        kwargs: dict
            Key-word arguments to pass to the function.

        Returns
        -------
        None.
        """
        if name not in self.metrics.keys():
            has_kwargs = kwargs != dict()
            print(f'Metric with name "{name}" has been added{f" (kwargs = {kwargs})" if has_kwargs else ""}.')
            self.metrics[name] = (func, kwargs)
        else:
            print(f'Metric with name "{name}" already present in list. It has not been added!')

    def run_analysis(self) -> pd.DataFrame:
        """
        Runs each metric on each waveform and stores the result in a
        pandas DataFrame object.

        Parameters
        ----------
        None.

        Returns
        -------
        analysis_data: pd.DataFrame
            The resulting metrics stored as a pandas DataFrame.
        """
        data = {'channel_id': self.channel_id,
                'flange_name': self.flange_name,
                'slot_id': self.slot_id,
                'local_id': self.local_id,
                'ploc': self.ploc, 'mloc': self.mloc,
                'is_pulsed': self.is_pulsed}
        for k, v in self.metrics.items():
            res = [v[0](x, **v[1]) if self.is_pulsed[xi] else 0 for xi, x in enumerate(self.pulses)]
            if np.any([isinstance(r, dict) for r in res]):
                keys = res[np.where([isinstance(r, dict) for r in res])[0]].keys()
                for sk in res[0].keys():
                    data[sk] = [r[sk] if sk in r.keys() else 0 for r in res]
            elif np.any([isinstance(r, list) for r in res]) or np.any([isinstance(r, tuple) for r in res]):
                num = len(res[np.array([isinstance(r, list) or isinstance(r, tuple) for r in res])[0]])
                for i in range(num):
                    print(res)
                    data[f'{k}_{i}'] = [r[i] if len(r) > 1 else 0 for r in res]
            else:
                data[k] = [r for r in res]

        self.analysis_data = pd.DataFrame(data)
        return self.analysis_data
    
    def plot_metric_board(self, name, yrange, metric_label, title, mask=None):
        """
        Plots a single metric as a 2D histogram across the common
        readout board grouping.

        Parameters
        ----------
        name: str
            The name of the metric.
        yrange: tuple[float]
            The range of the y-axis as a two-element tuple: (ylow, yhigh).
        metric_label: str
            The label for the metric to be placed on the y-axis
        title: str
            The title to place on the plot.
        mask: np.array
            Boolean array to mask out entries in the DataFrame.

        Returns
        -------
        None.
        """
        plt.style.use('../plot_style.mplstyle')
        figure = plt.figure(figsize=(8,6))
        ax = figure.add_subplot()
        if mask is None:
            ax.hist2d(self.analysis_data['local_id'], self.analysis_data[name], bins=(64,50), range=((0,64), yrange), cmap='Blues')
        else:
            ax.hist2d(self.analysis_data.loc[mask]['local_id'], self.analysis_data.loc[mask][name], bins=(64,50), range=((0,64), yrange), cmap='Blues')
        ax.set_xticks([8*i for i in range(9)])
        ax.set_xlim(0, 64)
        ax.set_ylim(*yrange)
        ax.set_xlabel('Channel Number')
        ax.set_ylabel(metric_label)
        figure.suptitle(title)

    def plot_metric_crate(self, name, yrange, metric_label, title, mask=None):
        """
        Plots a single metric as a 2D histogram across the common
        readout mini-crate grouping.

        Parameters
        ----------
        name: str
            The name of the metric.
        yrange: tuple[float]
            The range of the y-axis as a two-element tuple: (ylow, yhigh).
        metric_label: str
            The label for the metric to be placed on the y-axis
        title: str
            The title to place on the plot.
        mask: np.array
            Boolean array to mask out entries in the DataFrame.

        Returns
        -------
        None.
        """
        plt.style.use('../plot_style.mplstyle')
        figure = plt.figure(figsize=(8,6))
        ax = figure.add_subplot()
        if mask is None:
            ax.hist2d(64 * self.analysis_data['slot_id'] + self.analysis_data['local_id'], self.analysis_data[name], bins=(576, 50), range=((0, 576), yrange), cmap='Blues')
        else:
            ax.hist2d(64 * self.analysis_data.loc[mask]['slot_id'] + self.analysis_data.loc[mask]['local_id'], self.analysis_data.loc[mask][name], bins=(576, 50), range=((0, 576), yrange), cmap='Blues')
        ax.set_xticks([64*i for i in range(10)])
        ax.set_xlim(0, 576)
        ax.set_ylim(*yrange)
        ax.set_xlabel('Channel Number')
        ax.set_ylabel(metric_label)
        figure.suptitle(title)

    @staticmethod
    def _is_pulsed_channel(w, thr=400) -> bool:
        """
        Checks the channel to see if it has been pulsed. The threshold
        parameter can be set sufficiently high to only find primary
        pulsed channels or can be set low enough to also find channels
        with crosstalk.

        Parameters
        ----------
        w: np.array
            The signal waveform.
        thr: float
            The threshold used to determine if the channel has been
            pulsed.

        Returns
        -------
        pulsed: bool
            Boolean tagging the channel as pulsed
        """
        return np.any(w > thr)
    
    def _read_channel_map(self, dbpath) -> None:
        """
        Reads the channel map from the SQLite database and stores the
        relevant information.

        Parameters
        ----------
        dbpath: str
            The full path to the SQLite database containing the channel
            map table.

        Returns
        -------
        None.
        """
        conn = sqlite3.connect(dbpath)
        chmap = pd.read_sql_query("SELECT * FROM channelinfo", conn)
        conn.close()
        chmap['crate_number'] = (chmap['fragment_id']/2 - 2048).astype(int) - 116*((chmap['fragment_id']-4096)/256).astype(int)
        chmap['join_index'] = 576*chmap['crate_number'] + 64*chmap['slot_id'] + chmap['local_id']
        chmap_values = {chmap.iloc[i]['join_index']: i for i in range(len(chmap))}
        keys = ['channel_id', 'flange_name', 'slot_id', 'local_id']
        defaults = [-1, 'None', -1, -1]
        chmap_data = [chmap.iloc[chmap_values[x]][keys] if x in chmap_values.keys() else defaults for x in self.chmap_join_index]
        self.channel_id = np.array([x[0] for x in chmap_data], dtype='int')
        self.flange_name = np.array([x[1] for x in chmap_data], dtype='str')
        self.slot_id = np.array([x[2] for x in chmap_data], dtype='int')
        self.local_id = np.array([x[3] for x in chmap_data], dtype='int')
    
    def __repr__(self) -> str:
        """
        Returns a string describing the characteristics of the
        TPDataset.

        Returns
        -------
        s: str
            The string describing the TPDataset.
        """
        s = f'Total Channels: {self.pulses.shape[0]}'
        s += f'\nPulsed Channels: {np.sum(self.is_pulsed)}'
        return s

    def __sub__(self, other):
        """
        Subtracts two TPDatasets channel-by-channel.

        Parameters
        ----------
        other: TPDataset
            The TPDataset to be subtracted from this one.
        
        Returns
        -------
        result: TPDataset
            The resulting difference between the two datasets.
        """
        if self.pulses.shape != other.pulses.shape:
            raise Exception(f'Datasets do not having matching shapes. Shapes are {self.pulses.shape} and {other.pulses.shape}.')
        result = TPDataset('', '', which='', pulses=self.pulses - other.pulses)
        result.ploc = self.ploc
        result.mloc = self.mloc
        result.channel_id = self.channel_id
        result.flange_name = self.flange_name
        result.slot_id = self.slot_id
        result.local_id = self.local_id

    def __add__(self, other):
        """
        Adds two TPDatasets channel-by-channel.

        Parameters
        ----------
        other: TPDataset
            The TPDataset to be added to this one.
        
        Returns
        -------
        result: TPDataset
            The resulting addition between the two datasets.
        """
        if self.pulses.shape != other.pulses.shape:
            raise Exception(f'Datasets do not having matching shapes. Shapes are {self.pulses.shape} and {other.pulses.shape}.')
        result = TPDataset('', '', which='', pulses=self.pulses + other.pulses)
        result.ploc = self.ploc
        result.mloc = self.mloc
        result.channel_id = self.channel_id
        result.flange_name = self.flange_name
        result.slot_id = self.slot_id
        result.local_id = self.local_id

    def join(self, other):
        """
        Joins two TPDatasets channel-by-channel where at least one of
        the two channels are pulsed. If both are tagged as pulsed, the
        default is to use pulse from the first (this) object.

        Parameters
        ----------
        other: TPDataset
            The TPDataset to be joined to this one.
        
        Returns
        -------
        result: TPDataset
            The result of joining the two datasets.
        """
        if self.pulses.shape != other.pulses.shape:
            raise Exception(f'Datasets do not having matching shapes. Shapes are {self.pulses.shape} and {other.pulses.shape}.')
        pulses = self.pulses
        replace_mask = ~self.is_pulsed & other.is_pulsed
        pulses[replace_mask] = other.pulses[replace_mask]
        result = TPDataset('', '', which='', pulses=pulses)
        result.ploc = self.ploc
        result.ploc[replace_mask] = other.ploc[replace_mask]
        result.mloc = self.mloc
        result.mloc[replace_mask] = other.mloc[replace_mask]
        result.channel_id = self.channel_id
        result.flange_name = self.flange_name
        result.slot_id = self.slot_id
        result.local_id = self.local_id
        return result