import numpy as np
import matplotlib.pyplot as plt
import uproot

def load_waveform_file(path, run, fragment, evt) -> np.array:
    """
    Loads waveforms from a file. Input files should be comma separated
    with each waveform in the crate having its own separate line in
    the file.

    Parameters
    ----------
    path: str
        The base path to the input files.
    run: int
        The run number.
    fragment: int
        The integer representation of the fragment ID.
    evt: int
        The event number.

    Returns
    -------
    waveforms: numpy.array
        The waveforms for the component with shape (576,4096) (nominal)
        or (512, 4096) (top crate).
    """
    with open(f'{path}run{run}_frag{fragment}_evt{evt}') as input_file:
        lines = input_file.readlines()
    waveforms = np.array([x.strip('\n').split(',') for x in lines], dtype=float)
    return waveforms

def align_and_average(waveforms, max=True, size=1250) -> np.array:
    """
    Calculate the average pulse. First, split the waveforms into
    separate, individual pulses. Afterwards, alignment using the min
    or max of the each piece is performed by "rolling" the waveform.
    The final step is to calculate the mean of each piece.

    Parameters
    ----------
    waveforms: numpy.array
        The input waveforms with shape (N, 4096).
    max: bool
        Boolean flag for using max(min) for alignment.
    size: int
        The number of ticks comprising a single pulse.

    Returns
    -------
    waveform: numpy.array
        The resulting average waveform.
    """
    waveforms = np.vstack(np.split(waveforms, [size*i for i in range(1, int(4096/size)+1)]))
    waveform = np.mean(np.roll(waveforms, np.argmax(waveforms, axis=1), axis=0), axis=0)
    tcenter = np.argmax(waveform) if max else np.argmin(waveform)
    waveform = np.roll(waveform, 75-tcenter)[0:150]
    return waveform

def average_pulse(path, run, fragment, ch, title, evt=1, scale=2200, internal=False, nchannels=64) -> None:
    """
    Plots the average test pulse shape for both the positive lobe and
    the inverted negative lobe. Waveform input files should be comma
    separated with each waveform in a crate having its own separate
    line in the file.

    Parameters
    ----------
    path: str
        The base path to the input files.  
    run: int
        The run number of the test pulse input.
    fragment: int
        The integer representation of the fragment ID.
    ch: int
        The first channel in the pulsed board. If external, the entire
        board is used. If internal, only the channels on the board
        congruent to the channel mod 2 are used.
    title: str
        The title of the plot
    evt: int
        The number of the event.
    scale: float
        The y-range of the plot. The range will be set to (-250, scale).
    internal: bool
        Boolean flag denoting internal vs. external test pulse.
    nchannels: int
        Number of channels to range over for the averaging. 

    Returns
    -------
    None.
    """
    plt.style.use('../plot_style.mplstyle')

    figure = plt.figure(figsize=(8,6))
    ax = figure.add_subplot()

    waveforms = load_waveform_file(path, run, fragment, evt)[ch:ch+nchannels:2 if internal else 1,:]
    pwaveform = align_and_average(waveforms, max=True, size=1250)
    mwaveform = -1 * align_and_average(waveforms, max=False, size=1250)

    ax.plot(np.arange(150), pwaveform, linestyle='-', linewidth=2, label='Positive Lobe')
    ax.plot(np.arange(150), mwaveform, linestyle='-', linewidth=2, label='Negative Lobe')
    ax.set_xlim(0,150)
    ax.set_ylim(-250, scale)
    ax.set_xlabel('Time [ticks]')
    ax.set_ylabel('Waveform Height [ADC]')
    ax.legend()
    figure.suptitle(title)

def average_waveform(path, run, fragment, ch, title, evt=1, scale=2200, internal=False, nchannels=64):
    """
    Plots the average test pulse waveform. Waveform input files should be comma
    separated with each waveform in a crate having its own separate
    line in the file.

    Parameters
    ----------
    path: str
        The base path to the input files.  
    run: int
        The run number of the test pulse input.
    fragment: int
        The integer representation of the fragment ID.
    ch: int
        The first channel in the pulsed board. If external, the entire
        board is used. If internal, only the channels on the board
        congruent to the channel mod 2 are used.
    title: str
        The title of the plot
    evt: int
        The number of the event.
    scale: float
        The y-range of the plot. The range will be set to (-scale, scale).
    internal: bool
        Boolean flag denoting internal vs. external test pulse.
    nchannels: int
        Number of channels to range over for the averaging. 

    Returns
    -------
    None.
    """
    plt.style.use('../plot_style.mplstyle')

    figure = plt.figure(figsize=(14,6))
    ax = figure.add_subplot()

    waveforms = load_waveform_file(path, run, fragment, evt)[ch:ch+nchannels:2 if internal else 1,:]
    waveform = np.mean(np.roll(waveforms, np.argmax(waveforms, axis=1), axis=0), axis=0)
    
    ax.plot(np.arange(4096), waveform, linestyle='-', linewidth=1)
    ax.set_xlim(0,4096)
    ax.set_ylim(-scale, scale)
    ax.set_xlabel('Time [ticks]')
    ax.set_ylabel('Waveform Height [ADC]')
    figure.suptitle(title)

def compare_average_pulse(path, runs, fragments, chs, strides, nchannels, labels, title, evt=1, scale=2200) -> None:
    """
    Plots the average test pulse shapes for each of the input runs.
    Waveform input files should be comma separated with each waveform
    a crate having its own separate line in the file.

    Parameters
    ----------
    path: str
        The base path to the input files.
    runs: list[int]
        The list of run numbers for the test pulse input.
    fragments: list[int]
        The list of the fragment IDs of the pulsed crates.
    chs: list[int]
        The list of the first channels in each of the pulsed crates.
    strides: list[int]
        The list of strides to use when selecting channels from the
        input waveforms (e.g. internal test pulses use every other
        channel, so stride=2).
    nchannels: list[int]
        The list of the number of channels to range over for the
        averaging.
    labels: list[str]
        The list of labels to use in the legend.
    title: str
        The title of the plot.
    evt: int
        The number of the event.
    scale: float
        The y-range of the plot. The range will be set to (-250, scale).

    Returns
    -------
    None.
    """

    plt.style.use('../plot_style.mplstyle')

    figure = plt.figure(figsize=(8,6))
    ax = figure.add_subplot()

    for ri, r in enumerate(runs):
        waveforms = load_waveform_file(path, r, fragments[ri], evt)[chs[ri]:chs[ri]+nchannels[ri]:strides[ri],:]
        waveform = align_and_average(waveforms, max=True, size=1250)
        ax.plot(np.arange(150), waveform, linestyle='-', linewidth=2, label=labels[ri])

    ax.set_xlim(0,150)
    ax.set_ylim(-250, scale)
    ax.set_xlabel('Time [ticks]')
    ax.set_ylabel('Waveform Height [ADC]')
    ax.legend()
    figure.suptitle(title)

def plot_single_test_waveform(path, run, fragment, title, evt, ch, scale=2200) -> None:
    """
    Plots a single waveform. Waveform input files should be comma
    separated with each waveform in a crate having its own separate
    line in the file.

    Parameters
    ----------
    path: str
        The base path to the input files.
    run: int
        The run number.
    fragment: int
        The integer representation of the fragment ID.
    title: str
        The title to place on the plot.
    scale: float
        The y-range of the plot. The range will be set to (-scale, scale).
    """
    plt.style.use('../plot_style.mplstyle')

    figure = plt.figure(figsize=(16,6))
    ax = figure.add_subplot()

    waveform = load_waveform_file(path, run, fragment, evt)[ch, :]
    ax.plot(np.arange(4096), waveform, linestyle='-', linewidth=2)
    ax.set_xlim(0,4096)
    ax.set_ylim(-scale, scale)
    ax.set_xlabel('Time [ticks]')
    ax.set_ylabel('Waveform Height [ADC]')
    figure.suptitle(title)

def plot_average_waveform_artdaq(path, run, title, channel=0, scale=2200):
    """
    Plots the average waveform for the specified channel using a ROOT
    file prepared by the TPCTestPulseArtDAQ module. This module aligns
    waveforms and stores them (as a sum) in a TH2I with a set of bins
    for each channel.

    Parameters
    ----------
    path: str
        The base path to the input ROOT file.
    run: int
        The run number of the input ROOT file.
    title: str
        The title to place on the plot.
    channel: int
        The channel number to retrieve and plot.
    scale: float
        The y-range of the plot. The range will be set to (-scale, scale).
    """
    plt.style.use('../plot_style.mplstyle')
    
    figure = plt.figure(figsize=(16,6))
    ax = figure.add_subplot()
    
    data = uproot.open(f'{path}waveforms_run{run}.root')['tpctestpulseartdaq/output'].values()
    waveform, count = data[:-1, channel].astype(float), data[-1,channel]
    waveform -= np.median(waveform)
    waveform /= count
    ax.plot(np.arange(len(waveform)), waveform, linestyle='-', linewidth=2)
    ax.set_xlim(0, len(waveform))
    ax.set_ylim(-scale, scale)
    ax.set_xlabel('Time [ticks]')
    ax.set_ylabel('Waveform Height [ADC]')
    figure.suptitle(title)

def plot_average_pulse_artdaq(path, run, title, channel=0, scale=2200):
    """
    Plots the average waveform for the specified channel using a ROOT
    file prepared by the TPCTestPulseArtDAQ module. This module aligns
    waveforms and stores them (as a sum) in a TH2I with a set of bins
    for each channel.

    Parameters
    ----------
    path: str
        The base path to the input ROOT file.
    run: int
        The run number of the input ROOT file.
    title: str
        The title to place on the plot.
    channel: int
        The channel number to retrieve and plot.
    scale: float
        The y-range of the plot. The range will be set to (-250, scale).
    """
    plt.style.use('../plot_style.mplstyle')
    
    figure = plt.figure(figsize=(8,6))
    ax = figure.add_subplot()
    
    data = uproot.open(f'{path}waveforms_run{run}.root')['tpctestpulseartdaq/output'].values()
    waveform, count = data[:-1, channel].astype(float), data[-1,channel]
    waveform -= np.median(waveform)
    waveform /= count

    pwaveform = waveform[:150]
    mpeak = np.argmin(waveform)
    mwaveform = -1 * waveform[mpeak-75:mpeak+75]
    ax.plot(np.arange(150), pwaveform, linestyle='-', linewidth=2, label='Positive Lobe')
    ax.plot(np.arange(150), mwaveform, linestyle='-', linewidth=2, label='Negative Lobe')
    ax.set_xlim(0, 150)
    ax.set_ylim(-250, scale)
    ax.set_xlabel('Time [ticks]')
    ax.set_ylabel('Waveform Height [ADC]')
    ax.legend()
    figure.suptitle(title)