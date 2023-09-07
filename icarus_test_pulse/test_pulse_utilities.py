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

def get_average_pulses_artdaq(path, run, which='average'):
    """
    Calculates the average pulse (for either signal lobe or the average)
    of each channel in the input ROOT histogram.

    Parameters
    ----------
    path: str
        The base path to the input ROOT file.
    run: int
        The run number of the input ROOT file.
    which: str
        Which type of average pulse to return ('positive', 'negative', 'average').

    Returns
    -------
    pulses: numpy.array
        The average pulse for each channel in the ROOT histogram.
        Shape = (nchannels=576, nticks=150).
    """

    data = uproot.open(f'{path}waveforms_run{run}.root')['tpctestpulseartdaq/output'].values()
    waveforms, counts = data[:-1, :].astype(float), data[-1, :]
    waveforms -= np.median(waveforms, axis=0)
    waveforms[:, counts > 0] /= counts[counts > 0]

    ppeaks = np.argmax(waveforms, axis=0)
    ppeaks[ppeaks < 75] = 75
    mpeaks = np.argmin(waveforms, axis=0)
    mpeaks[mpeaks < 75] = 75
    pwaveform = np.vstack([waveforms[x-75:x+75, xi] for xi, x in enumerate(ppeaks)])
    mwaveform = -1 * np.vstack([waveforms[x-75:x+75, xi] for xi, x in enumerate(mpeaks)])
    if which == 'average':
        pulses = (pwaveform+mwaveform)[:, :150] / 2.0
    elif which == 'positive':
        pulses = pwaveform[:, :150]
    else:
        pulses = mwaveform[:, :150]

    return pulses

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
    print(f'Count: {count}')
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
    awaveform = (pwaveform + mwaveform) / 2.0
    print(f'Max: {np.max(waveform)} [ADC]')
    print(f'Integral: {np.sum(awaveform) * 0.4:.2f} [ADC * us]')
    ax.plot(np.arange(150), pwaveform, linestyle='-', linewidth=2, label='Positive Lobe')
    ax.plot(np.arange(150), mwaveform, linestyle='-', linewidth=2, label='Negative Lobe')
    ax.plot(np.arange(150), awaveform, linestyle='-', linewidth=2, label='Average')
    ax.set_xlim(0, 150)
    ax.set_ylim(-250, scale)
    ax.set_xlabel('Time [ticks]')
    ax.set_ylabel('Waveform Height [ADC]')
    ax.legend()
    figure.suptitle(title)

def plot_all_connectors(path, run, fragment, title, scale=2200):
    """
    Wrapper function for plotting a single test waveform for each connector.

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

    Returns
    -------
    None.
    """
    plot_single_test_waveform(path, run, fragment, f'Run {run}: {title}CONN 3)', 1, 0, scale=1600)   # Connector 3
    plot_single_test_waveform(path, run, fragment, f'Run {run}: {title}CONN 1)', 1, 32, scale=1600)  # Connector 1
    plot_single_test_waveform(path, run, fragment, f'Run {run}: {title}CONN 4)', 1, 512, scale=1600) # Connector 2
    plot_single_test_waveform(path, run, fragment, f'Run {run}: {title}CONN 2)', 1, 544, scale=1600) # Connector 4

def get_pulse_statistics_artdaq(path, run):
    """
    Prints a list of the connectors (assuming a standard connection)
    which are actively being pulsed.

    Parameters
    ----------
    path: str
        The base path to the input ROOT file.
    run: int
        The run number of the input ROOT file.

    Returns
    -------
    None.
    """
    data = uproot.open(f'{path}waveforms_run{run}.root')['tpctestpulseartdaq/output'].values()
    waveforms, count = data[:-1, :].astype(float), data[-1, :]
    waveforms -= np.median(waveforms, axis=0)
    channels = np.arange(576)
    mask_threshold = np.any(waveforms > 1.75e5, axis=0)

    conn1_mask = ((channels % 64 >= 32) & (channels < 512))
    conn2_mask = ((channels % 64 >= 32) & (channels >= 512))
    conn3_mask = ((channels % 64 < 32) & (channels < 512))
    conn4_mask = ((channels % 64 < 32) & (channels >= 512))

    print(f'Run {run}: ({np.sum(mask_threshold & conn1_mask) > 128}, {np.sum(mask_threshold & conn2_mask) > 16}, {np.sum(mask_threshold & conn3_mask) > 128}, {np.sum(mask_threshold & conn4_mask) > 16})')