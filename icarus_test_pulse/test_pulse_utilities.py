import numpy as np
import matplotlib.pyplot as plt

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

def average_pulse(path, run, fragment, ch, title, evt=1, scale=2200, internal=False):
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

    Returns
    -------
    None.
    """
    plt.style.use('../plot_style.mplstyle')

    figure = plt.figure(figsize=(8,6))
    ax = figure.add_subplot()

    waveforms = load_waveform_file(path, run, fragment, evt)[ch:ch+64:2 if internal else 1,:]
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

def average_waveform(path, run, fragment, ch, title, evt=1, scale=2200, internal=False):
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
    """
    plt.style.use('../plot_style.mplstyle')

    figure = plt.figure(figsize=(14,6))
    ax = figure.add_subplot()

    waveforms = load_waveform_file(path, run, fragment, evt)[ch:ch+64:2 if internal else 1,:]
    waveform = np.mean(np.roll(waveforms, np.argmax(waveforms, axis=1), axis=0), axis=0)
    
    ax.plot(np.arange(4096), waveform, linestyle='-', linewidth=1)
    ax.set_xlim(0,4096)
    ax.set_ylim(-scale, scale)
    ax.set_xlabel('Time [ticks]')
    ax.set_ylabel('Waveform Height [ADC]')
    figure.suptitle(title)