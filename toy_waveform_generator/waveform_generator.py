import numpy as np

def generate_waveform(data, channel_id, cphase=None) -> np.array:
    """
    Generates a toy waveform for the specified channel using the
    frequency characteristics defined in the input Dataset. This models
    both coherent noise (at the level of 64 channels) and intrinsic
    noise separately.

    Parameters
    ----------
    data: Dataset
        The input dataset containing the FFTs/frequency characteristics
        of the toy model.
    channel_id: int
        The desired channel number for the toy model.
    cphase: np.array
        The array of phases to use for each frequency component of the
        correlated noise component. If None, then an array is generated
        randomly within the function call. This option allows for noise
        to be correlated across channels.
    
    Returns
    -------
    waveform: np.array
        The resulting waveform with shape = (4096,) and dtype = int.
    """
    
    group = data.chmap.query(f'channel_id == {channel_id}').iloc[0]['group']
    ffts = data.get_ffts(group)

    # Time and frequency
    time = np.arange(4096)
    frequency = (1.0 / (0.4 * 4096)) * np.arange(ffts.shape[1])

    # Coherent component
    if cphase is None:
        cphase = np.random.uniform(0, 2*np.pi, ffts.shape[1])
    camplitude = np.sqrt(ffts[2, :] * (0.4 / 4096))
    cwaveform  = np.sum(camplitude[:, None] * np.sin(frequency[:, None] * time[None, :] + cphase[:, None]), axis=0)

    # Intrinsic component
    iphase = np.random.uniform(0, 2*np.pi, ffts.shape[1])
    iamplitude = np.sqrt(ffts[1, :] * (0.4 / 4096))
    iwaveform  = np.sum(iamplitude[:, None] * np.sin(frequency[:, None] * time[None, :] + iphase[:, None]), axis=0)

    waveform = np.rint(cwaveform + iwaveform)
    return waveform

def generate_component(data, component) -> np.array:
    """
    Generates a set of toy waveforms for the specified component
    using the frequency characteristics defined in the input Dataset.
    This models both coherent noise (at the level of 64 channels) and
    intrinsic noise separately. Coherent noise is correlated by
    choosing the same phase for each board.

    Parameters
    ----------
    data: Dataset
        The input dataset containing the FFTs/frequency characteristics
        of the toy model.
    component: str
        The name of the component for the toy model.

    Returns
    -------
    waveforms: np.array
        The resulting waveforms for the component with shape = (N,4096)
        and dtype = int (N is the number of channels in the component).
    """
    mask = data.chmap['flange'] == component
    channels = data.chmap.loc[mask][['channel_id', 'readout_board_slot', 'channel_number', 'group']].to_numpy()
    channels[:,2] = 64*channels[:,1] + channels[:,2]
    channels = channels[np.argsort(channels[:,2]), :]
    unique_boards = np.unique(channels[:,1])
    unique_groups = np.unique(channels[:,3])
    nboards = unique_boards.shape[0]    
    sort_mask = np.argsort(unique_groups)
    groups = unique_groups[sort_mask]
    
    channel_to_group = {x[0]: np.argwhere(groups == x[3]) for x in channels}
    cphase = np.random.uniform(0, 2*np.pi, 2049*nboards).reshape((nboards,2049))
    waveforms = np.vstack([generate_waveform(data, x[0], cphase[:,channel_to_group[x[0]]]) for x in channels])
    return waveforms
