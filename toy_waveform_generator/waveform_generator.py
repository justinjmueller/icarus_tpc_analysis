import numpy as np

def generate_waveform(data, channel_id) -> np.array:
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
    cphase = np.random.uniform(0, 2*np.pi, ffts.shape[1])
    camplitude = np.sqrt(ffts[2, :] * (0.4 / 4096))
    cwaveform  = np.sum(camplitude[:, None] * np.sin(frequency[:, None] * time[None, :] + cphase[:, None]), axis=0)

    # Intrinsic component
    iphase = np.random.uniform(0, 2*np.pi, ffts.shape[1])
    iamplitude = np.sqrt(ffts[1, :] * (0.4 / 4096))
    iwaveform  = np.sum(iamplitude[:, None] * np.sin(frequency[:, None] * time[None, :] + iphase[:, None]), axis=0)

    waveform = np.rint(cwaveform + iwaveform)
    return waveform