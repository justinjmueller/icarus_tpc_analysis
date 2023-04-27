import numpy as np
import pandas as pd

def calculate_percentiles(data, key) -> tuple[float, float]:
    """
    Calculates the error bars that span 68.27% of the distribution about
    the median.

    Parameters
    ----------
    data: pandas.DataFrame
        The DataFrame with the column to be used in the calculation.
        Assumes that the DataFrame has already been restricted to the
        relevant population (e.g. wire plane, crate).
    key: str
        The name of the column in the DataFrame to be used in the
        calculation.

    Returns
    -------
    limits: tuple[float, float]
        A tuple containing the asymmetric error bar widths corresponding
        to 68.27% of the distribution about the median.
    """
    limits = np.percentile(data[key], 50-34.135), np.percentile(data[key], 50+34.135)
