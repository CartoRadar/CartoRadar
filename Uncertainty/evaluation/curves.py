import numpy as np
from typing import Tuple


def get_CDF_statistics(
    error_arr: np.ndarray, selection_arr: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the CDF statistics for plotting.
    Args:
        error_arr: a 1D array of the data
        selection_arr: a 1D bool array that has the same shape of error_arr. It indicates
            the elements in the array to be used.
    Returns:
        xdata: the x-axis of the CDF plot
        ydata: the percentage data, y-axis of the CDF plot
    """
    if selection_arr is not None:
        error_arr = error_arr[selection_arr]

    count, bins_count = np.histogram(error_arr, bins=1000)
    pdf = count / sum(count)
    cdf = np.insert(np.cumsum(pdf), 0, 0)
    return bins_count, cdf
