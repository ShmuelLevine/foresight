# data_functions.py

import pandas as pd
import numpy as np


def series_to_supervised(data,
                         n_in=1,
                         n_out=1,
                         dropnan=True,
                         separate_output_series=False):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if (type(data) is list or
                   (type(data) in [pd.DataFrame, pd.Series, np.ndarray]
                    and len(data.shape) == 1)) else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    outnames = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    if not separate_output_series:
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    else:

        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                outnames += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                outnames += [('var%d(t+%d)' % (j + 1, i))
                             for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = [*names, *outnames]
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    if separate_output_series:
        out = pd.DataFrame()
        for col in outnames:
            out[col] = agg[col]
            del agg[col]
        return agg.to_numpy(), out.to_numpy()
    return agg


def data_transformer(data,
                     transform=None,
                     remove_outliers=False,
                     scaler='MinMaxScaler'):
    """
    General input data transformation function

    :param data: Input dataset to be transformed.  This data is expected to be a 1D array of datapoints,
        already resampled and with the weekends removed from the data
    :type data: :class:`numpy.ndarray`

    :param transform: transformation make the data roughly stationary
    :type transform: valid values are [None, 'Diff', 'LogDiff']

    :param remove_outliers: a number denoting the number of sigmas for the cutoff or else false to indicate
        that outliers should not be removed
    :type remove_outliers: `integer` or False

    :param scaler: the type of scaler to use for the data to change the range of values
    :type scaler: [\'MinMaxScaler\' or None ]
"""

    import numpy as np
    import pandas as pd
    import sklearn.preprocessing as skpp

    assert (
        len(data.shape) == 1
    )  # The data should only be a 1D array, with the datetime stripped off

    # difference the data. numpy.diff automatically trims off the leading nan
    if transform == 'Diff':
        data_ = np.diff(data)
    elif transform == 'LogDiff':
        data_ = np.diff(np.log(data))
    elif transform == None:
        pass
    else:
        raise ValueError(
            'transform must be either \'Diff\', \'LogDiff\', or None')

    # optionally remove outliers. If the value of the parameter passed is not False (or equivilent), replace outlying data
    # with backfilled data - i.e. replace the outlier with the next valid value
    if remove_outliers:
        data_[np.abs(data_ - np.mean(data_)) > (remove_outliers *
                                                np.std(data_))] = np.nan
        data_ = pd.Series(data_).fillna(method='bfill').to_numpy()

    if scaler == 'MinMaxScaler':
        scaler_ = skpp.MinMaxScaler((-1, 1))
        data_ = scaler_.fit_transform(data_.reshape(-1, 1))

    return data_, scaler_
