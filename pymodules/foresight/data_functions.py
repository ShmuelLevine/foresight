# data_functions.py

import pandas as pd
import numpy as np
import foresight.util as fxu
import sklearn.preprocessing as skpp

def series_to_supervised(data,
                         n_in=1,
                         n_out=1,
                         dropnan=True,
                         separate_output_series=True):
    """
    Frame a time series as a supervised learning dataset.
    
    :param data: Sequence of observations as a list or NumPy array.
    :type data: list or class:`numpy.array`
    
    :param n_in: The number of lag observations to use as input
    :type n_in: int
    
    :param n_out: The number of lag observations to use as output
    :type n_out: int
    
    :param dropnan: Denotes whether or not to drop rows with NaN values
    :type dropnan: bool

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
    return agg, None


def clean_data(data, remove_duplicates = True, sample_frequency = '1T', sample_type = 'nearest', remove_weekends = True):
    """
    Clean dataset for use in timeseries model
    
    :param data: class:`pandas.dataframe` containing timeseries data, with the datetime as the index.
    :type data:  class:`pandas.dataframe`
    
    :param remove_duplicates: Denotes whether to remove duplicate indices.  This should probably always be True
    :type remove_duplicates: boolean
    
    :param sample_frequency: The frequency to sample data, expressed as a string with pandas notation
    :type sample_frequency: str
    
    :param sample_type: Strategy to use for sampling when data do not exist for a particular datetime
    :type sample_type: boolean

    :param remove_weekends: Denotes whether to remove rows for weekends, when there is no trading.  This should probably always be True
    :type remove_weekends: boolean
    
    Returns:
        Pandas DataFrame of cleaned timeseries
        
    """

    if (remove_duplicates):
        d1 = data[~data.index.duplicated()]
    else:
        d1 = data

    # resample the data, using the nearest value
    if (sample_type == 'nearest'):
        d2 = d1.resample(sample_frequency).nearest()
    elif (sample_type == 'pad'):
        d2 = d1.resample(sample_frequency).pad()
    elif (sample_type == 'bfill'):
        d2 = d1.resample(sample_frequency).bfill()
    else:
        raise ValueError('sample_type must be one of [\'nearest\', \'pad\', \'bfill\']')

    # remove the weekends when there is no trading
    if (remove_weekends):
        d3 = d2[d2.index.dayofweek < 5]
    else:
        d3 = d2
    
    return d3


class Data_Transformer():
    """
    General input data transformation function

    """
    
    def __init__(    self, 
                     transform=None,
                     remove_outliers=False,
                     scaler='MinMaxScaler'):

        """
        Data_Transformer Constructor
        
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

        self.transform = transform
        self.remove_outliers = remove_outliers
        self.scaler = scaler

        
        fxu.ValidateType(remove_outliers,
                         reqd_type=int,
                         arg_name='remove_outliers',
                         err_msg='must be None or integers',
                         allow_none=True)

    def Invert(self, data):
        return self.scaler.inverse_transform(data.reshape(-1,1))
    
    def __call__(self, data, remove_outliers = None):

        
        
        assert (
            len(data.shape) == 1
        )  # The data should only be a 1D array, with the datetime stripped off

        # difference the data. numpy.diff automatically trims off the leading nan
        if self.transform == 'Diff':
            data_ = np.diff(data)
        elif self.transform == 'LogDiff':
            data_ = np.diff(np.log(data))
        elif self.transform == None:
            pass
        else:
            raise ValueError(
                'transform must be either \'Diff\', \'LogDiff\', or None')

        # optionally remove outliers. If the value of the parameter passed is not False (or equivilent), replace outlying data
        # with backfilled data - i.e. replace the outlier with the next valid value
        # Convert remove_outliers to an int, to avoid issues in case True is passed as an argument
        if remove_outliers is None and self.remove_outliers:
            data_[np.abs(data_ - np.mean(data_)) > (int(self.remove_outliers) *
                                                    np.std(data_))] = np.nan
        data_ = pd.Series(data_).fillna(method='bfill').to_numpy()

        if hasattr(self.scaler, 'fit_transform'):
            scaler_ = self.scaler
        
        # if the scaler parameter is the string 'MinMaxScaler', then create and fit a new sklearn.preprocessing.MinMaxScaler
        # instance and replace self.scaler with this new MinMaxScaler instance.
        # This ensures that the transformer can be reused on different data but still performing the same scaling operation
        
        elif self.scaler == 'MinMaxScaler':
            scaler_ = skpp.MinMaxScaler((-1, 1))
            self.scaler = scaler_.fit(data_.reshape(-1, 1))
        else:
            raise ValueError(
                'scaler must be either a sklearn.preprocessing.Scaler object or else \'MinMaxScaler\'')

        data_ = self.scaler.transform(data_.reshape(-1,1))

        return data_, scaler_
    
    
