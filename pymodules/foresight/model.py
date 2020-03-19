# model.py

from tensorflow.keras import Model as tf_model
import numpy
import pandas
#from numpy import array as np_array
from pandas import Timedelta as TD
from collections.abc import Callable
import foresight.util as fxu
import string


class Model:
    """
    Defines a class which encapsulates a Keras.Model object used for backtesting and
    trading

    :param model: A handle to a :class:`tensorflow.keras.Model` object which represents
        the model being trained/used
    :type model: class:`tensorflow.keras.Model`

    :param data: A handle to a :class:`numpy.ndarray` containing the data for the model.
        The data should either be a 1D array containing just the bid prices or a multi-dim array
        otherwise including fields for datetime, bid, and (optionally) ask prices. If the array
        is 1D, it will be assumed that the data has already been properly sampled
    :type data: class:`numpy.ndarray`

    :param data_freq: A :class:`pandas.Datetime` object denoting the frequency to use for
        the input data
    :type data_freq: class:`pandas.Datetime`

    :param seq_len: The number of elements to use for each sequence for the LTSM model
    :type seq_len: int

    :param data_transform: A function object that contains the necessary transformations
        to perform on the data before using it in the model.  The same transformations will be
        used for additional data (used for backtesting and trading).  The return type of this function
        must be a tuple of :class:`numpy.ndarray` and a :class:`pandas.MinMaxScaler` or `None`
    :type data_transform: class:`collections.abc.Callable`

    """
    def __init__(self,
                 model,
                 data,
                 data_freq,
                 seq_len,
                 data_transform=None,
                 stationary_transform=None):

        fxu.ValidateType(
            model,
            arg_name='model',
            reqd_type=tf_model,
            err_msg='must be an instance of tensorflow.keras.Model')

        fxu.ValidateType(data,
                         arg_name='data',
                         reqd_type=numpy.ndarray,
                         err_msg='must be an instance of numpy.array')

        fxu.ValidateType(data_freq,
                         arg_name='data_freq',
                         reqd_type=TD,
                         err_msg='must be an instance of pandas.Timedelta')

        fxu.ValidateType(seq_len,
                         arg_name='seq_len',
                         reqd_type=int,
                         err_msg='must be an integer')

        fxu.ValidateType(data_transform,
                         arg_name='data_transform',
                         val_func=lambda x: callable(x),
                         err_msg='must be a callable type',
                         allow_none=True)

        fxu.ValidateType(stationary_transform,
                         reqd_type=str,
                         arg_name='stationary_transform',
                         err_msg='must [None, \'Diff\', \'LogDiff\']',
                         allow_none=True)

        # We need to store the entire transformed data to use with the model, but only enough
        # raw data to generate the next sequence when a new datum is added
        self.model = model
        self.rawdata = data[-seq_len:]  # store the last seq_len points
        self.data_freq = data_freq
        self.data_transform = data_transform
        self.data, self.scaler = self.data_transform(
            data)  # store the transformed data
        self.seq_len = seq_len

    def Fit(self, batch_size=128, epochs=2000):
        pass

    def AddDatum(self, datum):
        if (datum is None) or (not isinstance(datum, float)):
            raise TypeError('datum must be a float type')
