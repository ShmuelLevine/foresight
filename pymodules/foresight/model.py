# model.py

from tensorflow.keras import Model as tf_model
import numpy
import pandas
#from numpy import array as np_array
from pandas import Timedelta as TD
from collections import Callable
import foresight.util as fxu


class Model:
    """ Defines a class which encapsulates a Keras.Model object used for backtesting and trading"""
    def __init__(self, model, data, data_freq, seq_len, data_transform=None):
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

        self.model = model
        self.data = data
        self.data_freq = data_freq
        self.data_transform = data_transform
        self.seq_len = seq_len

    def Fit(self, batch_size=128, epochs=2000):
        pass

    def AddDatum(self, datum):
        if (datum is None) or (not isinstance(datum, float)):
            raise TypeError('datum must be a float type')
