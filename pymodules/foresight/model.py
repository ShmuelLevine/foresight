# model.py

from tensorflow.keras import Model as tf_model
import numpy
import pandas
from numpy import array as np_array
from  pandas import Timedelta as TD

class Model:
    """ Defines a class which encapsulates a Keras.Model object used for backtesting and trading"""
    def __init__(self, model, data, data_freq):
        if (model is None) or (not isinstance(model, tf_model)): 
            raise TypeError(' \'model\' must be an instance of tensorflow.keras.Model')
        if (data is None) or (not isinstance(data, numpy.ndarray)): 
            raise TypeError('\'data\' must be an instance of numpy.array')
        if (data_freq is None) or (not isinstance(data_freq, pandas.Timedelta)): 
            raise TypeError('\'data_freq\' must be an instance of pandas.Timedelta ')

        self.model = model
        self.data = data
        self.data_freq = data_freq
        