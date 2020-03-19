# model.py

from tensorflow.keras import Model as tf_model
import numpy
import pandas
#from numpy import array as np_array
from pandas import Timedelta as TD
from collections.abc import Callable
import foresight.util as fxu
import string
import foresight.data_functions as fx_df


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

    :param forecast_horizon: The number of timesteps to forecast for each input sequence
    :type forecast_horizon: int

    """
    def __init__(self,
                 model,
                 data,
                 data_freq,
                 seq_len,
                 forecast_horizon=1,
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
        self._model = model
        self._rawdata = data[-seq_len:]  # store the last seq_len points
        self._data_freq = data_freq
        self._data_transform = data_transform
        self._seq_len = seq_len

        assert (forecast_horizon == 1,
                'Code cannot handle forecast horizon other than 1 right now')

        # store the transformed data in an intermediate variable, as the data needs to be converted
        # into supervised learning data
        data_inter, self._scaler = self.data_transform(data)

        # convert the input data into 2 matrices representing inputs and expected outputs, to use for
        # model training
        self._in_data, self._out_data = fx_df.series_to_supervised(
            data=data_inter,
            n_in=self._seq_len,
            n_out=forecast_horizon,
            separate_output_series=True)

        # reshape the data to ensure it is appropriate for keras
        # inputs need to be of the dimensions (samples, sequence, features)
        self.in_data = self.in_data.reshape(
            (self.in_data.shape[0], self.seq_len, 1))
        self.out_data = self.out_data.reshape((-1, forecast_horizon))

    def Fit(self,
            batch_size=128,
            epochs=2000,
            train_frac=0.8,
            valid_frac=1 / 3,
            verbose=False,
            validate_model=True,
            print_test_stat=True):
        """ Fit the specified model to the data


        """
        n_train = fxu.round_down(0.8 * self.in_data.shape[0], base=1)
        n_valid = fxu.round_down(1 * (self.in_data.shape[0] - n_train) / 3,
                                 base=1)
        in_train = self.in_data[:n_train]
        in_test = self.in_data[n_train:-n_valid]
        in_valid = self.in_data[-n_valid:]
        out_train = self.out_data[:n_train]
        out_test = self.out_data[n_train:-n_valid]
        out_valid = self.out_data[-n_valid:]

        if verbose:
            print('Number of training samples: ', n_train)
            print('Number of test samples: ', in_test.shape[0])
            print('Number of validation samples: ', in_valid.shape[0])

            history = self._model.fit(in_train,
                                      out_train,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      verbose=verbose)
        if print_test_stat:
            loss = self._model.evaluate(in_test, out_test, verbose=1)

        if validate_model:
            yhat = self._model.predict(in_valid, verbose=0)

        return history, loss, yhat

    def AddDatum(self, datum):
        if (datum is None) or (not isinstance(datum, float)):
            raise TypeError('datum must be a float type')
