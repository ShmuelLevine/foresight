# model.py

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from tensorflow.keras import Model as tf_model
import numpy as np
import numpy
import pandas as pd
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
    def __init__(
        self,
        model,
        data,
        data_freq,
        seq_len,
        scaler=None,
        forecast_horizon=1,
        data_transform=None,
        #                 stationary_transform=None,
        max_training_data_factor=3
    ):

        fxu.ValidateType(
            model,
            arg_name='model',
            reqd_type=tf_model,
            err_msg='must be an instance of tensorflow.keras.Model'
        )

        fxu.ValidateType(
            data, arg_name='data', reqd_type=np.ndarray, err_msg='must be an instance of numpy.array'
        )

        fxu.ValidateType(
            data_freq, arg_name='data_freq', reqd_type=TD, err_msg='must be an instance of pandas.Timedelta'
        )

        fxu.ValidateType(seq_len, arg_name='seq_len', reqd_type=int, err_msg='must be an integer')

        fxu.ValidateType(
            data_transform,
            arg_name='data_transform',
            val_func=lambda x: callable(x),
            err_msg='must be a callable type',
            allow_none=True
        )

        #        fxu.ValidateType(stationary_transform,
        #                         reqd_type=str,
        #                         arg_name='stationary_transform',
        #                         err_msg='must [None, \'Diff\', \'LogDiff\']',
        #                         allow_none=True)

        # We need to store the entire transformed data to use with the model, but only enough
        # raw data to generate the next sequence when a new datum is added
        self._model = model
        #       self._rawdata = data[-seq_len:]  # store the last seq_len points
        self._data_freq = data_freq
        self._data_transform = data_transform
        self._seq_len = seq_len
        self._forecast_horizon = forecast_horizon
        self._max_training_data_factor = max_training_data_factor

        if not forecast_horizon == 1:
            raise ValueError('Code cannot handle forecast horizon other than 1 right now')

        # store the transformed data in an intermediate variable, as the data needs to be converted
        # into supervised learning data
        data_inter, self._scaler = numpy.array(self._data_transform(data))

        # convert the input data into 2 matrices representing inputs and expected outputs, to use for
        # model training
        self._in_data, self._out_data = fx_df.series_to_supervised(
            data=data_inter, n_in=self._seq_len, n_out=forecast_horizon, separate_output_series=True
        )

        # reshape the data to ensure it is appropriate for keras
        # inputs need to be of the dimensions (samples, sequence, features)
        self._in_data = self._in_data.reshape((self._in_data.shape[0], self._seq_len, 1))
        self._out_data = self._out_data.reshape((-1, forecast_horizon))

        self._max_training_samples = int(max_training_data_factor * self._in_data.shape[0])
        self.__oldest_datum = 0

    def Transform_Type(self):
        return self._data_transform.transform

    def _ReplaceTrainingData(self, input_seq, output_seq):
        """
        Function to add a new datum to the training/inference data, by replacing the oldest sequence

        This function encapsulates the logic and index required to add a new datum and remove the oldest.
        By doing so, it treats the member self._in_data and self._out_data as circular buffers, since the
        training process is not sensitive to the order of the sequences.

        The assumed use of this function is during backtesting.  During this process, a new datum is read from
        the actual data; a new sequence of data is generated and the model is run with this single sequence to 
        predict the next datum.  As such, it should also be necessary to append data to the outputs; however, 
        by nature, this index is lagged by 1 compared to the input data.

        To handle this, the initial index is set for None, as a sentinel value.  

        :param input_seq: A numpy array containing the new input sequences for the model
        :type datum: class:`numpy.ndarray`

        :param input_seq: A numpy array containing the new output values (sequences) for the model
        :type datum: class:`numpy.ndarray`

        """

        if (input_seq is None) or (not isinstance(input_seq, np.ndarray)):
            raise TypeError('input_seq must be a numpy array type')
        if (output_seq is None) or (not isinstance(output_seq, np.ndarray)):
            raise TypeError('output_seq must be a numpy array type')

        num_samples = self._in_data.shape[0]

        # reshape the new data into 3D array of samples, sequence, features
        input_seq = input_seq.reshape((input_seq.shape[0], input_seq.shape[1], -1))
        output_seq = output_seq.reshape((-1, self._forecast_horizon))

        new_samples = input_seq.shape[0]

        if new_samples > num_samples:
            self._in_data = input_seq[-num_samples:]
            self._out_data = output_seq[-num_samples:]
        else:
            indices = np.arange(self.__oldest_datum, new_samples + self.__oldest_datum) % num_samples
            for i_src, i_tgt in enumerate(indices):
                self._in_data[i_tgt] = input_seq[i_src]
                self._out_data[i_tgt] = output_seq[i_src]
            self.__oldest_datum = (self.__oldest_datum + new_samples) % num_samples

    def _AppendTrainingData(self, input_seq, output_seq):
        """
        Function to add a new datum to the training/inference data, by replacing the oldest sequence

        This function encapsulates the logic and index required to add a new datum and remove the oldest.
        By doing so, it treats the member self._in_data and self._out_data as circular buffers, since the
        training process is not sensitive to the order of the sequences.

        The assumed use of this function is during backtesting.  During this process, a new datum is read from
        the actual data; a new sequence of data is generated and the model is run with this single sequence to 
        predict the next datum.  As such, it should also be necessary to append data to the outputs; however, 
        by nature, this index is lagged by 1 compared to the input data.

        To handle this, the initial index is set for None, as a sentinel value.  

        :param input_seq: A numpy array containing the new input sequences for the model
        :type datum: class:`numpy.ndarray`

        :param input_seq: A numpy array containing the new output values (sequences) for the model
        :type datum: class:`numpy.ndarray`

        """

        if (input_seq is None) or (not isinstance(input_seq, np.ndarray)):
            raise TypeError('input_seq must be a numpy array type')
        if (output_seq is None) or (not isinstance(output_seq, np.ndarray)):
            raise TypeError('output_seq must be a numpy array type')

        num_samples = self._in_data.shape[0]

        # reshape the new data into 3D array of samples, sequence, features
        input_seq = input_seq.reshape((input_seq.shape[0], input_seq.shape[1], -1))
        output_seq = output_seq.reshape((-1, self._forecast_horizon))

        new_samples = input_seq.shape[0]

        self._in_data = np.concatenate((self._in_data, input_seq))
        self._out_data = np.concatenate((self._out_data, output_seq))

    def AddTrainingData(self, input_seq, output_seq):
        """
        Function to add new training data to the training data, by conditionally appending or replacing existing data, based on self._max_training_data_factor

        The assumed use of this function is during backtesting.  

        :param input_seq: A numpy array containing the new input sequences for the model
        :type datum: class:`numpy.ndarray`

        :param input_seq: A numpy array containing the new output values (sequences) for the model
        :type datum: class:`numpy.ndarray`

        """

        existing_samples = self._in_data.shape[0]
        if existing_samples < self._max_training_samples:
            self._AppendTrainingData(input_seq, output_seq)
        else:
            self._ReplaceTrainingData(input_seq, output_seq)

    def Fit(
        self,
        batch_size=128,
        epochs=2000,
        train_frac=0.8,
        valid_frac=1 / 3,
        verbose=False,
        validate_model=True,
        print_test_stat=True,
        callbacks=None
    ):
        """ Fit the specified model to the data

            :param batch_size: Batch size used for keras.model.fit() function
            :type batch_size: int

            :param epochs: Number of epochs to use for training the data
            :type epochs: int

            :param train_frac: Fraction of the dataset to use for training the model
            :type train_frac: float

            :param valid_frac: Fraction of the dataset remaining after removing the training data to use for validating the model. 
            :type valid_frac: float

            :param verbose: Boolean value denoting whether to provide fitting details (passed through to keras.model.fit() function)
            :type verbose: bool

            :param validate_model: Boolean value denoting whether to validate the model after fitting
            :type validate_model: bool

            :param print_test_stat
            :type print_test_stat

        """
        self._batch_size = batch_size

        n_train = fxu.round_down(train_frac * self._in_data.shape[0], base=1)
        n_valid = fxu.round_down(valid_frac * (self._in_data.shape[0] - n_train), base=1)
        in_train = self._in_data[:n_train]
        in_test = self._in_data[n_train:-n_valid]
        in_valid = self._in_data[-n_valid:]
        out_train = self._out_data[:n_train]
        out_test = self._out_data[n_train:-n_valid]
        out_valid = self._out_data[-n_valid:]

        # Ensure that the model has been compiled.  If not, compile it now
        if not self._model._is_compiled:
            self._model.compile(loss='mae', optimizer='nadam')

        # Define return variables
        history = None
        loss = None
        yhat = None

        if verbose:
            print('Number of training samples: ', n_train)
            print('Number of test samples: ', in_test.shape[0])
            print('Number of validation samples: ', in_valid.shape[0])

        history = self._model.fit(
            in_train, out_train, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks
        )

        if print_test_stat:
            loss = self._model.evaluate(in_test, out_test, verbose=1)

        if validate_model:
            yhat = self._model.predict(in_valid, verbose=0)

        return history, loss, yhat

    def Forecast(self, inputs):
        return self._model.predict(inputs.reshape((inputs.shape[0], -1, 1)))


#        return self._model(inputs.reshape( (inputs.shape[0], -1, 1)))

    def Refit(self, batch_size=None, epochs=50, verbose=True):
        _batch_size = self._batch_size if batch_size is None else batch_size

        return self.Fit(
            batch_size=_batch_size,
            epochs=epochs,
            train_frac=1,
            valid_frac=0,
            verbose=verbose,
            validate_model=False,
            print_test_stat=False
        )

    def Save_Weights(self, path):
        self._model.save_weights(path)

    def Load_Weights(self, path):
        self._model.load_weights(path)

    def Get_Weights(self):
        return self._model.get_weights()

    def Set_Weights(self, weights):
        self._model.set_weights(weights)

    def Calculate_Loss(self, data):
        return self._model.e


def reset_weights(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):  #if you're using a model as a layer
            reset_weights(layer)  #apply function recursively
            continue

        #where are the initializers?
        if hasattr(layer, 'cell'):
            init_container = layer.cell
        else:
            init_container = layer

        for key, initializer in init_container.__dict__.items():
            if "initializer" not in key:  #is this item an initializer?
                continue  #if no, skip it

            # find the corresponding variable, like the kernel or the bias
            if key == 'recurrent_initializer':  #special case check
                var = getattr(init_container, 'recurrent_kernel')
            else:
                var = getattr(init_container, key.replace("_initializer", ""))

            var.assign(initializer(var.shape, var.dtype))
            #use the initializer


def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.
    This is a fast approximation of re-initializing the weights of a model.
    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).
    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)
