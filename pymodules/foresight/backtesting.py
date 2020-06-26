import importlib

from pandas import Timedelta
import foresight.model
import foresight.util as fxu
import foresight.data_functions as fx_df
from foresight.account import Account
from foresight.trading_generic import *
from foresight.tradingRules import OpenLongPosition, OpenShortPosition, CloseOpenPositions
import math
import numpy as np
importlib.reload(fx_df)
importlib.reload(foresight.model)


class Backtester:
    """
    Class used for backtesting models against a timeseries

    :param model: The foresight.model.Model, containing a fitted Keras Model to be backtested
    :type model: class:`foresight.model.Model`
    
    :param trading_rules: A set of rules used to determine when to make trades
    :type trading_rules: class:`Trading_Rules`

    :param retraining_frequency: A pandas.Timedelta object denoting how
                frequently to refit the model.  NOTE: at this time, the retraining will
                not be 100% exact.
    :type retraining_frequency: class:`pandas.Timedelta`

    :param initial_money: An integer representing the amount of money initially used
                for trading. This amount is used by the trading rules in establishing the
                value of each trade
    :type initial_money: int
  
    """
    def __init__(
        self,
        model,
        retraining_freq,
        trading_rules={
            'trade_size': 250_000,
            'stop_loss': 0.00025,
            'take_profit': 0.00025,
            'min_change': 0.00005,
            'leverage': 1
        },
        initial_money=10_000
    ):

        #        fxu.ValidateType(
        #            model,
        #            arg_name='model',
        #            reqd_type=foresight.model.Model,
        #            err_msg='must be an instance of foresight.model.Model')

        # TODO: finish validation

        self._model = model
        self._trading_rules = trading_rules
        self._account = Account(initial_balance=initial_money)
        self._initial_money = initial_money

        # Calculate the retraining frequency (in number of datapoints)
        # First ensure that the values provided are of the correct type
        fxu.ValidateType(
            retraining_freq,
            arg_name='retraining_freq',
            reqd_type=Timedelta,
            err_msg='must be of type pandas.Timedelta'
        )

        self.retrain_freq = int(retraining_freq / model._data_freq)
        print("Restarting every", self.retrain_freq, "datapoints")
        print("New backtester created!!")

    def prepare_data(self, data):
        """
        
        Prepare raw tick data for use in backtesting models
        
        The data preparation operation is fairly straightforward:
        1. Resample the data again. This time, use the 'pad' ['backfill'] operation, since this is simulating
           data arriving in realtime
        2. Remove non-trading days from the dataset. This includes weekends and other days when the market
           are closed.  
           NOTE: this function needs to be updated to better handle forex holiday days

        """

        return fx_df.clean_data(
            data,
            remove_duplicates=True,
            sample_frequency=self._model._data_freq,
            sample_type='pad',
            remove_weekends=True
        )

#        freq=data._data_freq
#        _data = data[~data_raw.index.duplicated()]

#        # resample the data, using the nearest value
#        _data = _data.resample(sample_time).pad()

#        # remove the weekends when there is no trading
#        _data = df2d[df2d.index.dayofweek < 5]

    def HasOpenLongPositions(self):
        return self._account['position'] > 0

    def HasOpenShortPositions(self):
        return self._account['position'] < 0

    def HasNoOpenPositions(self):
        return self._account['position'] == 0

    def OpenPosition(self, position_type=None, price=None):
        if position_type == 'long':
            self._account['position'] += self._trading_rules['trade_size']
            self._account['balance'] -= self._trading_rules['trade_size'] * price[
                'bid']  # open long - use bid price
            self._account['txn_price'] = price['bid']
        elif position_type == 'short':
            self._account['position'] -= self._trading_rules['trade_size']
            self._account['balance'] -= self._trading_rules['trade_size'] * price[
                'ask']  # open short - use ask price
            self._account['txn_price'] = price['ask']
        else:
            raise ValueError('position_type must be either \'long\' or \'short\'')

    def ClosePosition(self, position_type=None, price=None):
        if self.HasOpenLongPositions():
            self._account['position'] -= self._trading_rules['trade_size']
            self._account['balance'] += self._trading_rules['trade_size'] * price[
                'ask']  # close long - use ask price

        elif self.HasOpenShortPositions:
            self._account['position'] += self._trading_rules['trade_size']
            self._account['balance'] += self._trading_rules['trade_size'] * price[
                'bid']  # close short - use bid price

        else:
            raise ValueError('position_type must be either \'long\' or \'short\'')

    def HasPriceHitTakeProfit(self, price):
        #        print('Function HasPriceHitTakeProfit\n type of price: ', type(price))
        if self.HasOpenLongPositions():
            #            print( price >= (self._account['txn_price'] + self._trading_rules['take_profit']))
            return price['ask'] >= (self._account['txn_price'] + self._trading_rules['take_profit'])
        else:
            #            print( price <= (self._account['txn_price'] - self._trading_rules['take_profit']))
            return price['bid'] <= (self._account['txn_price'] - self._trading_rules['take_profit'])
        pass

    def HasPriceHitStopLoss(self, price):
        #        print('Function HasPriceHitTakeProfit')
        if self.HasOpenLongPositions():
            return price['ask'] < (self._account['txn_price'] - self._trading_rules['stop_loss'])
        else:
            return price['bid'] > (self._account['txn_price'] - self._trading_rules['stop_loss'])
        pass

    def Reset_Model(self):
        # These instance variables will only exist / be defined after the backtesting has happened once
        # therefore, the variables exist, there exist saved data to restore.  Otherwise, do nothing
        try:
            if fxu.VarExists(self.weights_backup_path):
                self._model._model.load_weights(self.weights_backup_path)
                self._model._in_data = np.load(self.data_ins_backup_path)
                self._model._out_data = np.load(self.data_outs_backup_path)
        except (AttributeError):
            pass

    def Backtest(
        self,
        data,
        trading_rules,
        verbose=False,
        initial_retraining=0.15,
        retrain_epochs=25,
        retrain_verbose=False,
        retrain_batch_size=None
    ):
        """
        Backtest the model using the provided data
        
        The backtesting operation is intended to be self-contained, not at all
        dependant on the original dataset. In other words, it does not assume that
        the first datum in the new timeseries immediately follows from the last training
        datum.  Consequently, the function will generate a new data sequence.
        
        :param data: The new timeseries to be used for backtesting.  This should be tick-data,
            not rescaled, with the datetime as the index, with 2 series named ['bid', 'ask']
        :type data: class:`pandas.DataFrame`
        """

        cur_idx = 0
        print("Retraining every", self.retrain_freq, "points")

        idx_offset = self._model._seq_len  # offset of indices from output to the original dataset - the (current) bid price when processing model output idx (which forecasts the next bid price)

        if retrain_batch_size is None:
            retrain_batch_size = self._model._batch_size

        # The backtesting operation is fairly straightforward:
        # 0. Backup the original model dataset so that it can be reset after backtesting. This ensures
        #    that the backtesting can be run multiple times on the same model
        # 1. Prepare/clean the backtesting data
        # 1a. Run foresight.data_functions.clean_data on the dataset to ensure that all data are well-behaved (i.e. no duplicates, re-sampled)
        # 2b. Extract the bid prices and transform the data appropriately for use in the model
        # 2. Prepare the data for use as timeseries supervised learning data.  Until the model is retrained, the predictions can be performed as a batch operation
        # 3. If an initial_retraining value is provided, collect the first initial_retraining data and retrain the model before simulating realtime trading
        # 4. Restore the original weights and data

        ## Step 0
        import tempfile
        import os
        import datetime
        import copy
        
        now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.backup_dir_obj = tempfile.TemporaryDirectory()
        self.backup_dir = self.backup_dir_obj.name

        # setup filenames for data
        self.weights_backup_path = os.path.join(self.backup_dir, 'weights')
        self.data_ins_backup_path = os.path.join(self.backup_dir, 'data_in' + '.npy')
        self.data_outs_backup_path = os.path.join(self.backup_dir, 'data_outs' + '.npy')

        # save model weights and original dataset
        self._model._model.save_weights(self.weights_backup_path)
        np.save(self.data_ins_backup_path, self._model._in_data)
        np.save(self.data_outs_backup_path, self._model._out_data)

        # Step 1a - clean the backtesting data. Save this for calculating gains/losses based on trading strategy
        self._data = self.prepare_data(data)

        # Step 1b - obtain a numpy.ndarray of transformed data
        txn = copy.copy(self._model._data_transform)
        bid_txd, scaler = txn(self._data['bid'].to_numpy(), remove_outliers=False)

        _ins, _outs = fx_df.series_to_supervised(
            bid_txd,
            n_in=self._model._seq_len,
            n_out=self._model._forecast_horizon,
            dropnan=True,
            separate_output_series=True
        )

        #print(_ins.shape)
        _ins = _ins.reshape((-1, self._model._seq_len, 1))
        #print(_ins.shape)

        bid_actual = self._data['bid'].to_numpy()
        bid_forecasts = bid_actual.copy()  # initialize to match bid_actual. Update in loop below

        if initial_retraining:
            cur_idx = int(initial_retraining * self.retrain_freq)
            self._model.AddTrainingData(_ins[:cur_idx], _outs[:cur_idx])
            self._model.Refit(epochs=retrain_epochs, verbose=retrain_verbose, batch_size = retrain_batch_size)

        total_data = _ins.shape[0]

        #print(cur_idx , total_data, self.retrain_freq)
        # Create a range of indices for retraining the model.  It is assumed that the last index in the range will be smaller than the total number of data
        retrain_idx = range(cur_idx, total_data, self.retrain_freq)
        cur_data = self._data.iloc[-1]

        # main backtesting loop
        # Use data from indices
        for start_idx in retrain_idx:
            last_idx = start_idx + self.retrain_freq - 1 if start_idx + self.retrain_freq - 1 <= total_data else total_data
            print('Main backtesting loop - from indices: ', start_idx, ' to ', last_idx)
            #print(_ins[start_idx:last_idx].shape)
            y_hat = self._model.Forecast(_ins[start_idx:last_idx]).numpy().flatten()
            print('y_hat.shape', y_hat.shape)
            #   print(type(self._data.iloc[start_idx + idx_offset : last_idx + idx_offset]['bid']))
            y_hat_unscaled = txn.Invert_Scaling(y_hat)
            #print('np.exp(y_hat_unscaled).shape', np.exp(y_hat_unscaled).flatten().shape, '\n\n', np.exp(y_hat_unscaled))
            #print('\n\n', self._data.iloc[start_idx + idx_offset : last_idx + idx_offset]['bid'].to_numpy())
            bid_hat = np.multiply(
                np.exp(y_hat_unscaled).flatten(),
                self._data.iloc[start_idx + idx_offset :last_idx + idx_offset]['bid'].to_numpy()
            )
            bid_forecasts[start_idx + idx_offset + 1:last_idx + idx_offset + 1 ] = bid_hat
            #print(bid_hat.shape)
            # TODO: do something with predictions
            forecast_idx = 0
            for idx in range(start_idx, last_idx):
                cur_data = self._data.iloc[idx + idx_offset]
                fore_bid = bid_hat[forecast_idx]

                print('Index:', idx, 'current_bid:', cur_data['bid'].item(), 'next_bid:', self._data.iloc[idx+idx_offset + 1]['bid'], 'forecast bid:', fore_bid)
                trade_decision, trade_size = trading_rules.Trade_Decision(self._account, cur_data, fore_bid)

                if trade_decision in [OpenLongPosition, OpenShortPosition]:
                    self.account.OpenPosition(
                        position_type=LongPosition if trade_decision is OpenLongPosition else ShortPosition,
                        trade_size=trade_size,
                        current_prices=cur_data
                    )
                elif trade_decision is CloseOpenPositions:
                    self.account.CloseOpenPositions(cur_data)

                forecast_idx += 1

            # Retrain the model if there are more samples
            if start_idx + self.retrain_freq < total_data:
                print('Retraining model')
                self._model.AddTrainingData(_ins[start_idx:last_idx], _outs[start_idx:last_idx])
                self._model.Refit(
                    batch_size=retrain_batch_size, epochs=retrain_epochs, verbose=retrain_verbose
                )

        # close any open positions based on the last available price


#        print('\n\n\n', cur_data)
        if not self._account.HasNoOpenPositions():
            self._account.CloseOpenPositions(cur_data)

        print(
            'Final balance: ', self._account.balance, '   Return: ',
            (self._account.balance / self._initial_money - 1) * 100, '%'
        )

        # Finally, restore the original weights and data so that the model can be reused
        self.Reset_Model()

        return bid_actual, bid_forecasts
