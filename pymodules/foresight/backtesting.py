from pandas import Timedelta
import foresight.model
import foresight.util as fxu
import foresight.data_functions as fx_df
import math
import numpy as np

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

    def __init__(self,
                 model,
                 retraining_freq,
                 trading_rules = {'trade_size' : 1_000_000, 'stop_loss' : 0.00025, 'take_profit' : 0.00025, 'min_change' : 0.00005 },
                 initial_money=10_000):

#        fxu.ValidateType(
#            model,
#            arg_name='model',
#            reqd_type=foresight.model.Model,
#            err_msg='must be an instance of foresight.model.Model')
        
        # TODO: finish validation
        
        self._model = model
        self._trading_rules = trading_rules
        self._account = {'balance' : initial_money, 'position' : 0, 'txn_price' : 0.0}

        # Calculate the retraining frequency (in number of datapoints)
        # First ensure that the values provided are of the correct type
        fxu.ValidateType(retraining_freq,
                         arg_name='retraining_freq',
                         reqd_type=Timedelta,
                         err_msg='must be of type pandas.Timedelta')

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
        
        return fx_df.clean_data(data, remove_duplicates = True, sample_frequency = self._model._data_freq, sample_type = 'pad', remove_weekends = True)
        
#        freq=data._data_freq
#        _data = data[~data_raw.index.duplicated()]
        
#        # resample the data, using the nearest value
#        _data = _data.resample(sample_time).pad()

#        # remove the weekends when there is no trading
#        _data = df2d[df2d.index.dayofweek < 5]


    def Backtest(self, data, verbose=False, initial_retraining = 0.15):
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
        
        idx_offset = self._model._seq_len       # offset of indices from output to the original dataset - the (current) bid price when processing model output idx (which forecasts the next bid price)
             
        # The backtesting operation is fairly straightforward:
        # 1. Prepare/clean the backtesting data
        # 1a. Run foresight.data_functions.clean_data on the dataset to ensure that all data are well-behaved (i.e. no duplicates, re-sampled)
        # 2b. Extract the bid prices and transform the data appropriately for use in the model
        # 2. Prepare the data for use as timeseries supervised learning data.  Until the model is retrained, the predictions can be performed as a batch operation
        # 3. If an initial_retraining value is provided, collect the first initial_retraining data and retrain the model before simulating realtime trading
        
        # Step 1a - clean the backtesting data. Save this for calculating gains/losses based on trading strategy
        self._data = self.prepare_data(data)
        
        # Step 1b - obtain a numpy.ndarray of transformed data
        txn = self._model._data_transform
        bid_txd, scaler = txn(self._data['bid'].to_numpy(), remove_outliers = False)
        
        _ins, _outs = fx_df.series_to_supervised(bid_txd, n_in=self._model._seq_len,
                         n_out=self._model._forecast_horizon,
                         dropnan=True,
                         separate_output_series=True)
        
        #print(_ins.shape)
        _ins = _ins.reshape ((-1, self._model._seq_len, 1))
        #print(_ins.shape)
        
        if initial_retraining:
            cur_idx = int(initial_retraining * self.retrain_freq)
            self._model.AddTrainingData(_ins[:cur_idx], _outs[:cur_idx])
        
        total_data = _ins.shape[0]
        
        #print(cur_idx , total_data, self.retrain_freq)
        # Create a range of indices for retraining the model.  It is assumed that the last index in the range will be smaller than the total number of data
        retrain_idx = range(cur_idx , total_data, self.retrain_freq)
        
        # main backtesting loop
        # Use data from indices 
        for start_idx in retrain_idx:
            last_idx = start_idx + self.retrain_freq -1 if start_idx + self.retrain_freq -1 <= total_data else total_data
            y_hat = self._model.Forecast(_ins[start_idx:last_idx]).flatten()
            print('y_hat.shape', y_hat.shape)
         #   print(type(self._data.iloc[start_idx + idx_offset : last_idx + idx_offset]['bid']))
            y_hat_unscaled = txn.Invert(y_hat)
         #   print('np.exp(y_hat_unscaled).shape', np.exp(y_hat_unscaled).flatten().shape, '\n\n', np.exp(y_hat_unscaled))
         #   print('\n\n', self._data.iloc[start_idx + idx_offset : last_idx + idx_offset]['bid'].to_numpy())
            bid_hat = np.multiply( np.exp(y_hat_unscaled).flatten(), self._data.iloc[start_idx + idx_offset : last_idx + idx_offset]['bid'].to_numpy() )
            print(bid_hat.shape)
            # TODO: do something with predictions
            forecast_idx = 0
            for idx in range(start_idx, last_idx):
                cur_data = self._data.iloc[idx + idx_offset]
#                print(cur_data, type(cur_data), '\n\n', cur_data['bid'], '\n', type(cur_data['bid']), '\n', type(self._trading_rules['min_change']), '\n', type(cur_data['bid'].item()))
                #print(cur_data['bid'].item() - 2.3)
                #print(bid_hat.shape, '\n\n', bid_hat[forecast_idx], type(bid_hat[forecast_idx]))
                fore_bid = bid_hat[forecast_idx]

                # There are 3 Possibilities -- no open positions, short, or long
                # 1 - No open positions
                if self._account['position'] == 0:
                    # If the forecasted change is greater than the min_change
                    if math.fabs(fore_bid - cur_data['bid'].item()) >= self._trading_rules['min_change']:
                        if fore_bid > cur_data['bid']: # forecast price increase - open long position
                            self._account['position'] += self._trading_rules['trade_size']
                            self._account['balance'] -= self._trading_rules['trade_size'] * cur_data['bid'] # open long - use bid price
                            self._account['txn_price'] = cur_data['bid']
                        else: # forecast price decrease - open short position
                            self._account['position'] -= self._trading_rules['trade_size']
                            self._account['balance'] -= self._trading_rules['trade_size'] * cur_data['ask'] # open short - use ask price
                            self._account['txn_price'] = cur_data['ask']
                            
                # 2 - Existing long position
                elif self._account['position'] > 0:
                    # if hit take_profit, close
                    if cur_data['ask'] >= self._account['txn_price'] + self._trading_rules['take_profit']: # hit the take_profit point
                        # close the the position
                        self._account['position'] -= self._trading_rules['trade_size']
                        self._account['balance'] += self._trading_rules['trade_size'] * cur_data['ask'] # close long - use ask price
                        continue
                    # if hit stop_loss
                    if cur_data['ask'] < self._account['txn_price'] - self._trading_rules['stop_loss']: # hit the stop loss
                        self._account['position'] -= self._trading_rules['trade_size']
                        self._account['balance'] += self._trading_rules['trade_size'] * cur_data['ask'] # close long - use ask price
                        continue
                
                # 3 - Existing short position
                elif self._account['position'] < 0:
                    # if hit take_profit, close
                    if cur_data['bid'] < self._account['txn_price'] - self._trading_rules['take_profit']: # hit the take_profit point
                        # close the the position
                        self._account['position'] += self._trading_rules['trade_size']
                        self._account['balance'] += self._trading_rules['trade_size'] * cur_data['bid'] # close short - use bid price
                        continue
                    # if hit stop_loss
                    if cur_data['bid'] >= self._account['txn_price'] + self._trading_rules['stop_loss']: # hit the stop loss
                        self._account['position'] += self._trading_rules['trade_size']
                        self._account['balance'] += self._trading_rules['trade_size'] * cur_data['bid'] # close short - use bid price
                        continue
            
            # Retrain the model
            self._model.AddTrainingData(_ins[start_idx:last_idx], _outs[start_idx:last_idx])
            
        print (self._account['balance'])
