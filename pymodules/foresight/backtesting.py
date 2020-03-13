from pandas import Timedelta
import foresight.model
import foresight.util as fxu

class Backtester:
    """Class used for backtesting models against a timeseries"""

    def __init__(self, model, trading_rules, retraining_freq, initial_money = 10_000):
        """Initialize a Backtester instance

        Args:
            model (keras.Model) -- The fitted Keras Model to be backtested

            trading_rules -- A subclass of Trading_Rules which contains rules used to
               determine when to make trades

            retraining_frequency (pandas.Timedelta) -- A pandas.Timedelta object denoting how
                frequently to refit the model.  NOTE: at this time, the retraining will
                not be 100% exact.

            initial_money -- An integer representing the amount of money initially used
                for trading. This amount is used by the trading rules in establishing the
                value of each trade

        """
                fxu.ValidateType(model, arg_name='model', reqd_type=foresight.model.Model, err_msg='must be an instance of foresight.model.Model')
        self.model = model
        self.trading_rules = trading_rules
        self.initial_money = initial_money

        # Calculate the retraining frequency (in number of datapoints)
        # First ensure that the values provided are of the correct type
                fxu.ValidateType(retraining_freq, arg_name='retraining_freq', reqd_type=Timedelta, err_msg='must be of type pandas.Timedelta')

        self.retrain_freq =  retraining_freq / model.data_freq
        print("Restarting every", self.retrain_freq, "datapoints")
        print("New backtester created!!")

    def Backtest(self, data, verbose=False):
        cur_idx = 0
        print("Retraining every", self.retrain_freq, "points")
        pass
