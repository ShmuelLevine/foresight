from pandas import Timedelta

class Backtester:
    """Class used for backtesting models against a timeseries"""
    
    def __init__(self, model, data_freq, data_transform, trading_rules, retraining_freq, initial_money = 10_000):
        """Initialize a Backtester instance
        
        Args:
            model (keras.Model) -- The fitted Keras Model to be backtested
            data_freq (pandas.Timedelta) -- The frequency of the data used to fit the model
            data_transform -- A function object used to transform the data used as a model input
            trading_rules -- A subclass of Trading_Rules which contains rules used to determine when to make trades
            retraining_frequency (pandas.Timedelta) -- A pandas.Timedelta object denoting how frequently to refit the model.
                NOTE: at this time, the retraining will not be 100% exact.
            initial_money -- An integer representing the amount of money initially used for trading. This amount is used
                by the trading rules in establishing the value of each trade
        """
        
        self.model = model
#        self.data_transform = data_transform
        self.trading_rules = trading_rules
        #self.retraining_freq = retraining_freq
        self.initial_money = initial_money

        # Calculate the retraining frequency (in number of datapoints)
        # First ensure that the values provided are of the correct type
        if type(data_freq) is not Timedelta:
            print("Invalid data frequency provided. Must be of type pandas.Timedelta")
            raise TypeError
            
        if type(retraining_freq) is not Timedelta:
            print("Invalid retraining frequency provided. Must be of type pandas.Timedelta")
            raise TypeError
            
        self.retrain_freq =  retraining_freq / data_freq
        print("Restarting every", self.retrain_freq, "datapoints")
        print("New backtester created!!")
        
    def Backtest(self, data, verbose=False):
        cur_idx = 0
        print("Retraining every", self.retrain_freq, "points")
        pass
    
    
    
    
    
    
    
    