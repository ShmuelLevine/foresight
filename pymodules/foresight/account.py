# account.py

from foresight.trading_generic import *
from math import fabs

class Account():
    
    def __init__(self, initial_balance):
        self.balance = initial_balance
        self.position = 0
        self.txn_price = 0.0
        
    def HasOpenLongPositions(self):
        return self.position > 0
    
    def HasOpenShortPositions(self):
        return self.position < 0
    
    def HasNoOpenPositions(self):
        return self.position == 0
    
    def OpenPosition(self, position_type, trade_size, current_prices):
        if not isinstance(position_type, TradeType):
            raise TypeError('position_type must be a TradeType')
        
        if isinstance(position_type,LongPosition):
            self.position += trade_size
            price = current_prices['bid']
        elif isinstance(position_type, ShortPosition):
            self.position -= trade_size
            price = current_prices['ask']
        
        self.balance -= trade_size * price
        self.txn_price = price

    def ClosePosition(self, position_type, trade_size, current_prices):

        if isinstance(position_type,LongPosition):
            self.position -= trade_size
            price = current_prices['ask']
        elif isinstance(position_type, ShortPosition):
            self.position += trade_size
            price = current_prices['bid']
        
        self.balance += trade_size * price
        
    def CloseOpenPositions(self, current_prices):
        if self.HasOpenLongPositions():
            price = current_prices['ask']
        elif self.HasOpenShortPositions():
            price = current_prices['bid']
        
        self.balance += fabs(self.position) * price
        self.position = 0