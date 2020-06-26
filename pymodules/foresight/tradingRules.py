# tradingRules.py

from abc import ABC, abstractmethod
from math import fabs
from foresight.trading_generic import *

class OpenLongPosition: pass
class OpenShortPosition: pass
class CloseOpenPositions: pass


def PriceHitTakeProfit(self, account, current_price, take_profit_amount):
    if account.HasOpenLongPositions():
        return current_price['ask'] >= account.txn_price + take_profit_amount
    else:
        return current_price['bid'] <= account.txn_price + take_profit_amount

def PriceHitStopLoss(self, account, current_price, stop_loss_amount):
    if account.HasOpenLongPositions():
        return current_price['ask'] < account.txn_price + stop_loss_amount
    else:
        return current_price['bid'] > account.txn_price + stop_loss_amount


class TradingRules(ABC):

    def __init__(self): pass
    
    @abstractmethod
    def Trade_Decision(self, account, current_price, bid_forecast):
        pass
    
class BasicTradingRules(TradingRules):
    
    def __init__(self, trade_size, stop_loss, take_profit, min_change):
        super().__init__()
        self.trade_size = trade_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.min_change = min_change
    
    def Trade_Decision(self, account, current_price, bid_forecast):
        # There are 2 Possibilities -- [yes/no] open positions

        # 1 - No open positions
        if account.HasNoOpenPositions():
            if fabs(bid_forecast - current_price['bid'].item()) >= self.min_change:
                # forecast price increase - open long position
                if bid_forecast > current_price['bid']: 
                    return OpenLongPosition, self.trade_size
                # forecast price decrease - open short position
                else: 
                    return OpenShortPosition, self.trade_size
            else:
                return None, None
        
        # 2 - Existing open positions
        else:
            return CloseOpenPositions, self.trade_size
