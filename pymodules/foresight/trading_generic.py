# trading_generic.py
# general types and functions associated with trading

from abc import ABC, abstractmethod

# empty types to use for tags
class TradeType(ABC): pass
class LongPosition(TradeType): pass
class ShortPosition(TradeType): pass