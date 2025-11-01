from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
from scipy.signal import argrelextrema
from freqtrade.persistence import Trade
from datetime import datetime

class FawdaBot(IStrategy):
    INTERFACE_VERSION = 3

    # Strategy parameters
    risk_percent = 0.01
    rr = 2
    atr_multiplier = 0.6
    lot_size = 100.0  # Simulated lot size for 100 oz
    point_value = 1.0  # $ per USDT move per unit (XAUT)
    lz_width = 0.006  # ±0.6% for LZ around swings (adjusted for gold volatility)
    startup_candle_count = 400
    process_only_new_candles = True

    # No fixed stoploss/ROI - handled dynamically
    stoploss = -0.99
    minimal_roi = {"0": 0.0}

    # Trailing stop disabled in config, but can manage in custom_stoploss
    trailing_stop = False

    # Informative timeframes for HTF bias and zones
    informative_tf = ['4h', '1d']

    def informative_pairs(self):
        pairs = [(pair, tf) for pair in self.dp.current_whitelist() for tf in self.informative_tf]
        return pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Main timeframe indicators
        dataframe['atr14'] = ta.ATR(dataframe, timeperiod=14)

        # For entry triggers: RSI as optional, but mainly price action
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Process informative pairs
        for tf in self.informative_tf:
            inf_df = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=tf)
            inf_df['ema20'] = ta.EMA(inf_df, timeperiod=20)
            
            # Detect swing highs/lows on HTF
            inf_df['swing_high'] = (argrelextrema(inf_df['high'].values, np.greater, order=20)[0])
            inf_df['swing_low'] = (argrelextrema(inf_df['low'].values, np.less, order=20)[0])
            
            # FVG on HTF (adapted from smart-money-concepts example)
            fvg = np.full(len(inf_df), np.nan)
            for i in range(3, len(inf_df)-3):
                if inf_df['high'][i-1] < inf_df['low'][i+1] and inf_df['close'][i] > inf_df['high'][i+1]:
                    fvg[i] = inf_df['low'][i+1] - inf_df['high'][i-1]
                elif inf_df['low'][i-1] > inf_df['high'][i+1] and inf_df['close'][i] < inf_df['low'][i+1]:
                    fvg[i] = inf_df['high'][i+1] - inf_df['low'][i-1]
            inf_df['fvg'] = fvg
            
            # OB: Simplified as candle before strong move (e.g., large range candle)
            inf_df['ob_bull'] = (inf_df['close'].shift(1) > inf_df['open'].shift(1)) & (inf_df['close'] - inf_df['open'] > inf_df['atr14'] * 1.5)
            inf_df['ob_bear'] = (inf_df['close'].shift(1) < inf_df['open'].shift(1)) & (inf_df['open'] - inf_df['close'] > inf_df['atr14'] * 1.5)
            
            # Merge back
            dataframe = merge_informative_pair(dataframe, inf_df, self.timeframe, tf, ffill=True)

        return dataframe

    def compute_bias(self, dataframe: DataFrame) -> str:
        # Simple bias: Bullish if close > EMA20 on both 4h and 1d
        if (dataframe['close_4h'] > dataframe['ema20_4h']) and (dataframe['close_1d'] > dataframe['ema20_1d']):
            return 'bullish'
        elif (dataframe['close_4h'] < dataframe['ema20_4h']) and (dataframe['close_1d'] < dataframe['ema20_1d']):
            return 'bearish'
        return 'neutral'

    def is_in_zoi(self, dataframe: DataFrame, current_price: float) -> bool:
        # Simplified ZoI check: Overlap of LZ, FVG, OB on HTF
        # Get recent swings
        recent_high = dataframe['high_4h'].max()  # Simplified; in practice, use last swing
        recent_low = dataframe['low_4h'].min()
        
        # LZ: Around swings ± lz_width
        lz_high = recent_high * (1 + self.lz_width)
        lz_low = recent_low * (1 - self.lz_width)
        
        # FVG: Check if any non-nan FVG near price
        has_fvg = dataframe['fvg_4h'].notna().any()
        
        # OB: Check if OB near price
        has_ob = dataframe['ob_bull_4h'].any() or dataframe['ob_bear_4h'].any()
        
        # Super ZoI if overlap (simplified: if price in LZ and has FVG/OB)
        if lz_low <= current_price <= lz_high and (has_fvg or has_ob):
            return True
        return False

    def trigger_confirmed(self, dataframe: DataFrame) -> bool:
        # Simplified triggers: Bullish engulfing or pin bar
        # Engulfing: Current close > prev high, open < prev low, close > open
        engulfing = (dataframe['close'] > dataframe['high'].shift(1)) & \
                    (dataframe['open'] < dataframe['low'].shift(1)) & \
                    (dataframe['close'] > dataframe['open'])
        
        # Pin bar: Small body, long lower wick for bull
        pin_bar = (dataframe['close'] > dataframe['open']) & \
                  ((dataframe['open'] - dataframe['low']) > 2 * (dataframe['close'] - dataframe['open']))
        
        return engulfing.iloc[-1] or pin_bar.iloc[-1]

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: float, max_stake: float,
                           leverage: float, entry_tag: str, side: str,
                           **kwargs) -> float:
        # Dynamic stake based on risk
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) == 0:
            return proposed_stake
        
        atr = dataframe['atr14'].iloc[-1]
        balance = self.wallets.get_total(self.stake_currency)
        stop_distance = (balance * self.risk_percent) / (self.lot_size * self.point_value)
        stop_distance += self.atr_multiplier * atr
        
        # Quantity Q = risk / stop_distance (since loss = Q * stop_distance)
        quantity = (balance * self.risk_percent) / stop_distance
        stake = quantity * current_rate  # USDT stake = Q * price
        
        return min(stake, max_stake)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        bias = self.compute_bias(dataframe)
        current_price = dataframe['close'].iloc[-1]
        
        if bias == 'bullish' and self.is_in_zoi(dataframe, current_price) and self.trigger_confirmed(dataframe):
            dataframe.loc[:, 'enter_long'] = 1
            # Store SL distance in custom_info for later use
            atr = dataframe['atr14'].iloc[-1]
            balance = self.wallets.get_total(self.stake_currency)
            stop_distance = (balance * self.risk_percent) / (self.lot_size * self.point_value) + self.atr_multiplier * atr
            # Use custom field or trade custom_data in later versions
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit on opposite trigger or RSI overbought (simplified)
        if dataframe['rsi'] > 70:
            dataframe.loc[:, 'exit_long'] = 1
        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        # Move to breakeven at +1R
        if current_profit > (1 / self.rr):
            return 0.001  # Small positive to set at entry (breakeven)
        
        # Otherwise, use initial SL (calculated per trade, but simplified here)
        return -0.10  # Fallback

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float, max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> float:
        # Optional scale out at +1R: Close 50% position
        if current_profit > (1 / self.rr):
            return - (trade.stake_amount * 0.5)  # Reduce by 50%
        return None