import pandas as pd
import numpy as np
from typing import List
from scipy.signal import argrelextrema

class Features:
    def __init__(self, df: pd.DataFrame) -> None:
        # Work directly on the passed DataFrame to avoid memory overhead from copying
        self.df = df
        
        # Convert essential columns to NumPy arrays for efficient processing
        self.close_np = self.df['close'].astype(np.float32).values
        self.open_np = self.df['open'].astype(np.float32).values
        self.high_np = self.df['high'].astype(np.float32).values
        self.low_np = self.df['low'].astype(np.float32).values
        self.volume_np = self.df['volume'].astype(np.float32).values

    def _convert_to_float32(self) -> pd.DataFrame:
        """Convert numerical columns to float32 to save memory."""
        float_cols = [col for col in self.df.columns if self.df[col].dtype in ['float64', 'float32']]
        self.df[float_cols] = self.df[float_cols].astype('float32')
        return self.df

    def price_based_features(self) -> pd.DataFrame:
        """Compute price-based features and return the updated DataFrame."""
        self.df['price_change'] = np.diff(self.close_np, prepend=self.close_np[0])
        self.df['next_log_return'] = np.log(self.close_np / np.roll(self.close_np, 1))
        self.df['high_low_spread'] = self.high_np - self.low_np
        self.df['close_open_spread'] = self.close_np - self.open_np
        
        return self._convert_to_float32()

    def volume_based_features(self) -> pd.DataFrame:
        """Compute volume-based features and return the updated DataFrame."""
        price_change = np.diff(self.close_np, prepend=self.close_np[0])
        direction = np.sign(price_change)
        
        self.df['volume_change'] = np.diff(self.volume_np, prepend=self.volume_np[0])
        self.df['OBV'] = np.cumsum(direction * self.volume_np)
        
        return self._convert_to_float32()

    def volatility_features(self, window: int = 15) -> pd.DataFrame:
        """Compute volatility features and return the updated DataFrame."""
        window_view = np.lib.stride_tricks.sliding_window_view(self.close_np, window)
        rolling_std = np.std(window_view, axis=1)
        rolling_mean = np.mean(window_view, axis=1)

        self.df['rolling_std'] = np.concatenate([[np.nan] * (window - 1), rolling_std])
        self.df['upper_band'] = np.concatenate([[np.nan] * (window - 1), rolling_mean + 2 * rolling_std])
        self.df['lower_band'] = np.concatenate([[np.nan] * (window - 1), rolling_mean - 2 * rolling_std])

        # ATR (Average True Range)
        true_range = self.high_np - self.low_np
        atr = np.mean(np.lib.stride_tricks.sliding_window_view(true_range, window), axis=1)
        self.df['ATR'] = np.concatenate([[np.nan] * (window - 1), atr])

        return self._convert_to_float32()

    def momentum_features(self, window_short: int = 13, window_long: int = 95, rsi_window: int = 14, macd_windows: tuple = (12, 26, 9)) -> pd.DataFrame:
        """Compute momentum features (RSI, MACD, SMA, EMA, ROC) and return the updated DataFrame."""
        # RSI Calculation
        self.df['RSI'] = self._calculate_rsi(self.close_np, rsi_window)

        # MACD and Signal Line
        macd, signal_line = self._calculate_macd(self.close_np, macd_windows[0], macd_windows[1], macd_windows[2])
        self.df['MACD'] = macd
        self.df['Signal_Line'] = signal_line

        # SMA and EMA
        sma_l = np.mean(np.lib.stride_tricks.sliding_window_view(self.close_np, window_long), axis=1)
        sma_s = np.mean(np.lib.stride_tricks.sliding_window_view(self.close_np, window_short), axis=1)
        self.df['SMA_L'] = np.concatenate([[np.nan] * (window_long - 1), sma_l])
        self.df['SMA_S'] = np.concatenate([[np.nan] * (window_short - 1), sma_s])
        self.df['EMA'] = self.df['close'].ewm(span=window_short, adjust=False).mean()

        # Rate of Change (ROC)
        self.df['ROC'] = self._calculate_roc(self.close_np, window_short)

        return self._convert_to_float32()

    def trend_features(self, window: int = 14) -> pd.DataFrame:
        """Compute trend features (DMI, trend lines, Ichimoku) and return the updated DataFrame."""
        plus_di, minus_di, adx = self._calculate_dmi(window)
        self.df['+DI'] = plus_di
        self.df['-DI'] = minus_di
        self.df['ADX'] = adx

        support_line, resistance_line = self._calculate_trend_lines(self.close_np, window)
        self.df['support_line'] = support_line.reindex(self.df.index)
        self.df['resistance_line'] = resistance_line.reindex(self.df.index)

        self._calculate_ichimoku_cloud()

        return self._convert_to_float32()

    def lag_rolling_features(self, lags: List[int] = None, windows: List[int] = None) -> pd.DataFrame:
        """Compute lag and rolling statistics features and return the updated DataFrame."""
        lags = lags or [1, 2, 3]
        windows = windows or [5, 10, 20]

        # Lag features
        for lag in lags:
            self.df[f'lag_{lag}'] = np.roll(self.close_np, lag)

        # Rolling statistics
        for window in windows:
            rolling_mean = np.mean(np.lib.stride_tricks.sliding_window_view(self.close_np, window), axis=1)
            rolling_max = np.max(np.lib.stride_tricks.sliding_window_view(self.close_np, window), axis=1)
            rolling_min = np.min(np.lib.stride_tricks.sliding_window_view(self.close_np, window), axis=1)

            self.df[f'rolling_mean_{window}'] = np.concatenate([[np.nan] * (window - 1), rolling_mean])
            self.df[f'rolling_max_{window}'] = np.concatenate([[np.nan] * (window - 1), rolling_max])
            self.df[f'rolling_min_{window}'] = np.concatenate([[np.nan] * (window - 1), rolling_min])

        return self._convert_to_float32()

    def statistical_features(self, window: int = 20) -> pd.DataFrame:
        """Compute statistical features (Skew, Kurtosis, Z-score) and return the updated DataFrame."""
        rolling_skew = pd.Series(self.close_np).rolling(window=window).skew().values
        rolling_kurtosis = pd.Series(self.close_np).rolling(window=window).kurt().values
        self.df[f'Skew_{window}'] = rolling_skew
        self.df[f'kurtosis_{window}'] = rolling_kurtosis

        rolling_mean = np.mean(np.lib.stride_tricks.sliding_window_view(self.close_np, window), axis=1)
        rolling_std = np.std(np.lib.stride_tricks.sliding_window_view(self.close_np, window), axis=1)
        self.df[f'zscore_{window}'] = (self.close_np - np.concatenate([[np.nan] * (window - 1), rolling_mean])) / np.concatenate([[np.nan] * (window - 1), rolling_std])

        return self._convert_to_float32()

    def time_based_features(self) -> pd.DataFrame:
        """Compute time-based features and return the updated DataFrame."""
        self.df['date'] = pd.to_datetime(self.df.index, unit='s')
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        self.df['hour_of_day'] = self.df['date'].dt.hour
        self.df.drop(columns=['date'], inplace=True)

        return self._convert_to_float32()

    def _calculate_rsi(self, close: np.ndarray, window: int) -> np.ndarray:
        """Relative Strength Index (RSI) Calculation."""
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = np.convolve(gain, np.ones(window)/window, mode='valid')
        avg_loss = np.convolve(loss, np.ones(window)/window, mode='valid')

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return np.concatenate([[np.nan] * (window - 1), rsi])

    def _calculate_macd(self, close: np.ndarray, short_window=12, long_window=26, signal_window=9):
        """MACD Calculation."""
        ema_12 = pd.Series(close).ewm(span=short_window, adjust=False).mean().values
        ema_26 = pd.Series(close).ewm(span=long_window, adjust=False).mean().values
        macd = ema_12 - ema_26
        signal_line = pd.Series(macd).ewm(span=signal_window, adjust=False).mean().values
        return macd, signal_line

    def _calculate_roc(self, close: np.ndarray, window: int) -> np.ndarray:
        """Rate of Change (ROC)."""
        return ((close - np.roll(close, window)) / np.roll(close, window)) * 100

    def _calculate_dmi(self, window=14):
        """Directional Movement Index (DMI) Calculation."""
        delta_high = np.diff(self.high_np, prepend=self.high_np[0])
        delta_low = np.diff(self.low_np, prepend=self.low_np[0])

        plus_dm = np.where((delta_high > delta_low) & (delta_high > 0), delta_high, 0)
        minus_dm = np.where((delta_low > delta_high) & (delta_low > 0), delta_low, 0)

        true_range = self.high_np - self.low_np
        atr = np.mean(np.lib.stride_tricks.sliding_window_view(true_range, window), axis=1)
        atr = np.concatenate([[np.nan] * (window - 1), atr])

        plus_di = 100 * (np.convolve(plus_dm, np.ones(window) / window, mode='valid') / atr[window - 1:])
        minus_di = 100 * (np.convolve(minus_dm, np.ones(window) / window, mode='valid') / atr[window - 1:])

        dx = 100 * np.abs((plus_di - minus_di) / (plus_di + minus_di + 1e-10))
        adx = np.convolve(dx, np.ones(window) / window, mode='valid')

        plus_di = np.concatenate([[np.nan] * (window - 1), plus_di])
        minus_di = np.concatenate([[np.nan] * (window - 1), minus_di])
        adx = np.concatenate([[np.nan] * (window * 2 - 2), adx])

        return plus_di, minus_di, adx

    def _calculate_trend_lines(self, close: np.ndarray, window: int = 20) -> pd.Series:
        """Support and Resistance Lines Calculation."""
        local_min = argrelextrema(close, np.less_equal, order=window)[0]
        support_line = pd.Series(close[local_min], index=local_min)
        
        local_max = argrelextrema(close, np.greater_equal, order=window)[0]
        resistance_line = pd.Series(close[local_max], index=local_max)

        return support_line, resistance_line

    def _calculate_ichimoku_cloud(self):
        """Ichimoku Cloud Calculation."""
        tenkan_window = 20
        kijun_window = 60
        senkou_span_b_window = 120
        cloud_displacement = 30
        chikou_shift = -30

        tenkan_sen = (pd.Series(self.high_np).rolling(window=tenkan_window).max() + pd.Series(self.low_np).rolling(window=tenkan_window).min()) / 2
        self.df['tenkan_sen'] = tenkan_sen

        kijun_sen = (pd.Series(self.high_np).rolling(window=kijun_window).max() + pd.Series(self.low_np).rolling(window=kijun_window).min()) / 2
        self.df['kijun_sen'] = kijun_sen

        self.df['senkou_span_a'] = ((tenkan_sen + kijun_sen) / 2).shift(cloud_displacement)

        senkou_span_b = (pd.Series(self.high_np).rolling(window=senkou_span_b_window).max() + pd.Series(self.low_np).rolling(window=senkou_span_b_window).min()) / 2
        self.df['senkou_span_b'] = senkou_span_b.shift(cloud_displacement)

        self.df['chikou_span'] = self.df['close'].shift(chikou_shift)

        return self.df
    
    def fill(self, ffill: bool, bfill: bool) -> pd.DataFrame:
        """Fill NaNs with forward and/or backward fill for numeric columns, keeping float16 where possible."""
        # Select numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        # Identify float16 columns separately to handle them differently
        float16_cols = self.df.select_dtypes(include=['float16']).columns
        other_numeric_cols = numeric_cols.difference(float16_cols)

        # Apply forward fill (ffill) if specified
        if ffill:
            if not float16_cols.empty:
                # Temporarily convert float16 columns to float32 for fill, then convert back
                self.df[float16_cols] = self.df[float16_cols].astype('float32').ffill().astype('float16')
            if not other_numeric_cols.empty:
                self.df[other_numeric_cols] = self.df[other_numeric_cols].ffill()

        # Apply backward fill (bfill) if specified
        if bfill:
            if not float16_cols.empty:
                # Temporarily convert float16 columns to float32 for fill, then convert back
                self.df[float16_cols] = self.df[float16_cols].astype('float32').bfill().astype('float16')
            if not other_numeric_cols.empty:
                self.df[other_numeric_cols] = self.df[other_numeric_cols].bfill()

        return self.df
