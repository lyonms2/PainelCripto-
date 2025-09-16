import ccxt
import pandas as pd
import numpy as np
from strategies import strategy_ema_hull_cross, strategy_ema_cross_hull_trend

def get_ema(prices, period=50):
    return prices.ewm(span=period).mean()

def get_hull(prices, period=20):
    def wma(series, period):
        weights = np.arange(1, period + 1)
        return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    half_length = int(period / 2)
    sqrt_length = int(np.sqrt(period))
    wma_half = wma(prices, half_length)
    wma_full = wma(prices, period)
    hull = wma(2 * wma_half - wma_full, sqrt_length)
    return hull

def build_dataframe(ohlcv):
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['EMA50'] = get_ema(df['close'], 50)
    df['HULL20'] = get_hull(df['close'], 20)
    return df

def fetch_signals(symbols, timeframe):
    exchange = ccxt.kucoin()
    resultados = []
    for symbol in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=100)
            df = build_dataframe(ohlcv)
            hull_cross = strategy_ema_hull_cross(df)
            ema_cross = strategy_ema_cross_hull_trend(df)
            resultados.append({
                'Moeda': symbol,
                'Último Preço': df['close'].iloc[-1],
                'EMA50': df['EMA50'].iloc[-1],
                'HULL20': df['HULL20'].iloc[-1],
                'Hull Cross': hull_cross,
                'EMA50 Cross': ema_cross
            })
        except Exception as e:
            resultados.append({'Moeda': symbol, 'Erro': str(e)})
    return resultados
