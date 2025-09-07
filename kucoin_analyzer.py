import ccxt
import pandas as pd
import numpy as np

def get_ema(prices, period=50):
    return prices.ewm(span=period).mean()

def get_hull(prices, period=16):  # período típico da Hull é menor que EMA
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    def wma(series, period):
        weights = np.arange(1, period + 1)
        return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    half_length = int(period / 2)
    sqrt_length = int(np.sqrt(period))
    wma_half = wma(prices, half_length)
    wma_full = wma(prices, period)
    hull = wma(2 * wma_half - wma_full, sqrt_length)
    return hull

def get_signal(df):
    # Precisa de pelo menos 3 candles para detectar cruzamento na última vela fechada
    if len(df) < 3:
        return "-"
    # Usar a penúltima vela como referência (última fechada)
    close_now = df['close'].iloc[-2]
    close_prev = df['close'].iloc[-3]
    ema_now = df['EMA50'].iloc[-2]
    ema_prev = df['EMA50'].iloc[-3]
    hull_now = df['HULL16'].iloc[-2]
    hull_prev = df['HULL16'].iloc[-3]

    sinais = []

    # Sinal EMA50
    if close_prev <= ema_prev and close_now > ema_now:
        sinais.append("Compra EMA50")
    elif close_prev >= ema_prev and close_now < ema_now:
        sinais.append("Venda EMA50")
    elif close_now > ema_now:
        sinais.append("Acima EMA50")
    elif close_now < ema_now:
        sinais.append("Abaixo EMA50")

    # Sinal HULL16
    if close_prev <= hull_prev and close_now > hull_now:
        sinais.append("Compra HULL16")
    elif close_prev >= hull_prev and close_now < hull_now:
        sinais.append("Venda HULL16")
    elif close_now > hull_now:
        sinais.append("Acima HULL16")
    elif close_now < hull_now:
        sinais.append("Abaixo HULL16")

    return ", ".join(sinais) if sinais else "-"

def fetch_signals(symbols, timeframe):
    exchange = ccxt.kucoin()
    resultados = []
    for symbol in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['EMA50'] = get_ema(df['close'], 50)
            df['HULL16'] = get_hull(df['close'], 16)
            signal = get_signal(df)
            resultados.append({
                'Moeda': symbol,
                'Último Preço': df['close'].iloc[-1],
                'EMA50': df['EMA50'].iloc[-1],
                'HULL16': df['HULL16'].iloc[-1],
                'Sinal': signal
            })
        except Exception as e:
            resultados.append({'Moeda': symbol, 'Erro': str(e)})
    return resultados
