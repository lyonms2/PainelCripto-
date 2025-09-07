import ccxt
import pandas as pd

def get_ema(prices, period=50):
    return prices.ewm(span=period).mean()

def get_signal(df):
    if df['close'].iloc[-1] > df['EMA50'].iloc[-1]:
        return "Compra"
    elif df['close'].iloc[-1] < df['EMA50'].iloc[-1]:
        return "Venda"
    else:
        return "-"

def fetch_signals(symbols, timeframe):
    exchange = ccxt.kucoin()
    resultados = []
    for symbol in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['EMA50'] = get_ema(df['close'], 50)
            signal = get_signal(df)
            resultados.append({
                'Moeda': symbol,
                'Último Preço': df['close'].iloc[-1],
                'EMA50': df['EMA50'].iloc[-1],
                'Sinal': signal
            })
        except Exception as e:
            resultados.append({'Moeda': symbol, 'Erro': str(e)})
    return resultados