import ccxt
import pandas as pd

def get_ema(prices, period=50):
    return prices.ewm(span=period).mean()

def get_signal(df):
    # Precisa de pelo menos 2 candles para detectar cruzamento
    if len(df) < 2:
        return "-"
    close_now = df['close'].iloc[-1]
    close_prev = df['close'].iloc[-2]
    ema_now = df['EMA50'].iloc[-1]
    ema_prev = df['EMA50'].iloc[-2]

    # Cruzamento para cima (Compra)
    if close_prev <= ema_prev and close_now > ema_now:
        return "Compra"
    # Cruzamento para baixo (Venda)
    elif close_prev >= ema_prev and close_now < ema_now:
        return "Venda"
    # Preço acima da EMA50 (sem cruzamento)
    elif close_now > ema_now:
        return "Preço acima da EMA50"
    # Preço abaixo da EMA50 (sem cruzamento)
    elif close_now < ema_now:
        return "Preço abaixo da EMA50"
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
