import ccxt
import pandas as pd
import numpy as np
from strategies import strategy_ema_hull_cross

def get_ema(prices, period=55):
    return prices.ewm(span=period).mean()

def get_hull(prices, period=55):
    def wma(series, period):
        weights = np.arange(1, period + 1)
        return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    half_length = int(period / 2)
    sqrt_length = int(np.sqrt(period))
    wma_half = wma(prices, half_length)
    wma_full = wma(prices, period)
    hull = wma(2 * wma_half - wma_full, sqrt_length)
    return hull

def add_wavetrend(df, 
                  src='hlc3', 
                  channel_length=10, 
                  average_length=21, 
                  signal_length=4, 
                  reversion_threshold=100):
    # Calcula o preço conforme o parâmetro src
    if src == 'hlc3':
        price = (df['high'] + df['low'] + df['close']) / 3
    elif src == 'hl2':
        price = (df['high'] + df['low']) / 2
    elif src == 'ohlc4':
        price = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    elif src == 'oc2':
        price = (df['open'] + df['close']) / 2
    elif src == 'high':
        price = df['high']
    elif src == 'low':
        price = df['low']
    elif src == 'open':
        price = df['open']
    else:
        price = df['close']

    esa = price.ewm(span=channel_length, adjust=False).mean()
    d = price - esa
    d_std = d.rolling(window=channel_length).std()
    ci = (price - esa) / (0.015 * d_std)
    wt = ci.ewm(span=average_length, adjust=False).mean()
    wt_signal = wt.rolling(window=signal_length).mean()

    df['WT'] = wt
    df['WT_signal'] = wt_signal

    # Sinais
    df['WT_OB'] = ((wt > wt_signal) & (wt.shift(1) <= wt_signal.shift(1)) & (wt > reversion_threshold))
    df['WT_OS'] = ((wt < wt_signal) & (wt.shift(1) >= wt_signal.shift(1)) & (wt < -reversion_threshold))
    df['WT_Sobrecompra'] = (wt > reversion_threshold)
    df['WT_Sobrevenda'] = (wt < -reversion_threshold)
    df['WT_Azul_Rosa'] = (wt > wt_signal) & (wt.shift(1) <= wt_signal.shift(1))     # cruzamento para cima
    df['WT_Rosa_Azul'] = (wt < wt_signal) & (wt.shift(1) >= wt_signal.shift(1))     # cruzamento para baixo
    return df

def get_signal(df):
    if len(df) < 3:
        return "-"
    close_now = df['close'].iloc[-2]
    close_prev = df['close'].iloc[-3]
    ema_now = df['EMA55'].iloc[-2]
    ema_prev = df['EMA55'].iloc[-3]
    hull_now = df['HULL55'].iloc[-2]
    hull_prev = df['HULL55'].iloc[-3]

    sinais = []

    if close_prev <= ema_prev and close_now > ema_now:
        sinais.append("Compra EMA55")
    elif close_prev >= ema_prev and close_now < ema_now:
        sinais.append("Venda EMA55")
    elif close_now > ema_now:
        sinais.append("Acima EMA55")
    elif close_now < ema_now:
        sinais.append("Abaixo EMA55")

    if close_prev <= hull_prev and close_now > hull_now:
        sinais.append("Compra HULL55")
    elif close_prev >= hull_prev and close_now < hull_now:
        sinais.append("Venda HULL55")
    elif close_now > hull_now:
        sinais.append("Acima HULL55")
    elif close_now < hull_now:
        sinais.append("Abaixo HULL55")

    return ", ".join(sinais) if sinais else "-"

def get_wavetrend_signals(df):
    if len(df) < 2 or 'WT' not in df or 'WT_signal' not in df:
        return "-", "-", "-", "-", "-", "-"
    idx = -1
    ob = df['WT_OB'].iloc[idx]
    os = df['WT_OS'].iloc[idx]
    sobrecompra = df['WT_Sobrecompra'].iloc[idx]
    sobrevenda = df['WT_Sobrevenda'].iloc[idx]
    azul_rosa = df['WT_Azul_Rosa'].iloc[idx]
    rosa_azul = df['WT_Rosa_Azul'].iloc[idx]
    return ob, os, sobrecompra, sobrevenda, azul_rosa, rosa_azul

def fetch_signals(symbols, timeframe):
    exchange = ccxt.kucoin()
    resultados = []
    for symbol in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['EMA55'] = get_ema(df['close'], 55)
            df['HULL55'] = get_hull(df['close'], 55)
            df = add_wavetrend(df)  # Adiciona colunas do WT
            signal = get_signal(df)
            ob, os, sobrecompra, sobrevenda, azul_rosa, rosa_azul = get_wavetrend_signals(df)
            resultados.append({
                'Moeda': symbol,
                'Último Preço': df['close'].iloc[-1],
                'EMA55': df['EMA55'].iloc[-1],
                'HULL55': df['HULL55'].iloc[-1],
                'Sinal': signal,
                'WT_OB': ob,
                'WT_OS': os,
                'WT_Sobrecompra': sobrecompra,
                'WT_Sobrevenda': sobrevenda,
                'WT_Azul→Rosa': azul_rosa,
                'WT_Rosa→Azul': rosa_azul
            })
        except Exception as e:
            resultados.append({'Moeda': symbol, 'Erro': str(e)})
    return resultados
