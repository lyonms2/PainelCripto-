import streamlit as st
import ccxt
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("Análise WaveTrend Oscillator Kucoin")

# Parâmetros principais
SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'ADA/USDT', 'XRP/USDT', 'DOGE/USDT', 'TRX/USDT', 'LINK/USDT', 
    'LTC/USDT', 'DOT/USDT', 'BCH/USDT', 'SHIB/USDT', 'AVAX/USDT', 'OP/USDT', 'UNI/USDT', 'ATOM/USDT', 'ETC/USDT', 'XLM/USDT',
    'FIL/USDT', 'APT/USDT', 'SUI/USDT', 'HBAR/USDT', 'AR/USDT', 'INJ/USDT', 'VET/USDT', 'PEPE/USDT', 'NEAR/USDT',
    'GRT/USDT', 'CRO/USDT', 'STX/USDT', 'ALGO/USDT', 'QNT/USDT', 'IMX/USDT', 'WIF/USDT', 'TUSD/USDT', 'MINA/USDT',
    'DYDX/USDT', 'EGLD/USDT', 'AKT/USDT', 'TIA/USDT', 'JTO/USDT', 'MKR/USDT', 'AAVE/USDT', 'WBTC/USDT', 'PYTH/USDT', 'SAND/USDT',
    'CAKE/USDT', 'XMR/USDT', 'CHZ/USDT', 'BLUR/USDT', 'BTT/USDT', 'GMX/USDT', 'LDO/USDT', 'ENJ/USDT', 'FET/USDT', 'OSMO/USDT',
    'BICO/USDT', 'RPL/USDT', 'DYM/USDT', 'ZIL/USDT', 'SFP/USDT', 'SKL/USDT', '1INCH/USDT', 'GMT/USDT',
    'FLOW/USDT', 'CKB/USDT', 'XDC/USDT', 'XEC/USDT', 'SSV/USDT', 'ID/USDT', 'MEME/USDT', 'BOME/USDT',
    'BAND/USDT', 'BICO/USDT', 'LPT/USDT', 'SCRT/USDT', 'NKN/USDT', 'OGN/USDT', 'STORJ/USDT',
    'SUN/USDT', 'SXP/USDT', 'XNO/USDT', 'YGG/USDT', 'ZRX/USDT', 'RUNE/USDT', 'CELO/USDT'
]
TIMEFRAME = '15m'

# --- Funções WaveTrend ---
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
    df['OB'] = ((wt > wt_signal) & (wt.shift(1) <= wt_signal.shift(1)) & (wt > reversion_threshold))
    df['OS'] = ((wt < wt_signal) & (wt.shift(1) >= wt_signal.shift(1)) & (wt < -reversion_threshold))
    df['Sobrecompra'] = (wt > reversion_threshold)
    df['Sobrevenda'] = (wt < -reversion_threshold)
    df['Azul→Rosa'] = (wt > wt_signal) & (wt.shift(1) <= wt_signal.shift(1))
    df['Rosa→Azul'] = (wt < wt_signal) & (wt.shift(1) >= wt_signal.shift(1))
    return df

def fetch_wavetrend_signals(symbols, timeframe):
    exchange = ccxt.kucoin()
    resultados = []
    for symbol in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = add_wavetrend(df)
            idx = -1
            resultados.append({
                'Moeda': symbol,
                'Último Preço': df['close'].iloc[idx],
                'WT': round(df['WT'].iloc[idx], 2),
                'WT_signal': round(df['WT_signal'].iloc[idx], 2),
                'Sinal OB': "✅" if df['OB'].iloc[idx] else "",
                'Sinal OS': "✅" if df['OS'].iloc[idx] else "",
                'Sobrecompra': "🟣" if df['Sobrecompra'].iloc[idx] else "",
                'Sobrevenda': "🔵" if df['Sobrevenda'].iloc[idx] else "",
                'Azul→Rosa': "⬆️" if df['Azul→Rosa'].iloc[idx] else "",
                'Rosa→Azul': "⬇️" if df['Rosa→Azul'].iloc[idx] else ""
            })
        except Exception as e:
            resultados.append({'Moeda': symbol, 'Erro': str(e)})
    return resultados

# --- Interface ---
st.info("""
- **Sinal OB:** (✅) Overbought — Sinal de Sobrecompra
- **Sinal OS:** (✅) Oversold — Sinal de Sobrevenda
- **Sobrecompra:** (🟣) Está dentro da zona de sobrecompra
- **Sobrevenda:** (🔵) Está dentro da zona de sobrevenda
- **Azul→Rosa:** (⬆️) WT cruzou para cima da linha de sinal (tendência de alta)
- **Rosa→Azul:** (⬇️) WT cruzou para baixo da linha de sinal (tendência de baixa)
""")

if st.button("Atualizar análise"):
    resultados = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(SYMBOLS)

    for idx, symbol in enumerate(SYMBOLS, 1):
        status_text.text(f"Analisando: {symbol} ({idx}/{total})")
        result = fetch_wavetrend_signals([symbol], TIMEFRAME)[0]
        resultados.append(result)
        progress_bar.progress(idx / total)

    progress_bar.empty()
    status_text.empty()
    st.success("Análise finalizada!")
    st.dataframe(resultados, use_container_width=True)
else:
    st.info("Clique em 'Atualizar análise' para rodar a análise.")
