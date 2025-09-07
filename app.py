import streamlit as st
import ccxt
from kucoin_analyzer import fetch_signals

st.title("Análise EMA50 Kucoin")

# Slider para o usuário escolher o range de volume
min_volume, max_volume = st.slider(
    'Selecione o range de volume (USDT):',
    min_value=100_000, max_value=10_000_000,
    value=(100_000, 10_000_000), step=50_000
)

# Carrega os mercados da Kucoin e filtra apenas pares USDT ativos
st.info("Carregando lista de pares USDT da Kucoin...")
exchange = ccxt.kucoin()
markets = exchange.load_markets()
usdt_pairs = [s for s, m in markets.items() if s.endswith('/USDT') and m['active']]

st.write(f"Total de pares USDT ativos: {len(usdt_pairs)}")

# Filtra pares pelo volume dentro do range selecionado
st.info("Filtrando pares pelo volume negociado (USDT)...")
filtered_symbols = []
progress_vol = st.progress(0)
for idx, symbol in enumerate(usdt_pairs, 1):
    try:
        ticker = exchange.fetch_ticker(symbol)
        volume = ticker['quoteVolume']
        if min_volume <= volume <= max_volume:
            filtered_symbols.append(symbol)
    except Exception:
        pass
    progress_vol.progress(idx / len(usdt_pairs))
progress_vol.empty()

st.success(f"Pares filtrados pelo volume: {len(filtered_symbols)}")

# Só continua se houver pares filtrados
if len(filtered_symbols) == 0:
    st.warning("Nenhum par encontrado dentro do volume selecionado.")
    st.stop()

# Barra de progresso para análise dos sinais
st.header("Analisando sinais EMA50...")
resultados = []
progress_bar = st.progress(0)
status_text = st.empty()
total = len(filtered_symbols)

for idx, symbol in enumerate(filtered_symbols, 1):
    status_text.text(f"Analisando: {symbol} ({idx}/{total}) | Faltam: {total - idx}")
    result = fetch_signals([symbol], '1h')[0]
    resultados.append(result)
    progress_bar.progress(idx / total)

progress_bar.empty()
status_text.empty()
st.success("Análise finalizada!")
st.dataframe(resultados)
