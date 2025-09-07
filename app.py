import streamlit as st
from kucoin_analyzer import fetch_signals
from config import SYMBOLS, TIMEFRAME

st.title("Análise EMA50 Kucoin")

resultados = []
progress_bar = st.progress(0)
status_text = st.empty()

total = len(SYMBOLS)

# Analisa moeda por moeda e atualiza barra
for idx, symbol in enumerate(SYMBOLS, 1):
    status_text.text(f"Analisando: {symbol} ({idx}/{total}) | Faltam: {total - idx}")
    result = fetch_signals([symbol], TIMEFRAME)[0]  # fetch_signals espera lista, pega resultado único
    resultados.append(result)
    progress_bar.progress(idx / total)

# Limpa barra e status após finalizar
progress_bar.empty()
status_text.empty()

# Exibe resultados finais
st.success("Análise finalizada!")
st.dataframe(resultados)
