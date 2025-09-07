import streamlit as st
import pandas as pd
from kucoin_analyzer import fetch_signals
from config import SYMBOLS, TIMEFRAME

st.title("Análise EMA 50 - Criptomoedas KuCoin")
resultados = []
progress_bar = st.progress(0)
status_text = st.empty()

total = len(SYMBOLS)

for idx, symbol in enumerate(SYMBOLS, 1):
    status_text.text(f"Analisando: {symbol} ({idx}/{total}) | Faltam: {total - idx}")
    result = fetch_signals([symbol], TIMEFRAME)[0]
    resultados.append(result)
    progress_bar.progress(idx / total)

st.write("Análise finalizada!")
st.dataframe(resultados)
