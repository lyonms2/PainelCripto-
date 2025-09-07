import streamlit as st
from indicator import fetch_signals
from config import SYMBOLS, TIMEFRAME
from view import show_results

st.title("Análise EMA/HULL Kucoin")

if st.button("Atualizar dados"):
    resultados = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(SYMBOLS)
    for idx, symbol in enumerate(SYMBOLS, 1):
        status_text.text(f"Analisando: {symbol} ({idx}/{total}) | Faltam: {total - idx}")
        result = fetch_signals([symbol], TIMEFRAME)[0]
        resultados.append(result)
        progress_bar.progress(idx / total)
    progress_bar.empty()
    status_text.empty()
    st.success("Análise finalizada!")
    show_results(resultados)
else:
    st.info("Clique em 'Atualizar dados' para rodar a análise.")
