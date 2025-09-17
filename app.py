import streamlit as st
from indicator import fetch_signals
from config import SYMBOLS, TIMEFRAME

st.title("Análise EMA55, HULL55 e WaveTrend Kucoin")

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
    st.dataframe(resultados)
    st.info("Colunas WT_OB/WT_OS = Sinal de Sobrecompra/Sobrevenda. WT_Azul→Rosa/WT_Rosa→Azul = Mudança de cor do WaveTrend.")
else:
    st.info("Clique em 'Atualizar dados' para rodar a análise.")
