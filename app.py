import streamlit as st
import pandas as pd
from kucoin_analyzer import fetch_signals
from config import SYMBOLS, TIMEFRAME

st.title("Análise EMA 50 - Criptomoedas KuCoin")

resultados = fetch_signals(SYMBOLS, TIMEFRAME)

st.dataframe(pd.DataFrame(resultados))