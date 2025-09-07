import streamlit as st

def show_results(resultados):
    st.subheader("Resultados da Análise")
    st.dataframe(resultados)