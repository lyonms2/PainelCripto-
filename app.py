import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json

# Configuração da página
st.set_page_config(
    page_title="KuCoin Crypto Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

class KuCoinAPI:
    def __init__(self):
        self.base_url = "https://api.kucoin.com"
    
    def get_all_symbols(self):
        """Obter todos os símbolos disponíveis na KuCoin"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/symbols")
            if response.status_code == 200:
                return response.json()['data']
            return []
        except Exception as e:
            st.error(f"Erro ao obter símbolos: {e}")
            return []
    
    def get_ticker_24hr(self):
        """Obter dados de ticker 24h para todas as moedas"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/market/allTickers")
            if response.status_code == 200:
                return response.json()['data']['ticker']
            return []
        except Exception as e:
            st.error(f"Erro ao obter dados de ticker: {e}")
            return []
    
    def get_market_stats(self, symbol):
        """Obter estatísticas do mercado para um símbolo específico"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/market/stats?symbol={symbol}")
            if response.status_code == 200:
                return response.json()['data']
            return None
        except Exception as e:
            st.error(f"Erro ao obter estatísticas do mercado: {e}")
            return None
    
    def get_klines(self, symbol, type_="1hour", start_at=None, end_at=None):
        """Obter dados de candlestick (klines)"""
        try:
            params = {
                'symbol': symbol,
                'type': type_
            }
            if start_at:
                params['startAt'] = start_at
            if end_at:
                params['endAt'] = end_at
                
            response = requests.get(f"{self.base_url}/api/v1/market/candles", params=params)
            if response.status_code == 200:
                return response.json()['data']
            return []
        except Exception as e:
            st.error(f"Erro ao obter dados de klines: {e}")
            return []

def format_currency(value):
    """Formatar valores monetários"""
    try:
        val = float(value)
        if val >= 1e9:
            return f"${val/1e9:.2f}B"
        elif val >= 1e6:
            return f"${val/1e6:.2f}M"
        elif val >= 1e3:
            return f"${val/1e3:.2f}K"
        else:
            return f"${val:.2f}"
    except:
        return "N/A"

def format_percentage(value):
    """Formatar percentuais"""
    try:
        val = float(value) * 100
        return f"{val:+.2f}%"
    except:
        return "N/A"

@st.cache_data(ttl=60)  # Cache por 1 minuto
def load_crypto_data():
    """Carregar dados das criptomoedas"""
    api = KuCoinAPI()
    ticker_data = api.get_ticker_24hr()
    
    if not ticker_data:
        return pd.DataFrame()
    
    # Converter para DataFrame
    df = pd.DataFrame(ticker_data)
    
    # Filtrar apenas pares USD, USDT, BTC principais
    df = df[df['symbol'].str.contains('-USDT|=BTC|-USD', regex=True)]
    
    # Converter colunas numéricas
    numeric_columns = ['last', 'changePrice', 'changeRate', 'high', 'low', 'vol', 'volValue']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Adicionar colunas calculadas
    df['market_cap'] = df['last'] * df['vol']
    df['price_formatted'] = df['last'].apply(lambda x: f"${x:.4f}" if x < 1 else f"${x:.2f}")
    df['change_formatted'] = df['changeRate'].apply(format_percentage)
    df['volume_formatted'] = df['volValue'].apply(format_currency)
    
    return df.sort_values('volValue', ascending=False)

def create_price_chart(symbol_data, symbol):
    """Criar gráfico de preços"""
    api = KuCoinAPI()
    
    # Obter dados das últimas 24 horas
    end_time = int(time.time())
    start_time = end_time - (24 * 60 * 60)  # 24 horas atrás
    
    klines = api.get_klines(symbol, type_="1hour", start_at=start_time, end_at=end_time)
    
    if not klines:
        st.warning("Não foi possível obter dados históricos")
        return None
    
    # Converter klines para DataFrame
    df_klines = pd.DataFrame(klines, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
    
    # Converter timestamp para datetime
    df_klines['datetime'] = pd.to_datetime(df_klines['timestamp'], unit='s')
    
    # Converter para float
    for col in ['open', 'close', 'high', 'low', 'volume']:
        df_klines[col] = pd.to_numeric(df_klines[col], errors='coerce')
    
    # Criar gráfico de candlestick
    fig = go.Figure(data=go.Candlestick(
        x=df_klines['datetime'],
        open=df_klines['open'],
        high=df_klines['high'],
        low=df_klines['low'],
        close=df_klines['close'],
        name=symbol
    ))
    
    fig.update_layout(
        title=f"Preço de {symbol} - Últimas 24 horas",
        xaxis_title="Tempo",
        yaxis_title="Preço (USD)",
        template="plotly_white",
        height=400
    )
    
    return fig

def main():
    st.title("🚀 KuCoin Cryptocurrency Dashboard")
    st.markdown("Dashboard em tempo real com dados de criptomoedas da KuCoin")
    
    # Sidebar
    st.sidebar.title("⚙️ Configurações")
    
    # Carregar dados
    with st.spinner("Carregando dados das criptomoedas..."):
        df = load_crypto_data()
    
    if df.empty:
        st.error("Não foi possível carregar os dados. Tente novamente mais tarde.")
        return
    
    # Filtros na sidebar
    st.sidebar.subheader("🔍 Filtros")
    
    # Filtro por volume mínimo
    min_volume = st.sidebar.number_input(
        "Volume mínimo (USD)", 
        min_value=0, 
        value=100000, 
        step=50000,
        format="%d"
    )
    
    # Filtro por mudança de preço
    price_change = st.sidebar.selectbox(
        "Filtrar por mudança de preço",
        ["Todos", "Apenas ganhos", "Apenas perdas", "Ganhos > 5%", "Perdas > 5%"]
    )
    
    # Aplicar filtros
    df_filtered = df[df['volValue'] >= min_volume]
    
    if price_change == "Apenas ganhos":
        df_filtered = df_filtered[df_filtered['changeRate'] > 0]
    elif price_change == "Apenas perdas":
        df_filtered = df_filtered[df_filtered['changeRate'] < 0]
    elif price_change == "Ganhos > 5%":
        df_filtered = df_filtered[df_filtered['changeRate'] > 0.05]
    elif price_change == "Perdas > 5%":
        df_filtered = df_filtered[df_filtered['changeRate'] < -0.05]
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_coins = len(df_filtered)
        st.metric("Total de Moedas", total_coins)
    
    with col2:
        gainers = len(df_filtered[df_filtered['changeRate'] > 0])
        st.metric("Em Alta", gainers, delta=f"{gainers/total_coins*100:.1f}%")
    
    with col3:
        losers = len(df_filtered[df_filtered['changeRate'] < 0])
        st.metric("Em Baixa", losers, delta=f"-{losers/total_coins*100:.1f}%")
    
    with col4:
        total_volume = df_filtered['volValue'].sum()
        st.metric("Volume Total", format_currency(total_volume))
    
    # Abas principais
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Visão Geral", "🏆 Top Moedas", "📈 Análise", "🔍 Busca Detalhada"])
    
    with tab1:
        st.subheader("Maiores Variações (24h)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("🚀 **Maiores Altas**")
            top_gainers = df_filtered.nlargest(10, 'changeRate')[['symbol', 'price_formatted', 'change_formatted', 'volume_formatted']]
            st.dataframe(top_gainers, use_container_width=True, hide_index=True)
        
        with col2:
            st.write("📉 **Maiores Baixas**")
            top_losers = df_filtered.nsmallest(10, 'changeRate')[['symbol', 'price_formatted', 'change_formatted', 'volume_formatted']]
            st.dataframe(top_losers, use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("Top 20 Criptomoedas por Volume")
        
        top_20 = df_filtered.head(20)
        
        # Gráfico de barras do volume
        fig_volume = px.bar(
            top_20, 
            x='symbol', 
            y='volValue',
            title="Volume de Negociação (24h)",
            labels={'volValue': 'Volume (USD)', 'symbol': 'Símbolo'}
        )
        fig_volume.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_volume, use_container_width=True)
        
        # Tabela detalhada
        display_columns = ['symbol', 'price_formatted', 'change_formatted', 'volume_formatted', 'high', 'low']
        st.dataframe(
            top_20[display_columns].rename(columns={
                'symbol': 'Símbolo',
                'price_formatted': 'Preço',
                'change_formatted': 'Mudança 24h',
                'volume_formatted': 'Volume',
                'high': 'Alta 24h',
                'low': 'Baixa 24h'
            }),
            use_container_width=True,
            hide_index=True
        )
    
    with tab3:
        st.subheader("Análise de Distribuição")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de pizza - distribuição de mudanças
            change_dist = pd.cut(df_filtered['changeRate'], 
                               bins=[-float('inf'), -0.1, -0.05, 0, 0.05, 0.1, float('inf')],
                               labels=['< -10%', '-10% a -5%', '-5% a 0%', '0% a 5%', '5% a 10%', '> 10%'])
            
            fig_pie = px.pie(
                values=change_dist.value_counts().values,
                names=change_dist.value_counts().index,
                title="Distribuição de Mudanças de Preço (24h)"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Histograma de mudanças de preço
            fig_hist = px.histogram(
                df_filtered,
                x='changeRate',
                bins=30,
                title="Histograma de Mudanças de Preço (24h)",
                labels={'changeRate': 'Mudança (%)', 'count': 'Frequência'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    
    with tab4:
        st.subheader("Busca Detalhada")
        
        # Seleção de moeda
        symbols = df_filtered['symbol'].tolist()
        selected_symbol = st.selectbox("Selecione uma criptomoeda:", symbols)
        
        if selected_symbol:
            symbol_data = df_filtered[df_filtered['symbol'] == selected_symbol].iloc[0]
            
            # Informações detalhadas
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Preço Atual", symbol_data['price_formatted'])
                st.metric("Alta 24h", f"${symbol_data['high']:.4f}")
            
            with col2:
                st.metric("Mudança 24h", symbol_data['change_formatted'], 
                         delta=symbol_data['change_formatted'])
                st.metric("Baixa 24h", f"${symbol_data['low']:.4f}")
            
            with col3:
                st.metric("Volume", symbol_data['volume_formatted'])
                st.metric("Volume Base", f"{symbol_data['vol']:.0f}")
            
            # Gráfico de preços
            fig_price = create_price_chart(symbol_data, selected_symbol)
            if fig_price:
                st.plotly_chart(fig_price, use_container_width=True)
    
    # Rodapé
    st.markdown("---")
    st.markdown(
        """
        **📊 KuCoin Crypto Dashboard**  
        Dados fornecidos pela API pública da KuCoin. Atualização automática a cada minuto.  
        ⚠️ *Este dashboard é apenas para fins informativos. Não constitui aconselhamento financeiro.*
        """
    )
    
    # Auto-refresh
    st.sidebar.markdown("---")
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (30s)")
    if auto_refresh:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()