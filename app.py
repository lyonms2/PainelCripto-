import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configuração da página
st.set_page_config(
    page_title="KuCoin Crypto Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_kucoin_data():
    """Obter dados das criptomoedas da KuCoin"""
    try:
        response = requests.get("https://api.kucoin.com/api/v1/market/allTickers", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('data', {}).get('ticker', [])
        else:
            st.error(f"Erro na API: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Erro de conexão: {str(e)}")
        return []
    except Exception as e:
        st.error(f"Erro inesperado: {str(e)}")
        return []

def process_data(raw_data):
    """Processar dados brutos da API"""
    if not raw_data:
        return pd.DataFrame()
    
    try:
        df = pd.DataFrame(raw_data)
        
        # Filtrar apenas pares USDT para simplicidade
        df = df[df['symbol'].str.contains('-USDT$', regex=True, na=False)]
        
        # Converter colunas numéricas essenciais
        numeric_cols = ['last', 'changeRate', 'volValue', 'vol']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Filtrar dados válidos
        df = df.dropna(subset=['last', 'changeRate', 'volValue'])
        df = df[df['volValue'] > 0]  # Volume deve ser positivo
        
        # Ordenar por volume
        df = df.sort_values('volValue', ascending=False)
        
        return df
        
    except Exception as e:
        st.error(f"Erro ao processar dados: {str(e)}")
        return pd.DataFrame()

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
    except (ValueError, TypeError):
        return "N/A"

def format_percentage(value):
    """Formatar percentuais"""
    try:
        return f"{float(value)*100:+.2f}%"
    except (ValueError, TypeError):
        return "N/A"

@st.cache_data(ttl=300)  # Cache por 5 minutos
def load_crypto_data():
    """Carregar e processar dados com cache"""
    raw_data = get_kucoin_data()
    return process_data(raw_data)

def create_volume_chart(df):
    """Criar gráfico de volume"""
    try:
        top_20 = df.head(20)
        fig = px.bar(
            top_20,
            x='symbol',
            y='volValue',
            title="Top 20 - Volume de Negociação (24h)",
            color='changeRate',
            color_continuous_scale='RdYlGn',
            labels={'volValue': 'Volume (USD)', 'symbol': 'Símbolo'}
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500,
            showlegend=False
        )
        return fig
    except Exception as e:
        st.error(f"Erro ao criar gráfico: {str(e)}")
        return None

def create_distribution_chart(df):
    """Criar gráfico de distribuição de mudanças"""
    try:
        # Categorizar mudanças
        conditions = [
            df['changeRate'] < -0.1,
            (df['changeRate'] >= -0.1) & (df['changeRate'] < -0.05),
            (df['changeRate'] >= -0.05) & (df['changeRate'] < 0),
            (df['changeRate'] >= 0) & (df['changeRate'] < 0.05),
            (df['changeRate'] >= 0.05) & (df['changeRate'] < 0.1),
            df['changeRate'] >= 0.1
        ]
        
        choices = ['< -10%', '-10% a -5%', '-5% a 0%', '0% a 5%', '5% a 10%', '> 10%']
        df['change_category'] = pd.Series(dtype='object')
        
        for i, condition in enumerate(conditions):
            df.loc[condition, 'change_category'] = choices[i]
        
        # Contar categorias
        counts = df['change_category'].value_counts()
        
        fig = px.pie(
            values=counts.values,
            names=counts.index,
            title="Distribuição de Mudanças de Preço (24h)",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        return fig
    except Exception as e:
        st.error(f"Erro ao criar gráfico de distribuição: {str(e)}")
        return None

def main():
    st.title("🚀 KuCoin Cryptocurrency Dashboard")
    st.markdown("Dashboard em tempo real com dados de criptomoedas da KuCoin")
    
    # Sidebar
    st.sidebar.title("⚙️ Configurações")
    
    # Filtro de volume mínimo
    min_volume = st.sidebar.selectbox(
        "Volume mínimo",
        [0, 50000, 100000, 500000, 1000000],
        index=2,
        format_func=lambda x: f"${x:,}"
    )
    
    # Carregar dados
    with st.spinner("🔄 Carregando dados..."):
        df = load_crypto_data()
    
    if df.empty:
        st.error("❌ Não foi possível carregar os dados. Verifique sua conexão e tente novamente.")
        st.stop()
    
    # Aplicar filtro de volume
    df_filtered = df[df['volValue'] >= min_volume]
    
    if df_filtered.empty:
        st.warning(f"⚠️ Nenhuma moeda encontrada com volume superior a {format_currency(min_volume)}")
        st.stop()
    
    # Métricas principais
    st.subheader("📊 Resumo do Mercado")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Moedas", len(df_filtered))
    
    with col2:
        gainers = len(df_filtered[df_filtered['changeRate'] > 0])
        st.metric("Em Alta", gainers, delta=f"{gainers/len(df_filtered)*100:.1f}%")
    
    with col3:
        losers = len(df_filtered[df_filtered['changeRate'] < 0])
        st.metric("Em Baixa", losers, delta=f"-{losers/len(df_filtered)*100:.1f}%")
    
    with col4:
        total_volume = df_filtered['volValue'].sum()
        st.metric("Volume Total", format_currency(total_volume))
    
    # Tabs principais
    tab1, tab2, tab3 = st.tabs(["🏆 Top Moedas", "📈 Maiores Variações", "📊 Análise"])
    
    with tab1:
        st.subheader("Top 20 Criptomoedas por Volume")
        
        # Gráfico de volume
        fig_volume = create_volume_chart(df_filtered)
        if fig_volume:
            st.plotly_chart(fig_volume, use_container_width=True)
        
        # Tabela detalhada
        top_20 = df_filtered.head(20).copy()
        top_20['Preço'] = top_20['last'].apply(lambda x: f"${x:.4f}")
        top_20['Mudança 24h'] = top_20['changeRate'].apply(format_percentage)
        top_20['Volume'] = top_20['volValue'].apply(format_currency)
        
        display_df = top_20[['symbol', 'Preço', 'Mudança 24h', 'Volume']]
        display_df.columns = ['Símbolo', 'Preço', 'Mudança 24h', 'Volume']
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("Maiores Variações nas Últimas 24h")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🚀 Maiores Altas")
            top_gainers = df_filtered.nlargest(10, 'changeRate').copy()
            top_gainers['Preço'] = top_gainers['last'].apply(lambda x: f"${x:.4f}")
            top_gainers['Mudança'] = top_gainers['changeRate'].apply(format_percentage)
            
            gainers_display = top_gainers[['symbol', 'Preço', 'Mudança']]
            gainers_display.columns = ['Símbolo', 'Preço', 'Mudança']
            st.dataframe(gainers_display, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("### 📉 Maiores Baixas")
            top_losers = df_filtered.nsmallest(10, 'changeRate').copy()
            top_losers['Preço'] = top_losers['last'].apply(lambda x: f"${x:.4f}")
            top_losers['Mudança'] = top_losers['changeRate'].apply(format_percentage)
            
            losers_display = top_losers[['symbol', 'Preço', 'Mudança']]
            losers_display.columns = ['Símbolo', 'Preço', 'Mudança']
            st.dataframe(losers_display, use_container_width=True, hide_index=True)
    
    with tab3:
        st.subheader("Análise de Distribuição do Mercado")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de distribuição
            fig_dist = create_distribution_chart(df_filtered)
            if fig_dist:
                st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Estatísticas
            st.markdown("### 📊 Estatísticas")
            
            avg_change = df_filtered['changeRate'].mean()
            median_change = df_filtered['changeRate'].median()
            std_change = df_filtered['changeRate'].std()
            
            st.metric("Mudança Média", format_percentage(avg_change))
            st.metric("Mudança Mediana", format_percentage(median_change))
            st.metric("Desvio Padrão", format_percentage(std_change))
            
            # Volume por faixa de mudança
            st.markdown("### 💰 Volume por Performance")
            volume_up = df_filtered[df_filtered['changeRate'] > 0]['volValue'].sum()
            volume_down = df_filtered[df_filtered['changeRate'] < 0]['volValue'].sum()
            
            st.metric("Volume - Moedas em Alta", format_currency(volume_up))
            st.metric("Volume - Moedas em Baixa", format_currency(volume_down))
    
    # Busca específica
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Busca Específica")
    search_term = st.sidebar.text_input("Digite o símbolo (ex: BTC-USDT)")
    
    if search_term:
        search_result = df_filtered[df_filtered['symbol'].str.contains(search_term.upper(), na=False)]
        if not search_result.empty:
            st.sidebar.success(f"✅ Encontrado: {search_term.upper()}")
            coin_data = search_result.iloc[0]
            st.sidebar.metric("Preço", f"${coin_data['last']:.4f}")
            st.sidebar.metric("Mudança 24h", format_percentage(coin_data['changeRate']))
            st.sidebar.metric("Volume", format_currency(coin_data['volValue']))
        else:
            st.sidebar.warning(f"❌ {search_term} não encontrado")
    
    # Rodapé
    st.markdown("---")
    st.markdown(
        """
        **📊 KuCoin Crypto Dashboard** | Dados em tempo real da API pública da KuCoin  
        ⚠️ *Apenas para fins informativos - não é aconselhamento financeiro*  
        🔄 *Dados atualizados a cada 5 minutos*
        """
    )
    
    # Botão de atualização manual
    if st.button("🔄 Atualizar Dados Agora", type="primary"):
        st.cache_data.clear()
        st.rerun()

if __name__ == "__main__":
    main()
