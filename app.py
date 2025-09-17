import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="WaveTrend Oscillator Pro",
    page_icon="📈",
    layout="wide"
)

# CSS customizado
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .signal-positive {
        background-color: #00d4aa;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .signal-negative {
        background-color: #ff6b6b;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .signal-neutral {
        background-color: #74b9ff;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 WaveTrend Oscillator Pro - Análise Avançada")
st.markdown("---")

# Sidebar com configurações
st.sidebar.header("⚙️ Configurações")

# Parâmetros do WaveTrend
st.sidebar.subheader("Parâmetros WaveTrend")
channel_length = st.sidebar.slider("Channel Length", 5, 30, 10)
average_length = st.sidebar.slider("Average Length", 10, 50, 21)
signal_length = st.sidebar.slider("Signal Length", 2, 10, 4)
reversion_threshold = st.sidebar.slider("Threshold", 50, 150, 100)

# Configurações de timeframe
timeframes = ['1h', '4h', '1d', '1w']
selected_timeframe = st.sidebar.selectbox("Timeframe", timeframes, index=1)

# Configurações de fonte de preço
price_sources = ['hlc3', 'hl2', 'ohlc4', 'oc2', 'close', 'high', 'low', 'open']
price_source = st.sidebar.selectbox("Fonte do Preço", price_sources, index=0)

# Auto-refresh
auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)")
if auto_refresh:
    st.sidebar.write("🔄 Próxima atualização em breve...")

# Símbolos principais
SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'HYPE/USDT', 'PUMP/USDT', 'ENA/USDT', 
    'FARTCOIN/USDT', 'BONK/USDT', 'BNB/USDT', 'ADA/USDT', 'XRP/USDT', 'DOGE/USDT',
    'TRX/USDT', 'LINK/USDT', 'LTC/USDT','PENGU/USDT', 'DOT/USDT', 'BCH/USDT', 
    'SHIB/USDT', 'AVAX/USDT', 'OP/USDT', 'UNI/USDT', 'ATOM/USDT', 'ETC/USDT', 
    'XLM/USDT', 'FIL/USDT', 'APT/USDT', 'SUI/USDT', 'HBAR/USDT', 'ZORA/USDT', 
    'AR/USDT', 'INJ/USDT', 'PEPE/USDT', 'NEAR/USDT', 'STX/USDT', 'ALGO/USDT', 
    'IMX/USDT', 'WIF/USDT', 'MINA/USDT', 'DYDX/USDT', 'TIA/USDT', 'JTO/USDT', 
    'AAVE/USDT', 'PYTH/USDT', 'SAND/USDT', 'CAKE/USDT', 'XMR/USDT', 'BLUR/USDT', 
    'GMX/USDT', 'LDO/USDT', 'FET/USDT', 'DYM/USDT', 'GMT/USDT', 'MEME/USDT', 
    'BOME/USDT', 'YGG/USDT', 'RUNE/USDT', 'CELO/USDT', 'WLD/USDT', 'ONDO/USDT', 
    'SEI/USDT', 'JUP/USDT', 'POPCAT/USDT', 'TAO/USDT', 'TON/USDT'
]

@st.cache_data(ttl=30)
def calculate_wavetrend(df, src='hlc3', channel_length=10, average_length=21, 
                       signal_length=4, reversion_threshold=100):
    """Calcula o WaveTrend Oscillator com cache"""
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

    # Cálculos WaveTrend
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
    df['Bullish_Cross'] = (wt > wt_signal) & (wt.shift(1) <= wt_signal.shift(1))
    df['Bearish_Cross'] = (wt < wt_signal) & (wt.shift(1) >= wt_signal.shift(1))
    
    return df

def get_signal_strength(wt, wt_signal, threshold):
    """Calcula a força do sinal"""
    if abs(wt) > threshold * 1.5:
        return "🔴 FORTE" if wt > 0 else "🟢 FORTE"
    elif abs(wt) > threshold:
        return "🟡 MODERADO"
    else:
        return "⚪ FRACO"

def fetch_data_and_analyze(symbols, timeframe, params):
    """Busca dados e realiza análise"""
    exchange = ccxt.kucoin()
    resultados = []
    
    for symbol in symbols:
        try:
            # Busca mais dados para análise histórica
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=200)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Calcula WaveTrend
            df = calculate_wavetrend(df, **params)
            
            # Análise da última vela
            idx = -1
            current = df.iloc[idx]
            prev = df.iloc[idx-1]
            
            # Calcula variação de preço
            price_change = ((current['close'] - prev['close']) / prev['close']) * 100
            
            # Determina tendência
            if current['WT'] > current['WT_signal']:
                trend = "📈 BULLISH"
            else:
                trend = "📉 BEARISH"
                
            # Força do sinal
            signal_strength = get_signal_strength(current['WT'], current['WT_signal'], params['reversion_threshold'])
            
            resultados.append({
                'Moeda': symbol.replace('/USDT', ''),
                'Preço': f"${current['close']:,.4f}",
                'Variação (%)': f"{price_change:+.2f}%",
                'WT': round(current['WT'], 2),
                'WT Signal': round(current['WT_signal'], 2),
                'Tendência': trend,
                'Força': signal_strength,
                'OB': "✅" if current['OB'] else "",
                'OS': "✅" if current['OS'] else "",
                'Sobrecompra': "🟣" if current['Sobrecompra'] else "",
                'Sobrevenda': "🔵" if current['Sobrevenda'] else "",
                'Bull Cross': "⬆️" if current['Bullish_Cross'] else "",
                'Bear Cross': "⬇️" if current['Bearish_Cross'] else "",
                'Volume': f"{current['volume']:,.0f}",
                'raw_data': df  # Para gráficos detalhados
            })
            
        except Exception as e:
            resultados.append({
                'Moeda': symbol.replace('/USDT', ''),
                'Erro': str(e)[:50],
                'raw_data': None
            })
            
    return resultados

def create_wavetrend_chart(df, symbol):
    """Cria gráfico detalhado do WaveTrend"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=[f'{symbol} - Preço', 'WaveTrend Oscillator'],
        row_heights=[0.3, 0.7]
    )
    
    # Gráfico de preço (candlestick)
    fig.add_trace(
        go.Candlestick(
            x=df['datetime'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Preço"
        ),
        row=1, col=1
    )
    
    # WaveTrend
    fig.add_trace(
        go.Scatter(
            x=df['datetime'],
            y=df['WT'],
            mode='lines',
            name='WT',
            line=dict(color='#2E86AB', width=2)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['datetime'],
            y=df['WT_signal'],
            mode='lines',
            name='WT Signal',
            line=dict(color='#F24236', width=2)
        ),
        row=2, col=1
    )
    
    # Linhas de referência
    fig.add_hline(y=reversion_threshold, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=-reversion_threshold, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)
    
    # Sinais de compra/venda
    buy_signals = df[df['OS'] == True]
    sell_signals = df[df['OB'] == True]
    
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals['datetime'],
                y=buy_signals['WT'],
                mode='markers',
                marker=dict(symbol='triangle-up', size=12, color='green'),
                name='Sinal Compra',
                showlegend=True
            ),
            row=2, col=1
        )
    
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals['datetime'],
                y=sell_signals['WT'],
                mode='markers',
                marker=dict(symbol='triangle-down', size=12, color='red'),
                name='Sinal Venda',
                showlegend=True
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        title=f"Análise WaveTrend - {symbol}",
        xaxis_rangeslider_visible=False,
        height=600,
        showlegend=True,
        template="plotly_dark"
    )
    
    return fig

# Interface principal
col1, col2, col3, col4 = st.columns(4)

# Botões de ação
if col1.button("🔄 Atualizar Análise", type="primary"):
    st.session_state['update_data'] = True

if col2.button("📊 Análise Completa"):
    st.session_state['full_analysis'] = True

if col3.button("🎯 Top Sinais"):
    st.session_state['top_signals'] = True

if col4.button("🔍 Análise Individual"):
    st.session_state['individual_analysis'] = True

# Auto-refresh logic
if auto_refresh and 'last_update' not in st.session_state:
    st.session_state['last_update'] = time.time()
    st.session_state['update_data'] = True

if auto_refresh and time.time() - st.session_state.get('last_update', 0) > 30:
    st.session_state['last_update'] = time.time()
    st.session_state['update_data'] = True

# Parâmetros para análise
params = {
    'src': price_source,
    'channel_length': channel_length,
    'average_length': average_length,
    'signal_length': signal_length,
    'reversion_threshold': reversion_threshold
}

# Análise principal
if st.session_state.get('update_data', False):
    with st.spinner('🔄 Analisando mercado...'):
        resultados = fetch_data_and_analyze(SYMBOLS, selected_timeframe, params)
    
    st.session_state['resultados'] = resultados
    st.session_state['update_data'] = False
    st.success("✅ Análise finalizada!")

# Exibir resultados
if 'resultados' in st.session_state:
    resultados = st.session_state['resultados']
    
    # Métricas gerais
    total_coins = len([r for r in resultados if 'Erro' not in r])
    bullish_count = len([r for r in resultados if 'Erro' not in r and '📈' in r.get('Tendência', '')])
    bearish_count = len([r for r in resultados if 'Erro' not in r and '📉' in r.get('Tendência', '')])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de Moedas", total_coins)
    col2.metric("Tendência Alta", bullish_count, delta=f"{(bullish_count/total_coins)*100:.1f}%")
    col3.metric("Tendência Baixa", bearish_count, delta=f"{(bearish_count/total_coins)*100:.1f}%")
    col4.metric("Timeframe", selected_timeframe)
    
    st.markdown("---")
    
    # Tabela principal
    df_results = pd.DataFrame([r for r in resultados if 'Erro' not in r])
    if not df_results.empty:
        # Filtros
        col1, col2 = st.columns(2)
        with col1:
            trend_filter = st.selectbox("Filtrar por Tendência", ["Todos", "📈 BULLISH", "📉 BEARISH"])
        with col2:
            signal_filter = st.selectbox("Filtrar por Sinais", ["Todos", "Com OB", "Com OS", "Com Cruzamentos"])
        
        # Aplicar filtros
        filtered_df = df_results.copy()
        if trend_filter != "Todos":
            filtered_df = filtered_df[filtered_df['Tendência'] == trend_filter]
        
        if signal_filter == "Com OB":
            filtered_df = filtered_df[filtered_df['OB'] != ""]
        elif signal_filter == "Com OS":
            filtered_df = filtered_df[filtered_df['OS'] != ""]
        elif signal_filter == "Com Cruzamentos":
            filtered_df = filtered_df[(filtered_df['Bull Cross'] != "") | (filtered_df['Bear Cross'] != "")]
        
        # Exibir tabela
        st.subheader(f"📊 Resultados ({len(filtered_df)} moedas)")
        display_df = filtered_df.drop(columns=['raw_data'], errors='ignore')
        st.dataframe(display_df, width="stretch", height=400)

# Análise individual
if st.session_state.get('individual_analysis', False):
    st.markdown("---")
    st.subheader("🔍 Análise Detalhada Individual")
    
    # Botão para fechar a análise individual
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("❌ Fechar Análise"):
            st.session_state['individual_analysis'] = False
            st.rerun()
    
    if 'resultados' in st.session_state:
        available_coins = [r['Moeda'] for r in st.session_state['resultados'] if 'Erro' not in r]
        selected_coin = st.selectbox("Escolha uma moeda para análise detalhada:", available_coins)
        
        if selected_coin:
            # Encontrar dados da moeda selecionada
            coin_data = None
            for r in st.session_state['resultados']:
                if r['Moeda'] == selected_coin and 'raw_data' in r:
                    coin_data = r
                    break
            
            if coin_data and coin_data['raw_data'] is not None:
                df = coin_data['raw_data']
                chart = create_wavetrend_chart(df, selected_coin)
                st.plotly_chart(chart, width="stretch")
                
                # Informações detalhadas
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Preço Atual", coin_data['Preço'])
                    st.metric("WT Atual", coin_data['WT'])
                with col2:
                    st.metric("Variação", coin_data['Variação (%)'])
                    st.metric("WT Signal", coin_data['WT Signal'])
                with col3:
                    st.metric("Volume", coin_data['Volume'])
                    st.write("**Status:**", coin_data['Tendência'])

# Top sinais
if st.session_state.get('top_signals', False):
    st.markdown("---")
    st.subheader("🎯 Top Sinais de Trading")
    
    if 'resultados' in st.session_state:
        # Sinais de compra (OS)
        buy_signals = [r for r in st.session_state['resultados'] if 'Erro' not in r and r.get('OS') == "✅"]
        # Sinais de venda (OB)
        sell_signals = [r for r in st.session_state['resultados'] if 'Erro' not in r and r.get('OB') == "✅"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 🟢 Sinais de Compra (OS)")
            if buy_signals:
                for signal in buy_signals[:10]:  # Top 10
                    st.write(f"**{signal['Moeda']}** - {signal['Preço']} | WT: {signal['WT']} | {signal['Força']}")
            else:
                st.write("Nenhum sinal de compra no momento")
        
        with col2:
            st.write("### 🔴 Sinais de Venda (OB)")
            if sell_signals:
                for signal in sell_signals[:10]:  # Top 10
                    st.write(f"**{signal['Moeda']}** - {signal['Preço']} | WT: {signal['WT']} | {signal['Força']}")
            else:
                st.write("Nenhum sinal de venda no momento")
    
    st.session_state['top_signals'] = False

# Informações sobre sinais
with st.expander("ℹ️ Informações sobre os Sinais"):
    st.markdown("""
    ### 📊 Interpretação dos Sinais WaveTrend
    
    **Sinais Principais:**
    - **OB (✅)**: Sinal de sobrecompra - possível reversão para baixa
    - **OS (✅)**: Sinal de sobrevenda - possível reversão para alta
    
    **Zonas:**
    - **🟣 Sobrecompra**: WT > threshold (cuidado com reversão)
    - **🔵 Sobrevenda**: WT < -threshold (oportunidade de compra)
    
    **Cruzamentos:**
    - **⬆️ Bull Cross**: WT cruza acima do sinal (tendência de alta)
    - **⬇️ Bear Cross**: WT cruza abaixo do sinal (tendência de baixa)
    
    **Força do Sinal:**
    - **🔴/🟢 FORTE**: Sinal muito confiável
    - **🟡 MODERADO**: Sinal com confiabilidade média
    - **⚪ FRACO**: Sinal com baixa confiabilidade
    """)

# Rodapé
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
        📈 WaveTrend Oscillator Pro | Desenvolvido para análise técnica avançada<br>
        ⚠️ Este app é apenas para fins educacionais. Sempre faça sua própria pesquisa antes de investir.
    </div>
    """, 
    unsafe_allow_html=True
)

# Inicialização do estado
if 'update_data' not in st.session_state:
    st.session_state['update_data'] = False
