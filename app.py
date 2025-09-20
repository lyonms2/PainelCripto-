import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import warnings
import requests
import json
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="WaveTrend Oscillator Pro + Telegram",
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
    .telegram-config {
        background-color: #0088cc;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 WaveTrend Oscillator Pro + 🤖 Telegram Notifications")
st.markdown("---")

# Sidebar com configurações
st.sidebar.header("⚙️ Configurações")

# Configurações do Telegram
st.sidebar.subheader("🤖 Configurações Telegram")
telegram_bot_token = st.sidebar.text_input("Bot Token", type="password", help="Token do seu bot do Telegram")
telegram_chat_id = st.sidebar.text_input("Chat ID", help="Seu chat ID ou ID do grupo")

# Configurações de notificação
st.sidebar.subheader("🔔 Configurações de Notificação")
notify_on_signals = st.sidebar.checkbox("Notificar Sinais OB/OS", value=True)
notify_on_crosses = st.sidebar.checkbox("Notificar Cruzamentos", value=True)
notify_strong_only = st.sidebar.checkbox("Apenas Sinais Fortes", value=False)
notification_cooldown = st.sidebar.slider("Intervalo entre notificações (min)", 5, 60, 15)

# Teste de conexão Telegram
if st.sidebar.button("🧪 Testar Telegram"):
    if telegram_bot_token and telegram_chat_id:
        test_result = send_telegram_message(
            telegram_bot_token, 
            telegram_chat_id, 
            "🧪 Teste de conexão bem-sucedido!\nSeu bot está funcionando corretamente! 🎉"
        )
        if test_result:
            st.sidebar.success("✅ Telegram conectado!")
        else:
            st.sidebar.error("❌ Erro na conexão")
    else:
        st.sidebar.warning("⚠️ Preencha Bot Token e Chat ID")

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

# Auto-refresh com notificações
auto_refresh = st.sidebar.checkbox("Auto Refresh + Notificações", value=False)
if auto_refresh:
    refresh_interval = st.sidebar.slider("Intervalo de refresh (segundos)", 30, 300, 60)
    st.sidebar.write(f"🔄 Próxima atualização em {refresh_interval}s")

# Funções auxiliares (definidas antes de serem usadas)
def send_telegram_message(bot_token, chat_id, message):
    """Envia mensagem via Telegram"""
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'HTML'
        }
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Erro ao enviar Telegram: {e}")
        return False

def format_telegram_alert(symbol, signal_type, data, timeframe):
    """Formata mensagem de alerta para Telegram"""
    emojis = {
        'OB': '🔴',
        'OS': '🟢',
        'BULL_CROSS': '⬆️',
        'BEAR_CROSS': '⬇️'
    }
    
    emoji = emojis.get(signal_type, '📊')
    
    if signal_type == 'OB':
        signal_name = "SINAL DE VENDA (Sobrecompra)"
    elif signal_type == 'OS':
        signal_name = "SINAL DE COMPRA (Sobrevenda)"
    elif signal_type == 'BULL_CROSS':
        signal_name = "CRUZAMENTO DE ALTA"
    elif signal_type == 'BEAR_CROSS':
        signal_name = "CRUZAMENTO DE BAIXA"
    else:
        signal_name = "SINAL DETECTADO"
    
    message = f"""
{emoji} <b>{signal_name}</b>

💰 <b>Moeda:</b> {symbol}
💵 <b>Preço:</b> {data['price']}
📈 <b>Variação:</b> {data['change']:+.2f}%
📊 <b>Timeframe:</b> {timeframe}

📉 <b>WT:</b> {data['wt']:.2f}
📉 <b>WT Signal:</b> {data['wt_signal']:.2f}
💪 <b>Força:</b> {data['strength']}

🕒 <b>Horário:</b> {datetime.now().strftime('%H:%M:%S')}

⚠️ <i>Sempre faça sua própria análise antes de operar!</i>
    """.strip()
    
# Símbolos principais
SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'HYPE/USDT', 'PUMP/USDT', 'ENA/USDT', 
    'FARTCOIN/USDT', 'BONK/USDT', 'BNB/USDT', 'ADA/USDT', 'XRP/USDT', 'DOGE/USDT',
    'TRX/USDT', 'LINK/USDT', 'LTC/USDT','PENGU/USDT', 'DOT/USDT', 'BCH/USDT', 
    'SHIB/USDT', 'AVAX/USDT', 'OP/USDT', 'UNI/USDT', 'ATOM/USDT', 'ETC/USDT', 
    'XLM/USDT', 'FIL/USDT', 'APT/USDT', 'SUI/USDT', 'HBAR/USDT', 'ZORA/USDT', 
    'AR/USDT', 'INJ/USDT', 'PEPE/USDT', 'NEAR/USDT', 'STX/USDT', 'ALGO/USDT', 
    'IMX/USDT', 'WIF/USDT', 'MINA/USDT', 'DYDX/USDT', 'TIA/USDT', 'JTO/USDT', 
    'AAVE/USDT', 'PYTH/USDT', 'SAND/USDT', 'CAKE/USDT', 'BLUR/USDT', 
    'GMX/USDT', 'LDO/USDT', 'FET/USDT', 'DYM/USDT', 'GMT/USDT', 'MEME/USDT', 
    'BOME/USDT', 'YGG/USDT', 'RUNE/USDT', 'CELO/USDT', 'WLD/USDT', 'ONDO/USDT', 
    'SEI/USDT', 'JUP/USDT', 'POPCAT/USDT', 'TAO/USDT', 'TON/USDT'
]
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

def is_strong_signal(strength):
    """Verifica se é um sinal forte"""
    return "FORTE" in strength

def should_notify(symbol, signal_type, last_notifications, cooldown_minutes):
    """Verifica se deve notificar baseado no cooldown"""
    key = f"{symbol}_{signal_type}"
    now = datetime.now()
    
    if key in last_notifications:
        time_diff = (now - last_notifications[key]).total_seconds() / 60
        if time_diff < cooldown_minutes:
            return False
    
    last_notifications[key] = now
    return True

def fetch_data_and_analyze(symbols, timeframe, params, telegram_config=None):
    """Busca dados e realiza análise com notificações Telegram"""
    exchange = ccxt.kucoin()
    resultados = []
    notifications_sent = 0
    
    # Inicializa sistema de notificações
    if 'last_notifications' not in st.session_state:
        st.session_state['last_notifications'] = {}
    
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
            
            # Dados para notificação
            alert_data = {
                'price': f"${current['close']:,.4f}",
                'change': price_change,
                'wt': current['WT'],
                'wt_signal': current['WT_signal'],
                'strength': signal_strength
            }
            
            # Verificar e enviar notificações Telegram
            if telegram_config and telegram_config['enabled']:
                notifications_to_send = []
                
                # Verificar sinais OB/OS
                if telegram_config['notify_signals']:
                    if current['OB'] and should_notify(symbol, 'OB', st.session_state['last_notifications'], telegram_config['cooldown']):
                        if not telegram_config['strong_only'] or is_strong_signal(signal_strength):
                            notifications_to_send.append(('OB', alert_data))
                    
                    if current['OS'] and should_notify(symbol, 'OS', st.session_state['last_notifications'], telegram_config['cooldown']):
                        if not telegram_config['strong_only'] or is_strong_signal(signal_strength):
                            notifications_to_send.append(('OS', alert_data))
                
                # Verificar cruzamentos
                if telegram_config['notify_crosses']:
                    if current['Bullish_Cross'] and should_notify(symbol, 'BULL_CROSS', st.session_state['last_notifications'], telegram_config['cooldown']):
                        if not telegram_config['strong_only'] or is_strong_signal(signal_strength):
                            notifications_to_send.append(('BULL_CROSS', alert_data))
                    
                    if current['Bearish_Cross'] and should_notify(symbol, 'BEAR_CROSS', st.session_state['last_notifications'], telegram_config['cooldown']):
                        if not telegram_config['strong_only'] or is_strong_signal(signal_strength):
                            notifications_to_send.append(('BEAR_CROSS', alert_data))
                
                # Enviar notificações
                for signal_type, data in notifications_to_send:
                    message = format_telegram_alert(symbol, signal_type, data, timeframe)
                    if send_telegram_message(telegram_config['token'], telegram_config['chat_id'], message):
                        notifications_sent += 1
            
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
    
    return resultados, notifications_sent

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

# Status das notificações
if telegram_bot_token and telegram_chat_id:
    st.info("🤖 Telegram configurado! Notificações ativadas.")
else:
    st.warning("⚠️ Configure o Telegram na sidebar para receber notificações.")

# Auto-refresh logic
if auto_refresh and 'last_update' not in st.session_state:
    st.session_state['last_update'] = time.time()
    st.session_state['update_data'] = True

if auto_refresh and time.time() - st.session_state.get('last_update', 0) > refresh_interval:
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

# Configurações Telegram
telegram_config = None
if telegram_bot_token and telegram_chat_id:
    telegram_config = {
        'enabled': True,
        'token': telegram_bot_token,
        'chat_id': telegram_chat_id,
        'notify_signals': notify_on_signals,
        'notify_crosses': notify_on_crosses,
        'strong_only': notify_strong_only,
        'cooldown': notification_cooldown
    }

# Análise principal
if st.session_state.get('update_data', False):
    with st.spinner('🔄 Analisando mercado e enviando notificações...'):
        resultados, notifications_sent = fetch_data_and_analyze(SYMBOLS, selected_timeframe, params, telegram_config)
    
    st.session_state['resultados'] = resultados
    st.session_state['update_data'] = False
    
    success_msg = "✅ Análise finalizada!"
    if notifications_sent > 0:
        success_msg += f" 📱 {notifications_sent} notificações enviadas!"
    st.success(success_msg)

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

# Seção de configuração do Telegram com instruções
st.markdown("---")
with st.expander("🤖 Como Configurar o Telegram Bot"):
    st.markdown("""
    ### 📝 Passo a Passo para Configurar Notificações Telegram
    
    #### 1. Criar um Bot Telegram:
    1. Abra o Telegram e procure por **@BotFather**
    2. Envie `/start` e depois `/newbot`
    3. Escolha um nome e username para seu bot
    4. Copie o **Bot Token** que aparecerá (ex: `123456789:ABCdefGHIjklMNOpqrSTUvwxYZ`)
    
    #### 2. Obter seu Chat ID:
    1. Envie uma mensagem para seu bot
    2. Visite: `https://api.telegram.org/bot<SEU_BOT_TOKEN>/getUpdates`
    3. Procure pelo número do **"chat":{"id":XXXXXX}**
    4. Use esse número como Chat ID
    
    #### 3. Configurações Recomendadas:
    - ✅ **Notificar Sinais OB/OS**: Para alertas de sobrecompra/sobrevenda
    - ✅ **Notificar Cruzamentos**: Para mudanças de tendência
    - ⚠️ **Apenas Sinais Fortes**: Para reduzir spam de notificações
    - 🕐 **Intervalo**: 15-30 minutos para evitar muitas mensagens
    
    #### 4. Exemplo de Mensagem:
    ```
    🟢 SINAL DE COMPRA (Sobrevenda)
    
    💰 Moeda: BTC
    💵 Preço: $45,234.56
    📈 Variação: -2.34%
    📊 Timeframe: 4h
    
    📉 WT: -120.45
    📉 WT Signal: -108.32
    💪 Força: 🟢 FORTE
    
    🕒 Horário: 14:32:15
    ```
    """)

# Histórico de notificações
if 'last_notifications' in st.session_state and st.session_state['last_notifications']:
    with st.expander("📱 Histórico de Notificações Recentes"):
        st.write("### 🕐 Últimas Notificações Enviadas:")
        for key, timestamp in sorted(st.session_state['last_notifications'].items(), 
                                   key=lambda x: x[1], reverse=True)[:20]:
            symbol, signal_type = key.split('_')
            signal_names = {
                'OB': '🔴 Sobrecompra',
                'OS': '🟢 Sobrevenda', 
                'BULL_CROSS': '⬆️ Cruzamento Alta',
                'BEAR_CROSS': '⬇️ Cruzamento Baixa'
            }
            signal_name = signal_names.get(signal_type, signal_type)
            time_ago = datetime.now() - timestamp
            st.write(f"**{symbol}** - {signal_name} - {timestamp.strftime('%H:%M:%S')} ({int(time_ago.total_seconds()/60)}min atrás)")

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
    
    ### 🔔 Sistema de Notificações:
    - **Cooldown**: Evita spam de notificações repetidas
    - **Filtro de Força**: Opção de receber apenas sinais fortes
    - **Múltiplos Timeframes**: Configure diferentes bots para diferentes timeframes
    - **Histórico**: Acompanhe todas as notificações enviadas
    """)

# Sistema de alertas personalizados
st.markdown("---")
with st.expander("⚙️ Alertas Personalizados Avançados"):
    st.subheader("🎛️ Configurações Avançadas de Alertas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Filtros por Moeda:**")
        priority_coins = st.multiselect(
            "Moedas Prioritárias (recebem todos os alertas)",
            options=[s.replace('/USDT', '') for s in SYMBOLS],
            default=['BTC', 'ETH', 'SOL']
        )
        
        st.write("**Filtros por Força:**")
        min_wt_threshold = st.slider("WT Mínimo para Alertas", 50, 200, 100)
        
    with col2:
        st.write("**Horários de Funcionamento:**")
        alert_start_time = st.time_input("Início dos Alertas", value=datetime.now().time().replace(hour=9, minute=0))
        alert_end_time = st.time_input("Fim dos Alertas", value=datetime.now().time().replace(hour=22, minute=0))
        
        st.write("**Dias da Semana:**")
        alert_days = st.multiselect(
            "Dias para Alertas",
            options=['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo'],
            default=['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta']
        )

    # Salvar configurações personalizadas
    if st.button("💾 Salvar Configurações Personalizadas"):
        custom_config = {
            'priority_coins': priority_coins,
            'min_wt_threshold': min_wt_threshold,
            'alert_start_time': str(alert_start_time),
            'alert_end_time': str(alert_end_time),
            'alert_days': alert_days
        }
        st.session_state['custom_alert_config'] = custom_config
        st.success("✅ Configurações salvas!")

# Monitor de performance dos sinais
if 'resultados' in st.session_state:
    with st.expander("📈 Monitor de Performance dos Sinais"):
        resultados = st.session_state['resultados']
        valid_results = [r for r in resultados if 'Erro' not in r]
        
        if valid_results:
            # Estatísticas gerais
            total_ob = len([r for r in valid_results if r.get('OB') == "✅"])
            total_os = len([r for r in valid_results if r.get('OS') == "✅"])
            total_bull_cross = len([r for r in valid_results if r.get('Bull Cross') == "⬆️"])
            total_bear_cross = len([r for r in valid_results if r.get('Bear Cross') == "⬇️"])
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Sinais OB Ativos", total_ob)
            col2.metric("Sinais OS Ativos", total_os)
            col3.metric("Bull Crosses", total_bull_cross)
            col4.metric("Bear Crosses", total_bear_cross)
            
            # Top moedas por categoria
            st.write("### 🏆 Top Moedas por Categoria:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🔥 Mais Voláteis (Variação %):**")
                try:
                    volatile_coins = sorted(valid_results, 
                                          key=lambda x: abs(float(x['Variação (%)'].replace('%', '').replace('+', ''))), 
                                          reverse=True)[:5]
                    for coin in volatile_coins:
                        st.write(f"• **{coin['Moeda']}**: {coin['Variação (%)']} | {coin['Preço']}")
                except:
                    st.write("Dados não disponíveis")
            
            with col2:
                st.write("**💪 Sinais Mais Fortes:**")
                strong_signals = [r for r in valid_results if 'FORTE' in r.get('Força', '')][:5]
                for signal in strong_signals:
                    st.write(f"• **{signal['Moeda']}**: {signal['Força']} | WT: {signal['WT']}")

# Rodapé com informações de contato e suporte
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🛠️ Suporte Técnico
    - Problemas com Telegram? Verifique o token e chat ID
    - Sinais não aparecem? Ajuste o threshold
    - App lento? Reduza o número de moedas
    """)

with col2:
    st.markdown("""
    ### 📊 Recursos
    - Auto-refresh com notificações
    - Múltiplos timeframes
    - Filtros avançados
    - Histórico de sinais
    """)

with col3:
    st.markdown("""
    ### ⚠️ Disclaimer
    - Apenas para fins educacionais
    - Não é conselho financeiro
    - Sempre faça sua própria pesquisa
    - Gerencie riscos adequadamente
    """)

st.markdown(
    """
    <div style='text-align: center; color: #888; margin-top: 2rem;'>
        🤖📈 WaveTrend Oscillator Pro + Telegram Notifications v2.0<br>
        Desenvolvido para análise técnica avançada com alertas inteligentes<br>
        <small>⚡ Powered by Streamlit | 📡 KuCoin API | 🤖 Telegram Bot API</small>
    </div>
    """, 
    unsafe_allow_html=True
)

# Inicialização do estado
if 'update_data' not in st.session_state:
    st.session_state['update_data'] = False
if 'last_notifications' not in st.session_state:
    st.session_state['last_notifications'] = {}
