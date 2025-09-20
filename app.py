import streamlit as st
from streamlit_autorefresh import st_autorefresh
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

# ============================
# FUNÇÕES AUXILIARES
# ============================

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
        'BEAR_CROSS': '⬇️',
        'OVERBOUGHT_ZONE': '🟣',
        'OVERSOLD_ZONE': '🔵'
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
    elif signal_type == 'OVERBOUGHT_ZONE':
        signal_name = "ZONA DE SOBRECOMPRA"
    elif signal_type == 'OVERSOLD_ZONE':
        signal_name = "ZONA DE SOBREVENDA"
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
    
    return message

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

def is_strong_signal(strength):
    """Verifica se é um sinal forte"""
    return "FORTE" in strength

def should_notify(symbol, signal_type, last_notifications, cooldown_minutes, force_notify=False):
    """Verifica se deve notificar baseado no cooldown"""
    if force_notify:
        return True
        
    key = f"{symbol}_{signal_type}"
    now = datetime.now()
    
    if key in last_notifications:
        time_diff = (now - last_notifications[key]).total_seconds() / 60
        if time_diff < cooldown_minutes:
            return False
    
    last_notifications[key] = now
    return True

def fetch_data_and_analyze(symbols, timeframe, params, telegram_config=None, debug_mode=False, force_notify=False):
    """Busca dados e realiza análise com notificações Telegram"""
    exchange = ccxt.kucoin()
    resultados = []
    notifications_sent = 0
    debug_info = []
    
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
            
            # Debug info para esta moeda
            symbol_debug = {
                'symbol': symbol,
                'wt': current['WT'],
                'wt_signal': current['WT_signal'],
                'threshold': params['reversion_threshold'],
                'ob_signal': current['OB'],
                'os_signal': current['OS'],
                'overbought': current['Sobrecompra'],
                'oversold': current['Sobrevenda'],
                'prev_overbought': prev['Sobrecompra'],
                'prev_oversold': prev['Sobrevenda'],
                'bull_cross': current['Bullish_Cross'],
                'bear_cross': current['Bearish_Cross'],
                'strength': signal_strength
            }
            
            # Verificar e enviar notificações Telegram
            if telegram_config and telegram_config['enabled']:
                notifications_to_send = []
                
                # Verificar sinais OB/OS (cruzamentos específicos)
                if telegram_config['notify_signals']:
                    if current['OB']:
                        symbol_debug['ob_check'] = f"OB detectado: WT={current['WT']:.2f} > threshold={params['reversion_threshold']}"
                        if should_notify(symbol, 'OB', st.session_state['last_notifications'], telegram_config['cooldown'], force_notify):
                            if not telegram_config['strong_only'] or is_strong_signal(signal_strength):
                                notifications_to_send.append(('OB', alert_data))
                                symbol_debug['ob_will_notify'] = True
                            else:
                                symbol_debug['ob_blocked'] = "Bloqueado por filtro de força"
                        else:
                            symbol_debug['ob_blocked'] = "Bloqueado por cooldown"
                    
                    if current['OS']:
                        symbol_debug['os_check'] = f"OS detectado: WT={current['WT']:.2f} < -threshold={-params['reversion_threshold']}"
                        if should_notify(symbol, 'OS', st.session_state['last_notifications'], telegram_config['cooldown'], force_notify):
                            if not telegram_config['strong_only'] or is_strong_signal(signal_strength):
                                notifications_to_send.append(('OS', alert_data))
                                symbol_debug['os_will_notify'] = True
                            else:
                                symbol_debug['os_blocked'] = "Bloqueado por filtro de força"
                        else:
                            symbol_debug['os_blocked'] = "Bloqueado por cooldown"
                
                # Verificar zonas de sobrecompra/sobrevenda (entrada nas zonas)
                if telegram_config['notify_zones']:
                    # Verifica se entrou na zona de sobrecompra
                    if current['Sobrecompra'] and not prev['Sobrecompra']:
                        symbol_debug['overbought_zone'] = f"Entrada em zona sobrecompra: WT={current['WT']:.2f} > {params['reversion_threshold']}"
                        if should_notify(symbol, 'OVERBOUGHT_ZONE', st.session_state['last_notifications'], telegram_config['cooldown'], force_notify):
                            if not telegram_config['strong_only'] or is_strong_signal(signal_strength):
                                notifications_to_send.append(('OVERBOUGHT_ZONE', alert_data))
                                symbol_debug['overbought_will_notify'] = True
                            else:
                                symbol_debug['overbought_blocked'] = "Bloqueado por filtro de força"
                        else:
                            symbol_debug['overbought_blocked'] = "Bloqueado por cooldown"
                    elif current['Sobrecompra']:
                        symbol_debug['overbought_zone'] = "Já está em zona sobrecompra (sem notificação)"
                    
                    # Verifica se entrou na zona de sobrevenda
                    if current['Sobrevenda'] and not prev['Sobrevenda']:
                        symbol_debug['oversold_zone'] = f"Entrada em zona sobrevenda: WT={current['WT']:.2f} < {-params['reversion_threshold']}"
                        if should_notify(symbol, 'OVERSOLD_ZONE', st.session_state['last_notifications'], telegram_config['cooldown'], force_notify):
                            if not telegram_config['strong_only'] or is_strong_signal(signal_strength):
                                notifications_to_send.append(('OVERSOLD_ZONE', alert_data))
                                symbol_debug['oversold_will_notify'] = True
                            else:
                                symbol_debug['oversold_blocked'] = "Bloqueado por filtro de força"
                        else:
                            symbol_debug['oversold_blocked'] = "Bloqueado por cooldown"
                    elif current['Sobrevenda']:
                        symbol_debug['oversold_zone'] = "Já está em zona sobrevenda (sem notificação)"
                
                # Verificar cruzamentos
                if telegram_config['notify_crosses']:
                    if current['Bullish_Cross']:
                        symbol_debug['bull_cross_check'] = "Bull cross detectado"
                        if should_notify(symbol, 'BULL_CROSS', st.session_state['last_notifications'], telegram_config['cooldown'], force_notify):
                            if not telegram_config['strong_only'] or is_strong_signal(signal_strength):
                                notifications_to_send.append(('BULL_CROSS', alert_data))
                                symbol_debug['bull_cross_will_notify'] = True
                            else:
                                symbol_debug['bull_cross_blocked'] = "Bloqueado por filtro de força"
                        else:
                            symbol_debug['bull_cross_blocked'] = "Bloqueado por cooldown"
                    
                    if current['Bearish_Cross']:
                        symbol_debug['bear_cross_check'] = "Bear cross detectado"
                        if should_notify(symbol, 'BEAR_CROSS', st.session_state['last_notifications'], telegram_config['cooldown'], force_notify):
                            if not telegram_config['strong_only'] or is_strong_signal(signal_strength):
                                notifications_to_send.append(('BEAR_CROSS', alert_data))
                                symbol_debug['bear_cross_will_notify'] = True
                            else:
                                symbol_debug['bear_cross_blocked'] = "Bloqueado por filtro de força"
                        else:
                            symbol_debug['bear_cross_blocked'] = "Bloqueado por cooldown"
                
                symbol_debug['notifications_to_send'] = len(notifications_to_send)
                
                # Enviar notificações
                for signal_type, data in notifications_to_send:
                    message = format_telegram_alert(symbol, signal_type, data, timeframe)
                    if send_telegram_message(telegram_config['token'], telegram_config['chat_id'], message):
                        notifications_sent += 1
                        symbol_debug['sent_success'] = True
                    else:
                        symbol_debug['sent_failed'] = True
            
            if debug_mode and (any('check' in str(symbol_debug) or 'zone' in str(symbol_debug) for _ in [1])):
                debug_info.append(symbol_debug)
            
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
            if debug_mode:
                debug_info.append({'symbol': symbol, 'error': str(e)})
    
    return resultados, notifications_sent, debug_info if debug_mode else None

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
    fig.add_hline(y=100, line_dash="dash", line_color="red", row=2, col=1)  # Usando valor fixo por enquanto
    fig.add_hline(y=-100, line_dash="dash", line_color="green", row=2, col=1)
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

# ============================
# CONFIGURAÇÃO DA PÁGINA
# ============================

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

# ============================
# SIDEBAR COM CONFIGURAÇÕES
# ============================

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
notify_on_zones = st.sidebar.checkbox("Notificar Zonas (Sobrecompra/Sobrevenda)", value=True)
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

# Debug das notificações
st.sidebar.subheader("🔍 Debug Notificações")
debug_mode = st.sidebar.checkbox("Modo Debug", help="Mostra informações detalhadas sobre as notificações")
force_notify = st.sidebar.checkbox("Forçar Notificações", help="Ignora o cooldown para testes")

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
    st_autorefresh(interval=refresh_interval*1000, limit=None, key="refresh_counter")
    st.session_state['update_data'] = True

# ============================
# INTERFACE PRINCIPAL
# ============================

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
if auto_refresh:
    count = st_autorefresh(interval=refresh_interval*60000, limit=None, key="refresh_counter")
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
        'notify_zones': notify_on_zones,
        'strong_only': notify_strong_only,
        'cooldown': notification_cooldown
    }

# Análise principal
if st.session_state.get('update_data', False):
    with st.spinner('🔄 Analisando mercado e enviando notificações...'):
        if debug_mode:
            resultados, notifications_sent, debug_info = fetch_data_and_analyze(SYMBOLS, selected_timeframe, params, telegram_config, debug_mode, force_notify)
        else:
            result = fetch_data_and_analyze(SYMBOLS, selected_timeframe, params, telegram_config, debug_mode, force_notify)
            resultados, notifications_sent = result[0], result[1]
            debug_info = None
    
    st.session_state['resultados'] = resultados
    st.session_state['update_data'] = False
    
    success_msg = "✅ Análise finalizada!"
    if notifications_sent > 0:
        success_msg += f" 📱 {notifications_sent} notificações enviadas!"
    st.success(success_msg)
    
    # Mostrar informações de debug se ativado
    if debug_mode and debug_info:
        st.subheader("🔍 Informações de Debug")
        
        # Filtrar apenas moedas com sinais detectados
        relevant_debug = [d for d in debug_info if any(key.endswith('_check') or 'zone' in key or 'will_notify' in key or 'blocked' in key for key in d.keys())]
        
        if relevant_debug:
            for info in relevant_debug[:10]:  # Mostrar apenas as primeiras 10
                with st.expander(f"📊 Debug {info['symbol']} - WT: {info['wt']:.2f}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Valores Atuais:**")
                        st.write(f"• WT: {info['wt']:.2f}")
                        st.write(f"• WT Signal: {info['wt_signal']:.2f}")
                        st.write(f"• Threshold: {info['threshold']}")
                        st.write(f"• Força: {info['strength']}")
                        
                        st.write("**Estados:**")
                        st.write(f"• Sobrecompra: {'✅' if info['overbought'] else '❌'}")
                        st.write(f"• Sobrevenda: {'✅' if info['oversold'] else '❌'}")
                        st.write(f"• Prev Sobrecompra: {'✅' if info['prev_overbought'] else '❌'}")
                        st.write(f"• Prev Sobrevenda: {'✅' if info['prev_oversold'] else '❌'}")
                    
                    with col2:
                        st.write("**Detecções:**")
                        for key, value in info.items():
                            if 'check' in key or 'zone' in key:
                                st.write(f"• {key}: {value}")
                        
                        st.write("**Status Notificação:**")
                        for key, value in info.items():
                            if 'will_notify' in key or 'blocked' in key:
                                st.write(f"• {key}: {value}")
                        
                        if 'notifications_to_send' in info:
                            st.write(f"• **Total para enviar**: {info['notifications_to_send']}")
        else:
            st.info("🔍 Nenhum sinal detectado nesta análise.")
            
            # Mostrar algumas moedas para verificar os valores
            st.write("**Valores WT de algumas moedas para referência:**")
            for i, info in enumerate(debug_info[:5]):
                st.write(f"• **{info['symbol']}**: WT={info['wt']:.2f}, Threshold=±{info['threshold']}")
        
        # Configurações atuais
        st.write("### ⚙️ Configurações Atuais")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"• **Notificar OB/OS**: {'✅' if telegram_config and telegram_config.get('notify_signals') else '❌'}")
            st.write(f"• **Notificar Zonas**: {'✅' if telegram_config and telegram_config.get('notify_zones') else '❌'}")
            st.write(f"• **Notificar Cruzamentos**: {'✅' if telegram_config and telegram_config.get('notify_crosses') else '❌'}")
        with col2:
            st.write(f"• **Apenas Sinais Fortes**: {'✅' if telegram_config and telegram_config.get('strong_only') else '❌'}")
            st.write(f"• **Cooldown**: {telegram_config.get('cooldown', 0) if telegram_config else 0} min")
            st.write(f"• **Forçar Notificações**: {'✅' if force_notify else '❌'}")

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

# ============================
# SEÇÕES INFORMATIVAS E CONFIGURAÇÕES AVANÇADAS
# ============================

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
    - ✅ **Notificar Sinais OB/OS**: Para alertas de sobrecompra/sobrevenda com cruzamento
    - ✅ **Notificar Cruzamentos**: Para mudanças de tendência
    - ✅ **Notificar Zonas**: Para entrada em zonas de sobrecompra/sobrevenda
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
            try:
                # Split apenas no último underscore para lidar com símbolos que têm underscore
                parts = key.rsplit('_', 1)
                if len(parts) == 2:
                    symbol, signal_type = parts
                else:
                    symbol, signal_type = key, 'UNKNOWN'
                
                signal_names = {
                    'OB': '🔴 Sobrecompra',
                    'OS': '🟢 Sobrevenda', 
                    'BULL_CROSS': '⬆️ Cruzamento Alta',
                    'BEAR_CROSS': '⬇️ Cruzamento Baixa',
                    'OVERBOUGHT_ZONE': '🟣 Zona Sobrecompra',
                    'OVERSOLD_ZONE': '🔵 Zona Sobrevenda'
                }
                signal_name = signal_names.get(signal_type, signal_type)
                time_ago = datetime.now() - timestamp
                st.write(f"**{symbol}** - {signal_name} - {timestamp.strftime('%H:%M:%S')} ({int(time_ago.total_seconds()/60)}min atrás)")
            except Exception as e:
                # Se houver erro, exibe a chave original
                st.write(f"**{key}** - {timestamp.strftime('%H:%M:%S')}")

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

# Painel de controle avançado
st.markdown("---")
with st.expander("🎛️ Painel de Controle Avançado"):
    st.subheader("📊 Estatísticas Detalhadas")
    
    if 'resultados' in st.session_state:
        valid_results = [r for r in st.session_state['resultados'] if 'Erro' not in r]
        
        if valid_results:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("#### 📈 Distribuição de Tendências")
                bullish_pct = len([r for r in valid_results if '📈' in r.get('Tendência', '')]) / len(valid_results) * 100
                bearish_pct = 100 - bullish_pct
                st.write(f"• **Bullish**: {bullish_pct:.1f}%")
                st.write(f"• **Bearish**: {bearish_pct:.1f}%")
                
            with col2:
                st.write("#### 💪 Distribuição de Força")
                forte_count = len([r for r in valid_results if 'FORTE' in r.get('Força', '')])
                moderado_count = len([r for r in valid_results if 'MODERADO' in r.get('Força', '')])
                fraco_count = len([r for r in valid_results if 'FRACO' in r.get('Força', '')])
                total = len(valid_results)
                st.write(f"• **Forte**: {forte_count} ({forte_count/total*100:.1f}%)")
                st.write(f"• **Moderado**: {moderado_count} ({moderado_count/total*100:.1f}%)")
                st.write(f"• **Fraco**: {fraco_count} ({fraco_count/total*100:.1f}%)")
                
            with col3:
                st.write("#### 🎯 Sinais Ativos")
                st.write(f"• **OB**: {len([r for r in valid_results if r.get('OB') == '✅'])}")
                st.write(f"• **OS**: {len([r for r in valid_results if r.get('OS') == '✅'])}")
                st.write(f"• **Bull Cross**: {len([r for r in valid_results if r.get('Bull Cross') == '⬆️'])}")
                st.write(f"• **Bear Cross**: {len([r for r in valid_results if r.get('Bear Cross') == '⬇️'])}")

# Seção de backup e restauração
with st.expander("💾 Backup e Configurações"):
    st.subheader("📁 Gerenciamento de Configurações")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### 💾 Backup das Configurações")
        if st.button("📤 Exportar Configurações"):
            config_backup = {
                'telegram_settings': {
                    'notify_signals': notify_on_signals,
                    'notify_crosses': notify_on_crosses,
                    'strong_only': notify_strong_only,
                    'cooldown': notification_cooldown
                },
                'wavetrend_params': {
                    'channel_length': channel_length,
                    'average_length': average_length,
                    'signal_length': signal_length,
                    'threshold': reversion_threshold,
                    'timeframe': selected_timeframe,
                    'price_source': price_source
                }
            }
            st.json(config_backup)
            st.success("✅ Configurações exportadas!")
    
    with col2:
        st.write("#### 🗑️ Limpeza de Dados")
        if st.button("🧹 Limpar Histórico de Notificações"):
            if 'last_notifications' in st.session_state:
                st.session_state['last_notifications'] = {}
            st.success("✅ Histórico limpo!")
        
        if st.button("🔄 Reset Completo"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("✅ App resetado!")

# ============================
# RODAPÉ
# ============================

# Rodapé com informações de contato e suporte
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🛠️ Suporte Técnico
    - Problemas com Telegram? Verifique o token e chat ID
    - Sinais não aparecem? Ajuste o threshold
    - App lento? Reduza o número de moedas
    - Use o teste de conexão para verificar o Telegram
    """)

with col2:
    st.markdown("""
    ### 📊 Recursos Disponíveis
    - Auto-refresh com notificações inteligentes
    - Múltiplos timeframes e fontes de preço
    - Filtros avançados por tendência e sinais
    - Histórico completo de notificações
    - Sistema anti-spam com cooldown
    """)

with col3:
    st.markdown("""
    ### ⚠️ Avisos Importantes
    - Este app é apenas para fins educacionais
    - Não constitui conselho financeiro
    - Sempre faça sua própria pesquisa (DYOR)
    - Gerencie seus riscos adequadamente
    - Nunca invista mais do que pode perder
    """)

# Status do sistema
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🔄 Status", "Online", delta="Funcionando")

with col2:
    telegram_status = "Conectado" if telegram_bot_token and telegram_chat_id else "Desconectado"
    st.metric("🤖 Telegram", telegram_status)

with col3:
    if 'last_update' in st.session_state:
        last_update = datetime.fromtimestamp(st.session_state['last_update'])
        st.metric("⏰ Última Análise", last_update.strftime("%H:%M:%S"))
    else:
        st.metric("⏰ Última Análise", "Nunca")

with col4:
    notifications_count = len(st.session_state.get('last_notifications', {}))
    st.metric("📱 Total Alertas", notifications_count)

st.markdown(
    """
    <div style='text-align: center; color: #888; margin-top: 2rem; padding: 1rem; border-top: 1px solid #333;'>
        🤖📈 <b>WaveTrend Oscillator Pro + Telegram Notifications v2.0</b><br>
        Desenvolvido para análise técnica avançada com sistema de alertas inteligentes<br>
        <small>⚡ Powered by Streamlit | 📡 KuCoin API | 🤖 Telegram Bot API | 📊 Plotly Charts</small><br><br>
        <i>💡 Dica: Configure o auto-refresh para monitoramento contínuo com notificações automáticas!</i>
    </div>
    """, 
    unsafe_allow_html=True
)

# ============================
# INICIALIZAÇÃO DO ESTADO
# ============================

# Inicialização do estado (sempre no final)
if 'update_data' not in st.session_state:
    st.session_state['update_data'] = False
if 'last_notifications' not in st.session_state:
    st.session_state['last_notifications'] = {}
if 'individual_analysis' not in st.session_state:
    st.session_state['individual_analysis'] = False
if 'top_signals' not in st.session_state:
    st.session_state['top_signals'] = False
if 'full_analysis' not in st.session_state:
    st.session_state['full_analysis'] = False
