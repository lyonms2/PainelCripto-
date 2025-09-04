import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import time
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')

# Configurar página
st.set_page_config(
    page_title="Hull VWAP Indicator - KuCoin",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class KucoinDataFetcher:
    """Fetcher de dados da KuCoin API pública"""
    
    def __init__(self):
        self.base_url = "https://api.kucoin.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    @st.cache_data(ttl=300)
    def get_popular_symbols(_self) -> List[str]:
        popular = [
            'BTC-USDT', 'ETH-USDT', 'BNB-USDT', 'ADA-USDT', 'SOL-USDT',
            'DOT-USDT', 'LINK-USDT', 'MATIC-USDT', 'AVAX-USDT', 'UNI-USDT',
            'ATOM-USDT', 'FTM-USDT', 'NEAR-USDT', 'ALGO-USDT', 'XRP-USDT',
            'LTC-USDT', 'BCH-USDT', 'ETC-USDT', 'XLM-USDT', 'TRX-USDT'
        ]
        return popular
    
    def get_klines(self, symbol: str, type_: str = '1day', start_time: Optional[int] = None, end_time: Optional[int] = None) -> pd.DataFrame:
        try:
            url = f"{self.base_url}/api/v1/market/candles"
            params = {'symbol': symbol, 'type': type_}
            
            if start_time:
                params['startAt'] = start_time
            if end_time:
                params['endAt'] = end_time
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            if data.get('code') != '200000':
                raise Exception(f"API Error: {data}")
            
            klines = data.get('data', [])
            if not klines:
                raise Exception(f"Nenhum dado encontrado para {symbol}")
            
            df = pd.DataFrame(klines, columns=['time', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
            df['time'] = pd.to_datetime(df['time'].astype(int), unit='s')
            
            for col in ['open', 'close', 'high', 'low', 'volume', 'turnover']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.sort_values('time').reset_index(drop=True)
            df.set_index('time', inplace=True)
            df = df.dropna()
            
            return df
            
        except Exception as e:
            raise Exception(f"Erro ao buscar dados de {symbol}: {e}")
    
    @st.cache_data(ttl=60)
    def get_market_data(_self, symbol: str, timeframe: str = '1day', days: int = 365) -> pd.DataFrame:
        try:
            end_time = int(datetime.now().timestamp())
            start_time = int((datetime.now() - timedelta(days=days)).timestamp())
            
            df = _self.get_klines(symbol=symbol, type_=timeframe, start_time=start_time, end_time=end_time)
            
            if df.empty:
                raise Exception("Nenhum dado retornado")
            
            df_formatted = pd.DataFrame({
                'high': df['high'],
                'low': df['low'],
                'close': df['close'],
                'open': df['open'],
                'volume': df['volume']
            }, index=df.index)
            
            return df_formatted
            
        except Exception as e:
            raise Exception(f"Erro em get_market_data: {e}")

class DynamicVWAP:
    """Dynamic Swing Anchored VWAP - Implementação exata do Pine Script"""
    
    def __init__(self, swing_period: int = 50, base_apt: float = 20.0, use_adapt: bool = False, vol_bias: float = 10.0):
        self.swing_period = swing_period
        self.base_apt = base_apt
        self.use_adapt = use_adapt
        self.vol_bias = vol_bias
        
        # State variables (como no Pine Script)
        self.ph = np.nan
        self.pl = np.nan
        self.phL = 0
        self.plL = 0
        self.prev = np.nan
        self.p = 0.0
        self.vol = 0.0
        self.direction = 0
        self.vwap_points = []

    def calculate(self, df: pd.DataFrame) -> dict:
        """Calcula VWAP seguindo exatamente a lógica do Pine Script"""
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        hlc3 = (df['high'] + df['low'] + df['close']) / 3
        
        n = len(df)
        vwap_values = np.full(n, np.nan)
        pivot_highs = np.full(n, np.nan)
        pivot_lows = np.full(n, np.nan)
        directions = np.full(n, 0)
        
        # ATR para adaptação
        atr_len = 50
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]  # Fix first value
        
        atr = pd.Series(tr).rolling(window=atr_len, min_periods=1).mean().values
        atr_avg = pd.Series(atr).rolling(window=atr_len, min_periods=1).mean().values
        
        ratio = np.where(atr_avg > 0, atr / atr_avg, 1.0)
        
        # Calcular APT series
        if self.use_adapt:
            apt_raw = self.base_apt / np.power(ratio, self.vol_bias)
            apt_series = np.clip(apt_raw, 5.0, 300.0).round()
        else:
            apt_series = np.full(n, self.base_apt)
        
        # Função alpha do Pine Script
        def alpha_from_apt(apt):
            apt = max(1.0, apt)
            decay = np.exp(-np.log(2.0) / apt)
            return 1.0 - decay
        
        # Loop principal (seguindo exatamente o Pine Script)
        for i in range(n):
            # Detectar pivots (ta.highestbars e ta.lowestbars)
            if i >= self.swing_period:
                # Check for pivot high
                start_idx = max(0, i - self.swing_period)
                end_idx = min(n, i + 1)
                window_high = high[start_idx:end_idx]
                if len(window_high) > 0 and high[i] == np.max(window_high):
                    self.ph = high[i]
                    self.phL = i
                    pivot_highs[i] = high[i]
                
                # Check for pivot low
                window_low = low[start_idx:end_idx]
                if len(window_low) > 0 and low[i] == np.min(window_low):
                    self.pl = low[i]
                    self.plL = i
                    pivot_lows[i] = low[i]
            
            # Direção (como no Pine Script: phL > plL ? 1 : -1)
            current_dir = 1 if self.phL > self.plL else -1
            directions[i] = current_dir
            
            # Lógica principal do Pine Script
            if current_dir != self.direction and i > 0:  # dir != dir[1]
                # Novo swing detectado
                self.direction = current_dir
                
                # Definir x, y (ponto do swing)
                x = self.plL if current_dir > 0 else self.phL
                y = self.pl if current_dir > 0 else self.ph
                
                # Calcular txt para label
                if current_dir > 0:  # Swing low
                    txt = 'LL' if self.pl < self.prev else 'HL'
                    self.prev = self.ph if i > 0 else self.ph
                else:  # Swing high
                    txt = 'LH' if self.ph < self.prev else 'HH'
                    self.prev = self.pl if i > 0 else self.pl
                
                # Reset VWAP (como no Pine Script)
                barsback = i - x
                if barsback < len(volume) and x < len(volume):
                    self.p = y * volume[max(0, min(x, len(volume)-1))]
                    self.vol = volume[max(0, min(x, len(volume)-1))]
                
                # Recalcular VWAP desde o swing
                for j in range(max(0, x), i + 1):
                    if j >= n:
                        break
                    
                    apt_j = apt_series[j]
                    alpha = alpha_from_apt(apt_j)
                    
                    pxv = hlc3.iloc[j] * volume[j]
                    v_j = volume[j]
                    
                    self.p = (1.0 - alpha) * self.p + alpha * pxv
                    self.vol = (1.0 - alpha) * self.vol + alpha * v_j
                
                vwap_values[i] = self.p / self.vol if self.vol > 0 else np.nan
                
            else:
                # Continuar VWAP existente (como no Pine Script - parte else)
                apt_0 = apt_series[i]
                alpha = alpha_from_apt(apt_0)
                
                pxv = hlc3.iloc[i] * volume[i]
                v0 = volume[i]
                
                self.p = (1.0 - alpha) * self.p + alpha * pxv
                self.vol = (1.0 - alpha) * self.vol + alpha * v0
                
                vwap_values[i] = self.p / self.vol if self.vol > 0 else np.nan
        
        return {
            'vwap': pd.Series(vwap_values, index=df.index),
            'pivot_highs': pd.Series(pivot_highs, index=df.index),
            'pivot_lows': pd.Series(pivot_lows, index=df.index),
            'directions': pd.Series(directions, index=df.index),
            'apt_series': pd.Series(apt_series, index=df.index)
        }

class HullMA:
    """Hull Moving Average implementations"""
    
    @staticmethod
    def wma(series: pd.Series, length: int) -> pd.Series:
        """Weighted Moving Average"""
        def calc_wma(x):
            if len(x) < length:
                return np.nan
            weights = np.arange(1, length + 1)
            return np.average(x.iloc[-length:], weights=weights)
        return series.rolling(window=length, min_periods=length).apply(calc_wma, raw=False)
    
    @staticmethod
    def ema(series: pd.Series, length: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=length, adjust=False).mean()
    
    @staticmethod
    def hma(series: pd.Series, length: int) -> pd.Series:
        """Hull Moving Average"""
        half_length = max(1, int(length / 2))
        sqrt_length = max(1, int(np.sqrt(length)))
        
        wma_half = HullMA.wma(series, half_length)
        wma_full = HullMA.wma(series, length)
        raw_hma = 2 * wma_half - wma_full
        
        return HullMA.wma(raw_hma, sqrt_length)
    
    @staticmethod
    def ehma(series: pd.Series, length: int) -> pd.Series:
        """Exponential Hull Moving Average"""
        half_length = max(1, int(length / 2))
        sqrt_length = max(1, int(np.sqrt(length)))
        
        ema_half = HullMA.ema(series, half_length)
        ema_full = HullMA.ema(series, length)
        raw_ehma = 2 * ema_half - ema_full
        
        return HullMA.ema(raw_ehma, sqrt_length)
    
    @staticmethod
    def thma(series: pd.Series, length: int) -> pd.Series:
        """Triangular Hull Moving Average"""
        third_length = max(1, int(length / 3))
        half_length = max(1, int(length / 2))
        
        wma_third = HullMA.wma(series, third_length)
        wma_half = HullMA.wma(series, half_length)
        wma_full = HullMA.wma(series, length)
        
        raw_thma = wma_third * 3 - wma_half - wma_full
        return HullMA.wma(raw_thma, length)

def create_plotly_chart(df: pd.DataFrame, hull_ma: pd.Series, vwap_results: dict, symbol: str) -> go.Figure:
    """Create interactive Plotly chart"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=[f'{symbol} - Hull VWAP Analysis', 'Volume']
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ),
        row=1, col=1
    )
    
    # Hull MA
    if not hull_ma.isna().all():
        hull_trend = hull_ma > hull_ma.shift(1)
        hull_color = '#00ff00' if hull_trend.iloc[-1] else '#ff0000'
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=hull_ma,
                mode='lines',
                name='Hull MA',
                line=dict(color=hull_color, width=2)
            ),
            row=1, col=1
        )
    
    # VWAP
    if not vwap_results['vwap'].isna().all():
        # Colorir VWAP baseado na direção
        directions = vwap_results['directions']
        vwap_colors = ['#00ff88' if d > 0 else '#ff4444' for d in directions]
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=vwap_results['vwap'],
                mode='lines',
                name='Dynamic VWAP',
                line=dict(color='#0088ff', width=2)
            ),
            row=1, col=1
        )
    
    # Pivot points
    pivot_highs = vwap_results['pivot_highs'].dropna()
    pivot_lows = vwap_results['pivot_lows'].dropna()
    
    if len(pivot_highs) > 0:
        fig.add_trace(
            go.Scatter(
                x=pivot_highs.index,
                y=pivot_highs.values,
                mode='markers',
                name='Pivot Highs',
                marker=dict(symbol='triangle-down', size=12, color='red')
            ),
            row=1, col=1
        )
    
    if len(pivot_lows) > 0:
        fig.add_trace(
            go.Scatter(
                x=pivot_lows.index,
                y=pivot_lows.values,
                mode='markers',
                name='Pivot Lows',
                marker=dict(symbol='triangle-up', size=12, color='lime')
            ),
            row=1, col=1
        )
    
    # Volume
    colors = ['#00ff88' if close >= open else '#ff4444' for close, open in zip(df['close'], df['open'])]
    fig.add_trace(
        go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color=colors, opacity=0.6),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f'{symbol} - Dynamic Swing Anchored VWAP',
        xaxis_rangeslider_visible=False,
        height=800,
        template='plotly_dark'
    )
    
    return fig

def main():
    st.title("📈 Dynamic Swing Anchored VWAP - KuCoin")
    st.markdown("**Implementação exata do Pine Script do Zeiierman**")
    
    # Initialize fetcher
    if 'fetcher' not in st.session_state:
        st.session_state.fetcher = KucoinDataFetcher()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Data settings
        st.subheader("📊 Dados")
        symbols = st.session_state.fetcher.get_popular_symbols()
        symbol = st.selectbox("Symbol", symbols, index=0)
        
        timeframes = {
            '1min': '1min', '5min': '5min', '15min': '15min', 
            '30min': '30min', '1hour': '1hour', '4hour': '4hour',
            '1day': '1day', '1week': '1week'
        }
        timeframe = st.selectbox("Timeframe", list(timeframes.keys()), index=6)
        days = st.slider("Days Back", 7, 365, 90)
        
        # Hull settings
        st.subheader("🔄 Hull MA")
        hull_type = st.selectbox("Hull Type", ['HMA', 'EHMA', 'THMA'])
        hull_length = st.slider("Hull Length", 5, 200, 55)
        
        # VWAP settings
        st.subheader("📈 VWAP Settings")
        swing_period = st.slider("Swing Period", 5, 100, 50)
        base_apt = st.slider("Base APT", 5.0, 100.0, 20.0)
        use_adapt = st.checkbox("Use ATR Adaptation", False)
        
        if use_adapt:
            vol_bias = st.slider("Volatility Bias", 0.1, 20.0, 10.0)
        else:
            vol_bias = 10.0
        
        update_btn = st.button("🔄 Update Data", type="primary")
    
    # Main content
    try:
        with st.spinner(f'Loading {symbol} data...'):
            df = st.session_state.fetcher.get_market_data(
                symbol=symbol,
                timeframe=timeframes[timeframe],
                days=days
            )
        
        if df.empty:
            st.error("No data available")
            return
        
        # Calculate Hull MA
        src = df['close']
        if hull_type == 'HMA':
            hull_ma = HullMA.hma(src, hull_length)
        elif hull_type == 'EHMA':
            hull_ma = HullMA.ehma(src, hull_length)
        else:  # THMA
            hull_ma = HullMA.thma(src, hull_length)
        
        # Calculate VWAP
        vwap_calculator = DynamicVWAP(
            swing_period=swing_period,
            base_apt=base_apt,
            use_adapt=use_adapt,
            vol_bias=vol_bias
        )
        
        with st.spinner('Calculating Dynamic VWAP...'):
            vwap_results = vwap_calculator.calculate(df)
        
        # Create chart
        fig = create_plotly_chart(df, hull_ma, vwap_results, symbol)
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = df['close'].iloc[-1]
            price_change = ((current_price - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
            st.metric("Current Price", f"${current_price:.6f}", f"{price_change:.2f}%")
        
        with col2:
            current_vwap = vwap_results['vwap'].iloc[-1]
            if not np.isnan(current_vwap):
                st.metric("Current VWAP", f"${current_vwap:.6f}")
            else:
                st.metric("Current VWAP", "N/A")
        
        with col3:
            pivot_highs_count = vwap_results['pivot_highs'].count()
            pivot_lows_count = vwap_results['pivot_lows'].count()
            st.metric("Pivot Points", f"{pivot_highs_count + pivot_lows_count}")
        
        with col4:
            hull_trend = "🟢 Bullish" if hull_ma.iloc[-1] > hull_ma.iloc[-2] else "🔴 Bearish"
            st.metric("Hull Trend", hull_trend)
        
        # Data table
        with st.expander("📈 Raw Data"):
            display_df = df.copy()
            display_df['Hull_MA'] = hull_ma
            display_df['VWAP'] = vwap_results['vwap']
            display_df['Direction'] = vwap_results['directions']
            st.dataframe(display_df.tail(50))
        
        # Footer
        st.markdown("---")
        st.markdown("**Dynamic Swing Anchored VWAP** - Implementação fiel ao Pine Script | Dados: KuCoin API")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
        if st.button("📊 Load Sample Data"):
            # Sample data fallback
            dates = pd.date_range(start=datetime.now() - timedelta(days=90), end=datetime.now(), freq='1D')
            np.random.seed(42)
            
            price_base = 45000
            returns = np.random.normal(0, 0.02, len(dates))
            price_data = price_base * np.cumprod(1 + returns)
            
            sample_df = pd.DataFrame({
                'high': price_data * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
                'low': price_data * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
                'close': price_data,
                'open': np.roll(price_data, 1),
                'volume': np.random.lognormal(mean=15, sigma=0.5, size=len(dates))
            }, index=dates)
            
            hull_ma = HullMA.hma(sample_df['close'], hull_length)
            vwap_calc = DynamicVWAP(swing_period, base_apt, use_adapt, vol_bias)
            vwap_results = vwap_calc.calculate(sample_df)
            
            fig = create_plotly_chart(sample_df, hull_ma, vwap_results, "BTC-USDT (Sample)")
            st.plotly_chart(fig, use_container_width=True)
            st.success("Sample data loaded successfully!")

if __name__ == "__main__":
    main()
