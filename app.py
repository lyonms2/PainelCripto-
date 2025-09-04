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

# Classes do indicador (copiadas do código anterior)
class KucoinDataFetcher:
    """Fetcher de dados da KuCoin API pública"""
    
    def __init__(self):
        self.base_url = "https://api.kucoin.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    @st.cache_data(ttl=300)  # Cache por 5 minutos
    def get_popular_symbols(_self) -> List[str]:
        """Retorna símbolos populares para facilitar a seleção"""
        popular = [
            'BTC-USDT', 'ETH-USDT', 'BNB-USDT', 'ADA-USDT', 'SOL-USDT',
            'DOT-USDT', 'LINK-USDT', 'MATIC-USDT', 'AVAX-USDT', 'UNI-USDT',
            'ATOM-USDT', 'FTM-USDT', 'NEAR-USDT', 'ALGO-USDT', 'XRP-USDT',
            'LTC-USDT', 'BCH-USDT', 'ETC-USDT', 'XLM-USDT', 'TRX-USDT'
        ]
        return popular
    
    def get_klines(self, 
                   symbol: str, 
                   type_: str = '1day',
                   start_time: Optional[int] = None,
                   end_time: Optional[int] = None) -> pd.DataFrame:
        """Busca dados de candlestick da KuCoin"""
        try:
            url = f"{self.base_url}/api/v1/market/candles"
            
            params = {
                'symbol': symbol,
                'type': type_
            }
            
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
            
            # Converter para DataFrame
            df = pd.DataFrame(klines, columns=[
                'time', 'open', 'close', 'high', 'low', 'volume', 'turnover'
            ])
            
            # Converter tipos
            df['time'] = pd.to_datetime(df['time'].astype(int), unit='s')
            for col in ['open', 'close', 'high', 'low', 'volume', 'turnover']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Reordenar por data (mais antigo primeiro)
            df = df.sort_values('time').reset_index(drop=True)
            df.set_index('time', inplace=True)
            
            # Remover dados inválidos
            df = df.dropna()
            
            return df
            
        except Exception as e:
            raise Exception(f"Erro ao buscar dados de {symbol}: {e}")
    
    @st.cache_data(ttl=60)  # Cache por 1 minuto
    def get_market_data(_self, 
                       symbol: str, 
                       timeframe: str = '1day',
                       days: int = 365) -> pd.DataFrame:
        """Busca dados de mercado formatados para o indicador"""
        try:
            # Calcular timestamps
            end_time = int(datetime.now().timestamp())
            start_time = int((datetime.now() - timedelta(days=days)).timestamp())
            
            # Buscar dados
            df = _self.get_klines(
                symbol=symbol,
                type_=timeframe,
                start_time=start_time,
                end_time=end_time
            )
            
            if df.empty:
                raise Exception("Nenhum dado retornado")
            
            # Renomear colunas para compatibilidade
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

class HullVWAPIndicator:
    """Hull Suite + Dynamic Swing Anchored VWAP Hybrid Indicator"""
    
    def __init__(self, 
                 hull_source: str = 'close',
                 hull_variation: str = 'Hma',
                 hull_length: int = 55,
                 length_mult: float = 1.0,
                 swing_period: int = 50,
                 base_apt: float = 20.0,
                 use_adapt: bool = False,
                 vol_bias: float = 10.0,
                 signal_type: str = 'Hull + VWAP',
                 show_hull_band: bool = True,
                 show_vwap: bool = True,
                 show_swing_labels: bool = True,
                 show_signals: bool = True):
        
        self.hull_source = hull_source
        self.hull_variation = hull_variation
        self.hull_length = hull_length
        self.length_mult = length_mult
        self.swing_period = swing_period
        self.base_apt = base_apt
        self.use_adapt = use_adapt
        self.vol_bias = vol_bias
        self.signal_type = signal_type
        self.show_hull_band = show_hull_band
        self.show_vwap = show_vwap
        self.show_swing_labels = show_swing_labels
        self.show_signals = show_signals
    
    def wma(self, values: pd.Series, length: int) -> pd.Series:
        """Weighted Moving Average"""
        def calculate_wma(x):
            if len(x) < length or x.isna().any():
                return np.nan
            weights = np.arange(1, length + 1)
            return np.average(x.iloc[-length:], weights=weights)
        
        return values.rolling(window=length, min_periods=length).apply(calculate_wma, raw=False)
    
    def ema(self, values: pd.Series, length: int) -> pd.Series:
        """Exponential Moving Average"""
        return values.ewm(span=length, adjust=False).mean()
    
    def hma(self, src: pd.Series, length: int) -> pd.Series:
        """Hull Moving Average"""
        half_length = max(1, int(length / 2))
        sqrt_length = max(1, int(np.sqrt(length)))
        
        wma_half = self.wma(src, half_length)
        wma_full = self.wma(src, length)
        raw_hma = 2 * wma_half - wma_full
        
        return self.wma(raw_hma, sqrt_length)
    
    def ehma(self, src: pd.Series, length: int) -> pd.Series:
        """Exponential Hull Moving Average"""
        half_length = max(1, int(length / 2))
        sqrt_length = max(1, int(np.sqrt(length)))
        
        ema_half = self.ema(src, half_length)
        ema_full = self.ema(src, length)
        raw_ehma = 2 * ema_half - ema_full
        
        return self.ema(raw_ehma, sqrt_length)
    
    def thma(self, src: pd.Series, length: int) -> pd.Series:
        """Triangular Hull Moving Average"""
        third_length = max(1, int(length / 3))
        half_length = max(1, int(length / 2))
        
        wma_third = self.wma(src, third_length)
        wma_half = self.wma(src, half_length)
        wma_full = self.wma(src, length)
        
        raw_thma = wma_third * 3 - wma_half - wma_full
        return self.wma(raw_thma, length)
    
    def calculate_hull(self, src: pd.Series) -> pd.Series:
        """Calculate Hull MA based on selected variation"""
        adjusted_length = max(1, int(self.hull_length * self.length_mult))
        
        if self.hull_variation == 'Hma':
            return self.hma(src, adjusted_length)
        elif self.hull_variation == 'Ehma':
            return self.ehma(src, adjusted_length)
        elif self.hull_variation == 'Thma':
            return self.thma(src, max(1, int(adjusted_length / 2)))
        else:
            raise ValueError(f"Invalid hull_variation: {self.hull_variation}")
    
    def detect_swings(self, high: pd.Series, low: pd.Series) -> pd.Series:
        """Detect swing points (simplified for real-time performance)"""
        swing_points = pd.Series(index=high.index, dtype=float)
        
        for i in range(self.swing_period, len(high) - self.swing_period):
            # Check for swing high
            window_high = high.iloc[i-self.swing_period:i+self.swing_period+1]
            if high.iloc[i] == window_high.max():
                swing_points.iloc[i] = high.iloc[i]
            
            # Check for swing low  
            window_low = low.iloc[i-self.swing_period:i+self.swing_period+1]
            if low.iloc[i] == window_low.min():
                swing_points.iloc[i] = low.iloc[i]
        
        return swing_points
    
    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate simplified VWAP"""
        # Simplified VWAP calculation for better performance
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap
    
    def generate_signals(self, df: pd.DataFrame, hull_ma: pd.Series, vwap: pd.Series) -> tuple:
        """Generate buy/sell signals"""
        buy_signals = pd.Series([False] * len(df), index=df.index)
        sell_signals = pd.Series([False] * len(df), index=df.index)
        
        # Hull crossover conditions
        hull_cross_up = (df['close'] > hull_ma) & (df['close'].shift(1) <= hull_ma.shift(1))
        hull_cross_down = (df['close'] < hull_ma) & (df['close'].shift(1) >= hull_ma.shift(1))
        
        # VWAP conditions
        price_above_vwap = df['close'] > vwap
        price_below_vwap = df['close'] < vwap
        
        if self.signal_type == 'Hull Only':
            buy_signals = hull_cross_up
            sell_signals = hull_cross_down
        elif self.signal_type == 'VWAP Only':
            vwap_cross_up = (df['close'] > vwap) & (df['close'].shift(1) <= vwap.shift(1))
            vwap_cross_down = (df['close'] < vwap) & (df['close'].shift(1) >= vwap.shift(1))
            buy_signals = vwap_cross_up
            sell_signals = vwap_cross_down
        elif self.signal_type == 'Hull + VWAP':
            buy_signals = hull_cross_up & price_above_vwap
            sell_signals = hull_cross_down & price_below_vwap
        
        return buy_signals, sell_signals
    
    def calculate(self, df: pd.DataFrame) -> dict:
        """Main calculation method"""
        # Calculate Hull MA
        src = df[self.hull_source] if self.hull_source in df.columns else df['close']
        hull_main = self.calculate_hull(src)
        hull_band = hull_main.shift(2) if self.show_hull_band else pd.Series([np.nan] * len(df), index=df.index)
        
        # Calculate VWAP
        vwap = self.calculate_vwap(df) if self.show_vwap else pd.Series([np.nan] * len(df), index=df.index)
        
        # Detect swing points
        swing_points = self.detect_swings(df['high'], df['low']) if self.show_swing_labels else pd.Series([np.nan] * len(df), index=df.index)
        
        # Generate trading signals
        buy_signals, sell_signals = self.generate_signals(df, hull_main, vwap) if self.show_signals else (pd.Series([False] * len(df), index=df.index), pd.Series([False] * len(df), index=df.index))
        
        return {
            'hull_main': hull_main,
            'hull_band': hull_band,
            'vwap': vwap,
            'swing_points': swing_points,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'hull_trend': hull_main > hull_main.shift(1)
        }

def create_plotly_chart(df: pd.DataFrame, results: dict, symbol: str) -> go.Figure:
    """Create interactive Plotly chart"""
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=[f'{symbol} - Hull VWAP Analysis', 'Volume']
    )
    
    # Candlestick chart
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
    hull_color = '#00ff00' if results['hull_trend'].iloc[-1] else '#ff0000'
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=results['hull_main'],
            mode='lines',
            name='Hull MA',
            line=dict(color=hull_color, width=2)
        ),
        row=1, col=1
    )
    
    # Hull Band
    if not results['hull_band'].isna().all():
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=results['hull_band'],
                mode='lines',
                name='Hull Band',
                line=dict(color=hull_color, width=1, dash='dash'),
                opacity=0.6
            ),
            row=1, col=1
        )
    
    # VWAP
    if not results['vwap'].isna().all():
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=results['vwap'],
                mode='lines',
                name='VWAP',
                line=dict(color='#0088ff', width=2)
            ),
            row=1, col=1
        )
    
    # Buy signals
    buy_points = results['buy_signals']
    if buy_points.any():
        buy_indices = buy_points[buy_points].index
        fig.add_trace(
            go.Scatter(
                x=buy_indices,
                y=df.loc[buy_indices, 'low'] * 0.995,
                mode='markers',
                name='Buy Signals',
                marker=dict(
                    symbol='triangle-up',
                    size=15,
                    color='#00ff00'
                )
            ),
            row=1, col=1
        )
    
    # Sell signals
    sell_points = results['sell_signals']
    if sell_points.any():
        sell_indices = sell_points[sell_points].index
        fig.add_trace(
            go.Scatter(
                x=sell_indices,
                y=df.loc[sell_indices, 'high'] * 1.005,
                mode='markers',
                name='Sell Signals',
                marker=dict(
                    symbol='triangle-down',
                    size=15,
                    color='#ff0000'
                )
            ),
            row=1, col=1
        )
    
    # Volume chart
    colors = ['#00ff88' if close >= open else '#ff4444' 
              for close, open in zip(df['close'], df['open'])]
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.6
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} - Hull Suite + Dynamic VWAP Analysis',
        xaxis_rangeslider_visible=False,
        height=800,
        template='plotly_dark',
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

# Streamlit App
def main():
    st.title("📈 Hull VWAP Indicator - KuCoin Data")
    st.markdown("**Hull Suite + Dynamic Swing Anchored VWAP Hybrid Indicator**")
    
    # Initialize data fetcher
    if 'fetcher' not in st.session_state:
        st.session_state.fetcher = KucoinDataFetcher()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Symbol selection
        st.subheader("📊 Dados")
        popular_symbols = st.session_state.fetcher.get_popular_symbols()
        symbol = st.selectbox("Symbol", popular_symbols, index=0)
        
        timeframes = {
            '1min': '1min', '5min': '5min', '15min': '15min', 
            '30min': '30min', '1hour': '1hour', '4hour': '4hour',
            '1day': '1day', '1week': '1week'
        }
        timeframe = st.selectbox("Timeframe", list(timeframes.keys()), index=6)
        
        days = st.slider("Days Back", 7, 365, 90)
        
        # Hull Settings
        st.subheader("🔄 Hull MA Settings")
        hull_variation = st.selectbox("Hull Variation", ['Hma', 'Ehma', 'Thma'])
        hull_length = st.slider("Hull Length", 5, 200, 55)
        
        # VWAP Settings
        st.subheader("📈 VWAP Settings")
        swing_period = st.slider("Swing Period", 5, 100, 50)
        base_apt = st.slider("Base APT", 5.0, 100.0, 20.0)
        
        # Signal Settings
        st.subheader("🎯 Signal Settings")
        signal_types = ['Hull Only', 'VWAP Only', 'Hull + VWAP']
        signal_type = st.selectbox("Signal Type", signal_types, index=2)
        
        # Display Options
        st.subheader("👁️ Display Options")
        show_hull_band = st.checkbox("Show Hull Band", True)
        show_vwap = st.checkbox("Show VWAP", True)
        show_signals = st.checkbox("Show Signals", True)
        
        # Update button
        update_data = st.button("🔄 Update Data", type="primary")
    
    # Main content
    try:
        # Show loading spinner
        with st.spinner(f'Loading {symbol} data...'):
            df = st.session_state.fetcher.get_market_data(
                symbol=symbol,
                timeframe=timeframes[timeframe],
                days=days
            )
        
        if df.empty:
            st.error("No data available for the selected parameters")
            return
        
        # Create indicator
        indicator = HullVWAPIndicator(
            hull_variation=hull_variation,
            hull_length=hull_length,
            swing_period=swing_period,
            base_apt=base_apt,
            signal_type=signal_type,
            show_hull_band=show_hull_band,
            show_vwap=show_vwap,
            show_signals=show_signals
        )
        
        # Calculate indicators
        with st.spinner('Calculating indicators...'):
            results = indicator.calculate(df)
        
        # Create and display chart
        fig = create_plotly_chart(df, results, symbol)
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = df['close'].iloc[-1]
            price_change = ((current_price - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
            st.metric("Current Price", f"${current_price:.6f}", f"{price_change:.2f}%")
        
        with col2:
            buy_count = results['buy_signals'].sum()
            st.metric("Buy Signals", buy_count)
        
        with col3:
            sell_count = results['sell_signals'].sum()
            st.metric("Sell Signals", sell_count)
        
        with col4:
            trend = "🟢 Bullish" if results['hull_trend'].iloc[-1] else "🔴 Bearish"
            st.metric("Hull Trend", trend)
        
        # Recent signals table
        st.subheader("📋 Recent Signals")
        
        # Get recent signals
        recent_signals = []
        buy_signals = results['buy_signals']
        sell_signals = results['sell_signals']
        
        for i in range(len(df)):
            if buy_signals.iloc[i]:
                recent_signals.append({
                    'Time': df.index[i].strftime('%Y-%m-%d %H:%M'),
                    'Signal': '🟢 BUY',
                    'Price': f"${df['close'].iloc[i]:.6f}",
                    'Volume': f"{df['volume'].iloc[i]:,.0f}"
                })
            elif sell_signals.iloc[i]:
                recent_signals.append({
                    'Time': df.index[i].strftime('%Y-%m-%d %H:%M'),
                    'Signal': '🔴 SELL', 
                    'Price': f"${df['close'].iloc[i]:.6f}",
                    'Volume': f"{df['volume'].iloc[i]:,.0f}"
                })
        
        # Show last 10 signals
        if recent_signals:
            recent_df = pd.DataFrame(recent_signals[-10:])
            st.dataframe(recent_df, use_container_width=True)
        else:
            st.info("No recent signals found with current parameters")
        
        # Performance metrics
        st.subheader("📊 Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Hull MA Analysis**")
            hull_values = results['hull_main'].dropna()
            if len(hull_values) > 1:
                hull_change = ((hull_values.iloc[-1] - hull_values.iloc[0]) / hull_values.iloc[0]) * 100
                st.write(f"Hull MA Change: {hull_change:.2f}%")
                st.write(f"Current Hull: ${hull_values.iloc[-1]:.6f}")
                
        with col2:
            st.write("**VWAP Analysis**")
            vwap_values = results['vwap'].dropna()
            if len(vwap_values) > 1:
                price_vs_vwap = ((current_price - vwap_values.iloc[-1]) / vwap_values.iloc[-1]) * 100
                st.write(f"Price vs VWAP: {price_vs_vwap:.2f}%")
                st.write(f"Current VWAP: ${vwap_values.iloc[-1]:.6f}")
        
        # Data table
        with st.expander("📈 Raw Data"):
            display_df = df.copy()
            display_df['Hull_MA'] = results['hull_main']
            display_df['VWAP'] = results['vwap']
            display_df['Buy_Signal'] = results['buy_signals']
            display_df['Sell_Signal'] = results['sell_signals']
            
            st.dataframe(display_df.tail(50), use_container_width=True)
        
        # Export data
        st.subheader("💾 Export Data")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download CSV"):
                export_df = df.copy()
                export_df['Hull_MA'] = results['hull_main']
                export_df['VWAP'] = results['vwap']
                export_df['Buy_Signal'] = results['buy_signals']
                export_df['Sell_Signal'] = results['sell_signals']
                
                csv = export_df.to_csv()
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name=f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📊 Download Chart HTML"):
                fig_html = fig.to_html()
                st.download_button(
                    label="Download chart as HTML", 
                    data=fig_html,
                    file_name=f"{symbol}_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html"
                )
        
        # Footer info
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center'>
                <p><strong>Hull VWAP Indicator</strong> - Real-time crypto analysis with KuCoin data</p>
                <p><em>Data provided by KuCoin API | Not financial advice</em></p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        
        # Show troubleshooting tips
        st.subheader("🔧 Troubleshooting")
        st.write("""
        **Common issues:**
        1. **Network Error**: Check your internet connection
        2. **Invalid Symbol**: Make sure the trading pair exists on KuCoin
        3. **No Data**: Try a different timeframe or reduce the number of days
        4. **API Limits**: Wait a few seconds and try again
        
        **Tips:**
        - Use popular symbols like BTC-USDT, ETH-USDT
        - Start with daily timeframe for better stability
        - Reduce days if getting timeout errors
        """)
        
        # Show sample data as fallback
        if st.button("📊 Load Sample Data"):
            st.info("Loading with simulated data for demonstration...")
            
            # Create sample data
            dates = pd.date_range(start=datetime.now() - timedelta(days=90), end=datetime.now(), freq='1D')
            np.random.seed(42)
            
            price_base = 45000  # Sample BTC price
            returns = np.random.normal(0, 0.02, len(dates))
            price_data = price_base * np.cumprod(1 + returns)
            
            sample_df = pd.DataFrame({
                'high': price_data * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
                'low': price_data * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
                'close': price_data,
                'open': np.roll(price_data, 1),
                'volume': np.random.lognormal(mean=15, sigma=0.5, size=len(dates))
            }, index=dates)
            
            # Calculate with sample data
            indicator = HullVWAPIndicator(
                hull_variation=hull_variation,
                hull_length=hull_length,
                swing_period=swing_period,
                base_apt=base_apt,
                signal_type=signal_type,
                show_hull_band=show_hull_band,
                show_vwap=show_vwap,
                show_signals=show_signals
            )
            
            results = indicator.calculate(sample_df)
            fig = create_plotly_chart(sample_df, results, "BTC-USDT (Sample)")
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("Sample data loaded successfully! This demonstrates how the indicator works.")

if __name__ == "__main__":
    main()
