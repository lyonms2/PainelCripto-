import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st
from indicators import MarketMetrics


class CryptoCharts:
    """Classe para criar gráficos de criptomoedas"""
    
    def __init__(self):
        self.color_scale = 'RdYlGn'
        self.template = 'plotly_white'
    
    def create_volume_chart(self, df, top_n=20, title="Volume de Negociação (24h)"):
        """
        Criar gráfico de barras de volume
        
        Args:
            df: DataFrame com dados
            top_n: Número de moedas para mostrar
            title: Título do gráfico
        
        Returns:
            Plotly figure
        """
        try:
            top_coins = df.head(top_n)
            
            fig = px.bar(
                top_coins,
                x='symbol',
                y='volValue',
                title=title,
                color='changeRate',
                color_continuous_scale=self.color_scale,
                labels={
                    'volValue': 'Volume (USD)',
                    'symbol': 'Símbolo',
                    'changeRate': 'Mudança 24h'
                },
                hover_data={
                    'last': ':$.4f',
                    'changeRate': ':.2%'
                }
            )
            
            fig.update_layout(
                xaxis_tickangle=-45,
                height=500,
                showlegend=False,
                template=self.template,
                xaxis_title="Criptomoedas",
                yaxis_title="Volume (USD)"
            )
            
            # Formatação dos valores no hover
            fig.update_traces(
                hovertemplate="<b>%{x}</b><br>" +
                              "Volume: $%{y:,.0f}<br>" +
                              "Preço: $%{customdata[0]:,.4f}<br>" +
                              "Mudança: %{customdata[1]:+.2%}<br>" +
                              "<extra></extra>"
            )
            
            return fig
        except Exception as e:
            st.error(f"Erro ao criar gráfico de volume: {str(e)}")
            return None
    
    def create_price_change_distribution(self, df, title="Distribuição de Mudanças de Preço"):
        """
        Criar gráfico de pizza da distribuição de mudanças
        
        Args:
            df: DataFrame com dados
            title: Título do gráfico
        
        Returns:
            Plotly figure
        """
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
            
            labels = ['< -10%', '-10% a -5%', '-5% a 0%', '0% a 5%', '5% a 10%', '> 10%']
            colors = ['#FF4444', '#FF8888', '#FFCCCC', '#CCFFCC', '#88FF88', '#44FF44']
            
            # Contar em cada categoria
            counts = []
            for condition in conditions:
                counts.append(df[condition].shape[0])
            
            # Filtrar categorias vazias
            filtered_labels = []
            filtered_counts = []
            filtered_colors = []
            
            for i, count in enumerate(counts):
                if count > 0:
                    filtered_labels.append(labels[i])
                    filtered_counts.append(count)
                    filtered_colors.append(colors[i])
            
            fig = px.pie(
                values=filtered_counts,
                names=filtered_labels,
                title=title,
                color_discrete_sequence=filtered_colors
            )
            
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate="<b>%{label}</b><br>" +
                              "Quantidade: %{value}<br>" +
                              "Percentual: %{percent}<br>" +
                              "<extra></extra>"
            )
            
            fig.update_layout(
                template=self.template,
                height=400
            )
            
            return fig
        except Exception as e:
            st.error(f"Erro ao criar gráfico de distribuição: {str(e)}")
            return None
    
    def create_price_vs_volume_scatter(self, df, title="Preço vs Volume"):
        """
        Criar scatter plot de preço vs volume
        
        Args:
            df: DataFrame com dados
            title: Título do gráfico
        
        Returns:
            Plotly figure
        """
        try:
            # Filtrar dados válidos e top 100 para clareza
            df_plot = df.head(100)
            
            fig = px.scatter(
                df_plot,
                x='volValue',
                y='last',
                title=title,
                color='changeRate',
                size='abs_change',
                hover_name='symbol',
                color_continuous_scale=self.color_scale,
                labels={
                    'volValue': 'Volume (USD)',
                    'last': 'Preço (USD)',
                    'changeRate': 'Mudança 24h',
                    'abs_change': 'Variação Absoluta'
                }
            )
            
            fig.update_layout(
                template=self.template,
                height=500,
                xaxis_type="log",  # Escala logarítmica para volume
                yaxis_type="log"   # Escala logarítmica para preço
            )
            
            fig.update_traces(
                hovertemplate="<b>%{hovertext}</b><br>" +
                              "Volume: $%{x:,.0f}<br>" +
                              "Preço: $%{y:,.4f}<br>" +
                              "Mudança: %{customdata[0]:+.2%}<br>" +
                              "<extra></extra>"
            )
            
            return fig
        except Exception as e:
            st.error(f"Erro ao criar scatter plot: {str(e)}")
            return None
    
    def create_candlestick_chart(self, klines_data, symbol, title=None):
        """
        Criar gráfico de candlestick
        
        Args:
            klines_data: Dados de klines da API
            symbol: Símbolo da moeda
            title: Título personalizado
        
        Returns:
            Plotly figure
        """
        try:
            if not klines_data:
                return None
            
            # Converter dados para DataFrame
            df = pd.DataFrame(
                klines_data, 
                columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover']
            )
            
            # Converter timestamp para datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Converter valores para float
            for col in ['open', 'close', 'high', 'low', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Ordenar por data
            df = df.sort_values('datetime')
            
            # Criar subplots (preço + volume)
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=(f'Preço de {symbol}', 'Volume'),
                row_width=[0.7, 0.3]
            )
            
            # Candlestick
            fig.add_trace(
                go.Candlestick(
                    x=df['datetime'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='Preço',
                    increasing_line_color='green',
                    decreasing_line_color='red'
                ),
                row=1, col=1
            )
            
            # Volume
            colors = ['green' if close >= open else 'red' 
                     for close, open in zip(df['close'], df['open'])]
            
            fig.add_trace(
                go.Bar(
                    x=df['datetime'],
                    y=df['volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.6
                ),
                row=2, col=1
            )
            
            # Layout
            fig.update_layout(
                title=title or f"Gráfico de {symbol} - Últimas 24h",
                template=self.template,
                height=600,
                showlegend=False,
                xaxis_rangeslider_visible=False
            )
            
            fig.update_xaxes(title_text="Tempo", row=2, col=1)
            fig.update_yaxes(title_text="Preço (USD)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            
            return fig
        except Exception as e:
            st.error(f"Erro ao criar candlestick: {str(e)}")
            return None
    
    def create_top_gainers_losers_chart(self, df, n=10):
        """
        Criar gráfico comparativo de maiores altas e baixas
        
        Args:
            df: DataFrame com dados
            n: Número de moedas para cada categoria
        
        Returns:
            Plotly figure
        """
        try:
            top_gainers = df.nlargest(n, 'changeRate')
            top_losers = df.nsmallest(n, 'changeRate')
            
            # Combinar dados
            combined_data = pd.concat([
                top_gainers[['symbol', 'changeRate']].assign(category='Maiores Altas'),
                top_losers[['symbol', 'changeRate']].assign(category='Maiores Baixas')
            ])
            
            fig = px.bar(
                combined_data,
                x='symbol',
                y='changeRate',
                color='category',
                title=f"Top {n} Maiores Altas vs Baixas (24h)",
                color_discrete_map={
                    'Maiores Altas': 'green',
                    'Maiores Baixas': 'red'
                },
                labels={
                    'changeRate': 'Mudança (%)',
                    'symbol': 'Símbolo'
                }
            )
            
            fig.update_layout(
                template=self.template,
                height=400,
                xaxis_tickangle=-45,
                yaxis=dict(tickformat='.2%')
            )
            
            fig.update_traces(
                hovertemplate="<b>%{x}</b><br>" +
                              "Mudança: %{y:+.2%}<br>" +
                              "<extra></extra>"
            )
            
            return fig
        except Exception as e:
            st.error(f"Erro ao criar gráfico de gainers/losers: {str(e)}")
            return None
    
    def create_volume_heatmap(self, df, top_n=20):
        """
        Criar heatmap de volume por mudança de preço
        
        Args:
            df: DataFrame com dados
            top_n: Número de moedas para mostrar
        
        Returns:
            Plotly figure
        """
        try:
            top_coins = df.head(top_n)
            
            # Preparar dados para heatmap
            symbols = top_coins['symbol'].tolist()
            volumes = top_coins['volValue'].tolist()
            changes = top_coins['changeRate'].tolist()
            
            # Criar matriz para heatmap (1D convertida para 2D visual)
            n_cols = 5
            n_rows = (len(symbols) + n_cols - 1) // n_cols
            
            # Preencher matriz
            z_volume = [[0 for _ in range(n_cols)] for _ in range(n_rows)]
            z_change = [[0 for _ in range(n_cols)] for _ in range(n_rows)]
            text_labels = [["" for _ in range(n_cols)] for _ in range(n_rows)]
            
            for i, (symbol, volume, change) in enumerate(zip(symbols, volumes, changes)):
                row = i // n_cols
                col = i % n_cols
                z_volume[row][col] = volume
                z_change[row][col] = change
                text_labels[row][col] = f"{symbol}<br>{MarketMetrics.format_percentage(change)}"
            
            fig = go.Figure(data=go.Heatmap(
                z=z_change,
                text=text_labels,
                texttemplate="%{text}",
                textfont={"size": 10},
                colorscale=self.color_scale,
                hovertemplate="<b>%{text}</b><br>Mudança: %{z:+.2%}<extra></extra>",
                colorbar=dict(title="Mudança 24h (%)")
            ))
            
            fig.update_layout(
                title=f"Heatmap - Top {top_n} por Volume",
                template=self.template,
                height=300,
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False)
            )
            
            return fig
        except Exception as e:
            st.error(f"Erro ao criar heatmap: {str(e)}")
            return None
    
    def create_market_overview_chart(self, market_summary):
        """
        Criar gráfico de visão geral do mercado
        
        Args:
            market_summary: Dict com métricas do mercado
        
        Returns:
            Plotly figure
        """
        try:
            categories = ['Em Alta', 'Em Baixa', 'Estável']
            values = [
                market_summary['gainers'],
                market_summary['losers'],
                market_summary['stable']
            ]
            colors = ['green', 'red', 'gray']
            
            fig = px.pie(
                values=values,
                names=categories,
                title="Distribuição do Mercado (24h)",
                color_discrete_sequence=colors
            )
            
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate="<b>%{label}</b><br>" +
                              "Quantidade: %{value}<br>" +
                              "Percentual: %{percent}<br>" +
                              "<extra></extra>"
            )
            
            fig.update_layout(
                template=self.template,
                height=400
            )
            
            return fig
        except Exception as e:
            st.error(f"Erro ao criar visão geral: {str(e)}")
            return None