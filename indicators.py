import pandas as pd
import numpy as np


class MarketIndicators:
    """Classe para calcular indicadores de mercado"""
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        """
        Calcular Relative Strength Index (RSI)
        
        Args:
            prices: Series de preços
            period: Período para cálculo (default: 14)
        
        Returns:
            RSI values
        """
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception:
            return pd.Series([50] * len(prices))  # Valor neutro em caso de erro
    
    @staticmethod
    def calculate_sma(prices, period=20):
        """
        Calcular Simple Moving Average (SMA)
        
        Args:
            prices: Series de preços
            period: Período para média móvel
        
        Returns:
            SMA values
        """
        try:
            return prices.rolling(window=period).mean()
        except Exception:
            return prices  # Retorna preços originais em caso de erro
    
    @staticmethod
    def calculate_volatility(prices, period=20):
        """
        Calcular volatilidade (desvio padrão dos retornos)
        
        Args:
            prices: Series de preços
            period: Período para cálculo
        
        Returns:
            Volatility values
        """
        try:
            returns = prices.pct_change()
            volatility = returns.rolling(window=period).std() * np.sqrt(365)  # Anualizada
            return volatility
        except Exception:
            return pd.Series([0.2] * len(prices))  # Volatilidade padrão de 20%


class MarketMetrics:
    """Classe para métricas de mercado"""
    
    @staticmethod
    def format_currency(value):
        """Formatar valores monetários"""
        try:
            val = float(value)
            if val >= 1e12:
                return f"${val/1e12:.2f}T"
            elif val >= 1e9:
                return f"${val/1e9:.2f}B"
            elif val >= 1e6:
                return f"${val/1e6:.2f}M"
            elif val >= 1e3:
                return f"${val/1e3:.2f}K"
            else:
                return f"${val:.2f}"
        except (ValueError, TypeError):
            return "N/A"
    
    @staticmethod
    def format_percentage(value):
        """Formatar percentuais"""
        try:
            return f"{float(value)*100:+.2f}%"
        except (ValueError, TypeError):
            return "N/A"
    
    @staticmethod
    def format_number(value, decimals=2):
        """Formatar números"""
        try:
            return f"{float(value):,.{decimals}f}"
        except (ValueError, TypeError):
            return "N/A"
    
    @staticmethod
    def calculate_market_summary(df):
        """
        Calcular resumo do mercado
        
        Args:
            df: DataFrame com dados das moedas
        
        Returns:
            Dict com métricas do mercado
        """
        try:
            total_coins = len(df)
            gainers = len(df[df['changeRate'] > 0])
            losers = len(df[df['changeRate'] < 0])
            stable = total_coins - gainers - losers
            
            total_volume = df['volValue'].sum()
            avg_change = df['changeRate'].mean()
            median_change = df['changeRate'].median()
            
            # Volume por categoria
            volume_gainers = df[df['changeRate'] > 0]['volValue'].sum()
            volume_losers = df[df['changeRate'] < 0]['volValue'].sum()
            
            return {
                'total_coins': total_coins,
                'gainers': gainers,
                'losers': losers,
                'stable': stable,
                'gainers_pct': (gainers / total_coins * 100) if total_coins > 0 else 0,
                'losers_pct': (losers / total_coins * 100) if total_coins > 0 else 0,
                'total_volume': total_volume,
                'avg_change': avg_change,
                'median_change': median_change,
                'volume_gainers': volume_gainers,
                'volume_losers': volume_losers,
                'volume_dominance_up': (volume_gainers / total_volume * 100) if total_volume > 0 else 0
            }
        except Exception:
            return {
                'total_coins': 0,
                'gainers': 0,
                'losers': 0,
                'stable': 0,
                'gainers_pct': 0,
                'losers_pct': 0,
                'total_volume': 0,
                'avg_change': 0,
                'median_change': 0,
                'volume_gainers': 0,
                'volume_losers': 0,
                'volume_dominance_up': 0
            }
    
    @staticmethod
    def get_top_performers(df, n=10):
        """
        Obter top performers (maiores altas e baixas)
        
        Args:
            df: DataFrame com dados
            n: Número de resultados
        
        Returns:
            Dict com top gainers e losers
        """
        try:
            top_gainers = df.nlargest(n, 'changeRate')
            top_losers = df.nsmallest(n, 'changeRate')
            
            return {
                'gainers': top_gainers,
                'losers': top_losers
            }
        except Exception:
            return {
                'gainers': pd.DataFrame(),
                'losers': pd.DataFrame()
            }
    
    @staticmethod
    def categorize_by_performance(df):
        """
        Categorizar moedas por performance
        
        Args:
            df: DataFrame com dados
        
        Returns:
            DataFrame com coluna de categoria
        """
        try:
            conditions = [
                df['changeRate'] < -0.1,
                (df['changeRate'] >= -0.1) & (df['changeRate'] < -0.05),
                (df['changeRate'] >= -0.05) & (df['changeRate'] < -0.01),
                (df['changeRate'] >= -0.01) & (df['changeRate'] < 0.01),
                (df['changeRate'] >= 0.01) & (df['changeRate'] < 0.05),
                (df['changeRate'] >= 0.05) & (df['changeRate'] < 0.1),
                df['changeRate'] >= 0.1
            ]
            
            choices = [
                'Queda Forte (< -10%)',
                'Queda Moderada (-10% a -5%)',
                'Queda Leve (-5% a -1%)',
                'Estável (-1% a 1%)',
                'Alta Leve (1% a 5%)',
                'Alta Moderada (5% a 10%)',
                'Alta Forte (> 10%)'
            ]
            
            df_copy = df.copy()
            df_copy['performance_category'] = pd.Series(dtype='object')
            
            for i, condition in enumerate(conditions):
                df_copy.loc[condition, 'performance_category'] = choices[i]
            
            return df_copy
        except Exception:
            df_copy = df.copy()
            df_copy['performance_category'] = 'Indefinido'
            return df_copy
    
    @staticmethod
    def calculate_volume_distribution(df):
        """
        Calcular distribuição de volume por categoria
        
        Args:
            df: DataFrame com dados categorizados
        
        Returns:
            Dict com distribuição
        """
        try:
            if 'performance_category' not in df.columns:
                df = MarketMetrics.categorize_by_performance(df)
            
            volume_by_category = df.groupby('performance_category')['volValue'].sum().sort_values(ascending=False)
            total_volume = df['volValue'].sum()
            
            distribution = {}
            for category, volume in volume_by_category.items():
                distribution[category] = {
                    'volume': volume,
                    'percentage': (volume / total_volume * 100) if total_volume > 0 else 0,
                    'count': len(df[df['performance_category'] == category])
                }
            
            return distribution
        except Exception:
            return {}


class TechnicalAnalysis:
    """Classe para análises técnicas básicas"""
    
    @staticmethod
    def detect_support_resistance(prices, window=20):
        """
        Detectar níveis de suporte e resistência
        
        Args:
            prices: Series de preços
            window: Janela para cálculo
        
        Returns:
            Dict com níveis
        """
        try:
            rolling_max = prices.rolling(window=window).max()
            rolling_min = prices.rolling(window=window).min()
            
            current_price = prices.iloc[-1]
            
            # Níveis de resistência (máximas recentes)
            resistance_levels = rolling_max.dropna().unique()[-3:]
            resistance_levels = [level for level in resistance_levels if level > current_price]
            
            # Níveis de suporte (mínimas recentes)  
            support_levels = rolling_min.dropna().unique()[-3:]
            support_levels = [level for level in support_levels if level < current_price]
            
            return {
                'support': sorted(support_levels, reverse=True)[:2],
                'resistance': sorted(resistance_levels)[:2],
                'current_price': current_price
            }
        except Exception:
            return {
                'support': [],
                'resistance': [],
                'current_price': 0
            }
    
    @staticmethod
    def calculate_price_momentum(df, periods=[1, 7, 30]):
        """
        Calcular momentum de preço para diferentes períodos
        
        Args:
            df: DataFrame com dados históricos
            periods: Lista de períodos para calcular momentum
        
        Returns:
            DataFrame com momentum
        """
        try:
            momentum_data = []
            
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol].copy()
                if len(symbol_data) < max(periods):
                    continue
                
                momentum_info = {'symbol': symbol}
                current_price = symbol_data['last'].iloc[-1]
                
                for period in periods:
                    if len(symbol_data) >= period:
                        past_price = symbol_data['last'].iloc[-period]
                        momentum = (current_price - past_price) / past_price
                        momentum_info[f'momentum_{period}d'] = momentum
                
                momentum_data.append(momentum_info)
            
            return pd.DataFrame(momentum_data)
        except Exception:
            return pd.DataFrame()