import requests
import pandas as pd
import streamlit as st


class KuCoinAPI:
    """Classe para interagir com a API da KuCoin"""
    
    def __init__(self):
        self.base_url = "https://api.kucoin.com"
        self.timeout = 10
    
    def get_all_tickers(self):
        """Obter dados de ticker 24h para todas as moedas"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/market/allTickers", 
                timeout=self.timeout
            )
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
    
    def get_symbols(self):
        """Obter todos os símbolos disponíveis na KuCoin"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/symbols", 
                timeout=self.timeout
            )
            if response.status_code == 200:
                return response.json().get('data', [])
            return []
        except Exception as e:
            st.error(f"Erro ao obter símbolos: {e}")
            return []
    
    def get_market_stats(self, symbol):
        """Obter estatísticas do mercado para um símbolo específico"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/market/stats",
                params={'symbol': symbol},
                timeout=self.timeout
            )
            if response.status_code == 200:
                return response.json().get('data')
            return None
        except Exception as e:
            st.error(f"Erro ao obter estatísticas: {e}")
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
                
            response = requests.get(
                f"{self.base_url}/api/v1/market/candles", 
                params=params,
                timeout=self.timeout
            )
            if response.status_code == 200:
                return response.json().get('data', [])
            return []
        except Exception as e:
            st.error(f"Erro ao obter klines: {e}")
            return []


def process_ticker_data(raw_data, pair_filter='-USDT$'):
    """
    Processar dados brutos da API de tickers
    
    Args:
        raw_data: Dados brutos da API
        pair_filter: Filtro para pares (regex)
    
    Returns:
        DataFrame processado
    """
    if not raw_data:
        return pd.DataFrame()
    
    try:
        df = pd.DataFrame(raw_data)
        
        # Filtrar pares específicos
        if pair_filter:
            df = df[df['symbol'].str.contains(pair_filter, regex=True, na=False)]
        
        # Converter colunas numéricas essenciais
        numeric_cols = ['last', 'changeRate', 'changePrice', 'volValue', 'vol', 'high', 'low']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Filtrar dados válidos
        df = df.dropna(subset=['last', 'changeRate', 'volValue'])
        df = df[df['volValue'] > 0]  # Volume deve ser positivo
        
        # Adicionar colunas calculadas
        df['market_cap_estimate'] = df['last'] * df['vol']
        df['abs_change'] = abs(df['changeRate'])
        
        # Ordenar por volume
        df = df.sort_values('volValue', ascending=False)
        
        return df
        
    except Exception as e:
        st.error(f"Erro ao processar dados: {str(e)}")
        return pd.DataFrame()