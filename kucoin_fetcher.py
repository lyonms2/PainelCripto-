import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Optional, List, Dict
import json

class KucoinDataFetcher:
    """
    Fetcher de dados da KuCoin API pública
    """
    
    def __init__(self):
        self.base_url = "https://api.kucoin.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_symbols(self) -> List[Dict]:
        """Busca todos os símbolos disponíveis"""
        try:
            url = f"{self.base_url}/api/v1/symbols"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('code') == '200000':
                return data.get('data', [])
            else:
                print(f"Erro na API: {data}")
                return []
        except Exception as e:
            print(f"Erro ao buscar símbolos: {e}")
            return []
    
    def get_popular_symbols(self) -> List[str]:
        """Retorna símbolos populares para facilitar a seleção"""
        popular = [
            'BTC-USDT', 'ETH-USDT', 'BNB-USDT', 'ADA-USDT', 'SOL-USDT',
            'DOT-USDT', 'LINK-USDT', 'MATIC-USDT', 'AVAX-USDT', 'UNI-USDT',
            'ATOM-USDT', 'FTM-USDT', 'NEAR-USDT', 'ALGO-USDT', 'XRP-USDT'
        ]
        return popular
    
    def get_klines(self, 
                   symbol: str, 
                   type_: str = '1day',
                   start_time: Optional[int] = None,
                   end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Busca dados de candlestick da KuCoin
        
        Args:
            symbol: Par de trading (ex: 'BTC-USDT')
            type_: Timeframe ('1min', '3min', '5min', '15min', '30min', '1hour', '2hour', '4hour', '6hour', '8hour', '12hour', '1day', '1week')
            start_time: Timestamp de início (unix timestamp)
            end_time: Timestamp de fim (unix timestamp)
        """
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
            
            print(f"Buscando dados para {symbol} ({type_})...")
            
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
            
            print(f"✅ {len(df)} registros baixados para {symbol}")
            
            return df
            
        except requests.exceptions.Timeout:
            raise Exception(f"Timeout ao buscar dados de {symbol}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Erro de rede: {e}")
        except Exception as e:
            raise Exception(f"Erro ao buscar dados de {symbol}: {e}")
    
    def get_market_data(self, 
                       symbol: str, 
                       timeframe: str = '1day',
                       days: int = 365) -> pd.DataFrame:
        """
        Busca dados de mercado formatados para o indicador
        
        Args:
            symbol: Par de trading
            timeframe: Timeframe dos dados
            days: Número de dias históricos
        """
        try:
            # Calcular timestamps
            end_time = int(datetime.now().timestamp())
            start_time = int((datetime.now() - timedelta(days=days)).timestamp())
            
            # Buscar dados
            df = self.get_klines(
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
                'volume': df['volume']
            }, index=df.index)
            
            return df_formatted
            
        except Exception as e:
            print(f"Erro em get_market_data: {e}")
            raise

# Teste da classe
if __name__ == "__main__":
    fetcher = KucoinDataFetcher()
    
    # Testar busca de símbolos
    print("Testando busca de símbolos...")
    symbols = fetcher.get_symbols()
    if symbols:
        print(f"✅ {len(symbols)} símbolos encontrados")
    else:
        print("❌ Erro ao buscar símbolos")
    
    # Testar busca de dados
    print("\nTestando busca de dados...")
    try:
        df = fetcher.get_market_data('BTC-USDT', '1day', 30)
        print(f"✅ Dados de BTC-USDT: {len(df)} registros")
        print(df.head())
    except Exception as e:
        print(f"❌ Erro: {e}")
