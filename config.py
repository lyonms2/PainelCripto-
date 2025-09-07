import ccxt

# Carrega todos os mercados disponíveis na KuCoin
exchange = ccxt.kucoin()
markets = exchange.load_markets()
# Filtra apenas os pares ativos que terminam com '/USDT'
SYMBOLS = [s for s, m in markets.items() if s.endswith('/USDT') and m['active']]

# Timeframe padrão para análise
TIMEFRAME = '1d'
