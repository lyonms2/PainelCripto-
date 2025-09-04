# Configurações do KuCoin Crypto Dashboard

# Configurações da API
API_CONFIG = {
    'base_url': 'https://api.kucoin.com',
    'timeout': 10,
    'max_retries': 3,
    'retry_delay': 1  # segundos
}

# Configurações de cache
CACHE_CONFIG = {
    'default_ttl': 300,  # 5 minutos
    'quick_ttl': 60,     # 1 minuto
    'long_ttl': 900      # 15 minutos
}

# Configurações de filtros padrão
DEFAULT_FILTERS = {
    'min_volume': 100000,
    'pair_filter': '-USDT$',
    'top_n_default': 20,
    'max_results': 100
}

# Configurações de gráficos
CHART_CONFIG = {
    'color_scale': 'RdYlGn',
    'template': 'plotly_white',
    'height_default': 500,
    'height_small': 300,
    'height_large': 600
}

# Configurações de formatação
FORMAT_CONFIG = {
    'price_decimals': 6,
    'percentage_decimals': 2,
    'volume_decimals': 0
}

# Configurações de auto-refresh
REFRESH_CONFIG = {
    'default_interval': 30,  # segundos
    'min_interval': 10,
    'max_interval': 300
}

# Configurações de UI
UI_CONFIG = {
    'page_title': 'KuCoin Crypto Dashboard',
    'page_icon': '₿',
    'layout': 'wide',
    'sidebar_state': 'expanded'
}

# Mensagens padrão
MESSAGES = {
    'loading': '🔄 Carregando dados da KuCoin...',
    'no_data': 'Não foi possível carregar os dados',
    'api_error': 'Erro na conexão com a API',
    'cache_cleared': 'Cache limpo com sucesso!',
    'data_updated': 'Dados atualizados com sucesso!'
}

# Configurações de exportação
EXPORT_CONFIG = {
    'default_filename': 'kucoin_crypto_data.csv',
    'max_export_rows': 1000,
    'date_format': '%Y%m%d_%H%M%S'
}