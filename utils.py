import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import time


class CacheManager:
    """Gerenciador de cache personalizado"""
    
    @staticmethod
    @st.cache_data(ttl=300)  # 5 minutos
    def get_cached_data(_api_function, *args, **kwargs):
        """Cache genérico para funções da API"""
        return _api_function(*args, **kwargs)
    
    @staticmethod
    def clear_all_cache():
        """Limpar todo o cache"""
        st.cache_data.clear()
        st.success("Cache limpo com sucesso!")


class DataValidator:
    """Validador de dados"""
    
    @staticmethod
    def validate_dataframe(df, required_columns=None):
        """
        Validar se DataFrame tem as colunas necessárias
        
        Args:
            df: DataFrame para validar
            required_columns: Lista de colunas obrigatórias
        
        Returns:
            bool: True se válido
        """
        if df.empty:
            return False
        
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                st.error(f"Colunas ausentes: {missing_cols}")
                return False
        
        return True
    
    @staticmethod
    def validate_api_response(response_data):
        """
        Validar resposta da API
        
        Args:
            response_data: Dados da resposta
        
        Returns:
            bool: True se válido
        """
        if not response_data:
            st.error("Resposta da API está vazia")
            return False
        
        if isinstance(response_data, list) and len(response_data) == 0:
            st.warning("API retornou lista vazia")
            return False
        
        return True


class UIComponents:
    """Componentes de interface personalizados"""
    
    @staticmethod
    def create_metric_card(title, value, delta=None, help_text=None):
        """
        Criar card de métrica personalizado
        
        Args:
            title: Título da métrica
            value: Valor principal
            delta: Valor de variação (opcional)
            help_text: Texto de ajuda (opcional)
        """
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.metric(
                label=title,
                value=value,
                delta=delta,
                help=help_text
            )
    
    @staticmethod
    def create_info_box(title, content, box_type="info"):
        """
        Criar caixa de informação
        
        Args:
            title: Título da caixa
            content: Conteúdo
            box_type: Tipo (info, success, warning, error)
        """
        if box_type == "info":
            st.info(f"**{title}**\n\n{content}")
        elif box_type == "success":
            st.success(f"**{title}**\n\n{content}")
        elif box_type == "warning":
            st.warning(f"**{title}**\n\n{content}")
        elif box_type == "error":
            st.error(f"**{title}**\n\n{content}")
    
    @staticmethod
    def create_loading_spinner(message="Carregando dados..."):
        """Criar spinner de carregamento"""
        return st.spinner(message)
    
    @staticmethod
    def create_sidebar_filters(df):
        """
        Criar filtros na sidebar
        
        Args:
            df: DataFrame com dados
        
        Returns:
            dict: Filtros selecionados
        """
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔧 Filtros")
        
        # Filtro de volume mínimo
        if not df.empty:
            min_vol = int(df['volValue'].min())
            max_vol = int(df['volValue'].max())
            
            volume_filter = st.sidebar.select_slider(
                "Volume mínimo (USD)",
                options=[0, 10000, 50000, 100000, 500000, 1000000, 5000000],
                value=100000,
                format_func=lambda x: f"${x:,}"
            )
        else:
            volume_filter = 100000
        
        # Filtro de mudança de preço
        price_change_filter = st.sidebar.selectbox(
            "Filtro de mudança",
            [
                "Todos",
                "Apenas ganhos",
                "Apenas perdas",
                "Ganhos > 5%",
                "Perdas > 5%",
                "Ganhos > 10%",
                "Perdas > 10%"
            ]
        )
        
        # Filtro de número de resultados
        top_n = st.sidebar.slider(
            "Número de resultados",
            min_value=10,
            max_value=100,
            value=20,
            step=10
        )
        
        return {
            'volume_filter': volume_filter,
            'price_change_filter': price_change_filter,
            'top_n': top_n
        }
    
    @staticmethod
    def apply_filters(df, filters):
        """
        Aplicar filtros ao DataFrame
        
        Args:
            df: DataFrame original
            filters: Dict com filtros
        
        Returns:
            DataFrame filtrado
        """
        if df.empty:
            return df
        
        # Filtro de volume
        df_filtered = df[df['volValue'] >= filters['volume_filter']]
        
        # Filtro de mudança de preço
        change_filter = filters['price_change_filter']
        
        if change_filter == "Apenas ganhos":
            df_filtered = df_filtered[df_filtered['changeRate'] > 0]
        elif change_filter == "Apenas perdas":
            df_filtered = df_filtered[df_filtered['changeRate'] < 0]
        elif change_filter == "Ganhos > 5%":
            df_filtered = df_filtered[df_filtered['changeRate'] > 0.05]
        elif change_filter == "Perdas > 5%":
            df_filtered = df_filtered[df_filtered['changeRate'] < -0.05]
        elif change_filter == "Ganhos > 10%":
            df_filtered = df_filtered[df_filtered['changeRate'] > 0.1]
        elif change_filter == "Perdas > 10%":
            df_filtered = df_filtered[df_filtered['changeRate'] < -0.1]
        
        return df_filtered.head(filters['top_n'])


class SessionManager:
    """Gerenciador de sessão do Streamlit"""
    
    @staticmethod
    def init_session_state():
        """Inicializar variáveis de sessão"""
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
        
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = False
        
        if 'selected_coin' not in st.session_state:
            st.session_state.selected_coin = None
        
        if 'theme_mode' not in st.session_state:
            st.session_state.theme_mode = 'light'
    
    @staticmethod
    def update_last_refresh():
        """Atualizar timestamp da última atualização"""
        st.session_state.last_update = datetime.now()
    
    @staticmethod
    def get_time_since_update():
        """Obter tempo desde a última atualização"""
        if 'last_update' in st.session_state:
            delta = datetime.now() - st.session_state.last_update
            return delta.total_seconds()
        return 0
    
    @staticmethod
    def should_auto_refresh(interval=30):
        """
        Verificar se deve fazer auto-refresh
        
        Args:
            interval: Intervalo em segundos
        
        Returns:
            bool: True se deve atualizar
        """
        if not st.session_state.get('auto_refresh', False):
            return False
        
        return SessionManager.get_time_since_update() >= interval


class DataExporter:
    """Exportador de dados"""
    
    @staticmethod
    def to_csv(df, filename="crypto_data.csv"):
        """
        Converter DataFrame para CSV
        
        Args:
            df: DataFrame
            filename: Nome do arquivo
        
        Returns:
            Dados CSV como string
        """
        return df.to_csv(index=False)
    
    @staticmethod
    def create_download_button(df, filename="crypto_data.csv", label="📥 Baixar dados"):
        """
        Criar botão de download
        
        Args:
            df: DataFrame
            filename: Nome do arquivo
            label: Label do botão
        """
        if not df.empty:
            csv_data = DataExporter.to_csv(df, filename)
            st.download_button(
                label=label,
                data=csv_data,
                file_name=filename,
                mime='text/csv'
            )


class PerformanceMonitor:
    """Monitor de performance da aplicação"""
    
    @staticmethod
    def measure_time(func):
        """
        Decorator para medir tempo de execução
        
        Args:
            func: Função para medir
        
        Returns:
            Resultado da função e tempo
        """
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Log do tempo (apenas em desenvolvimento)
            if st.get_option('client.showErrorDetails'):
                st.sidebar.text(f"{func.__name__}: {execution_time:.2f}s")
            
            return result
        return wrapper
    
    @staticmethod
    def show_performance_metrics():
        """Mostrar métricas de performance na sidebar"""
        with st.sidebar.expander("⚡ Performance"):
            last_update = SessionManager.get_time_since_update()
            st.text(f"Última atualização: {last_update:.0f}s atrás")
            
            # Informações da sessão
            if hasattr(st.session_state, 'last_update'):
                st.text(f"Sessão iniciada: {st.session_state.last_update.strftime('%H:%M:%S')}")


class ErrorHandler:
    """Manipulador de erros"""
    
    @staticmethod
    def handle_api_error(error, context="API"):
        """
        Manipular erros da API
        
        Args:
            error: Exceção capturada
            context: Contexto do erro
        """
        error_message = f"Erro no {context}: {str(error)}"
        st.error(error_message)
        
        # Log detalhado apenas em desenvolvimento
        if st.get_option('client.showErrorDetails'):
            st.exception(error)
    
    @staticmethod
    def show_fallback_message(message="Não foi possível carregar os dados"):
        """
        Mostrar mensagem de fallback
        
        Args:
            message: Mensagem personalizada
        """
        st.warning(f"⚠️ {message}")
        st.info("💡 Tente atualizar a página ou aguarde alguns minutos.")
    
    @staticmethod
    def validate_and_show_error(condition, error_message):
        """
        Validar condição e mostrar erro se necessário
        
        Args:
            condition: Condição para validar
            error_message: Mensagem de erro
        
        Returns:
            bool: True se válido
        """
        if not condition:
            st.error(f"❌ {error_message}")
            return False
        return True
