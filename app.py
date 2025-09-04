import streamlit as st
import pandas as pd
import time

# Importar módulos personalizados
from kucoin_api import KuCoinAPI, process_ticker_data
from indicators import MarketMetrics, MarketIndicators, TechnicalAnalysis
from charts import CryptoCharts
from utils import (
    CacheManager, DataValidator, UIComponents, 
    SessionManager, DataExporter, PerformanceMonitor, ErrorHandler
)

# Configuração da página
st.set_page_config(
    page_title="KuCoin Crypto Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 1rem 0;
    background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 2rem;
}

.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border-left: 4px solid #1e3c72;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 2rem;
}

.stTabs [data-baseweb="tab"] {
    padding: 0.5rem 1rem;
    background-color: #f0f2f6;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)


class CryptoDashboardApp:
    """Classe principal do aplicativo"""
    
    def __init__(self):
        self.api = KuCoinAPI()
        self.charts = CryptoCharts()
        self.metrics = MarketMetrics()
        
        # Inicializar sessão
        SessionManager.init_session_state()
    
    @PerformanceMonitor.measure_time
    def load_data(self):
        """Carregar dados das criptomoedas"""
        try:
            with UIComponents.create_loading_spinner("🔄 Carregando dados da KuCoin..."):
                # Obter dados via cache
                raw_data = CacheManager.get_cached_data(
                    self.api.get_all_tickers
                )
                
                # Validar resposta
                if not DataValidator.validate_api_response(raw_data):
                    return pd.DataFrame()
                
                # Processar dados
                df = process_ticker_data(raw_data)
                
                # Validar DataFrame
                required_columns = ['symbol', 'last', 'changeRate', 'volValue']
                if not DataValidator.validate_dataframe(df, required_columns):
                    return pd.DataFrame()
                
                # Atualizar timestamp
                SessionManager.update_last_refresh()
                
                return df
        
        except Exception as e:
            ErrorHandler.handle_api_error(e, "carregamento de dados")
            return pd.DataFrame()
    
    def render_header(self):
        """Renderizar cabeçalho do dashboard"""
        st.markdown("""
        <div class="main-header">
            <h1>🚀 KuCoin Cryptocurrency Dashboard</h1>
            <p>Dashboard em tempo real com dados de criptomoedas da KuCoin</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self, df):
        """Renderizar sidebar com filtros e controles"""
        st.sidebar.title("⚙️ Painel de Controle")
        
        # Informações de atualização
        with st.sidebar.expander("📊 Status", expanded=True):
            last_update = SessionManager.get_time_since_update()
            st.info(f"⏱️ Última atualização: {last_update:.0f}s atrás")
            
            if not df.empty:
                st.success(f"✅ {len(df)} moedas carregadas")
            else:
                st.error("❌ Sem dados disponíveis")
        
        # Auto-refresh
        st.sidebar.markdown("---")
        auto_refresh = st.sidebar.checkbox(
            "🔄 Auto-refresh (30s)",
            value=st.session_state.auto_refresh,
            help="Atualizar dados automaticamente a cada 30 segundos"
        )
        st.session_state.auto_refresh = auto_refresh
        
        # Botões de controle
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🔄 Atualizar", type="primary"):
                CacheManager.clear_all_cache()
                st.rerun()
        
        with col2:
            if st.button("🧹 Limpar Cache"):
                CacheManager.clear_all_cache()
        
        # Filtros
        if not df.empty:
            return UIComponents.create_sidebar_filters(df)
        
        return {
            'volume_filter': 100000,
            'price_change_filter': 'Todos',
            'top_n': 20
        }
    
    def render_metrics_overview(self, df):
        """Renderizar métricas principais"""
        if df.empty:
            st.warning("⚠️ Sem dados para mostrar métricas")
            return
        
        st.subheader("📊 Visão Geral do Mercado")
        
        # Calcular métricas
        market_summary = self.metrics.calculate_market_summary(df)
        
        # Primeira linha de métricas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total de Moedas",
                market_summary['total_coins'],
                help="Total de criptomoedas monitoradas"
            )
        
        with col2:
            st.metric(
                "Em Alta",
                market_summary['gainers'],
                delta=f"{market_summary['gainers_pct']:.1f}%",
                delta_color="normal"
            )
        
        with col3:
            st.metric(
                "Em Baixa",
                market_summary['losers'],
                delta=f"-{market_summary['losers_pct']:.1f}%",
                delta_color="inverse"
            )
        
        with col4:
            st.metric(
                "Volume Total",
                self.metrics.format_currency(market_summary['total_volume'])
            )
        
        # Segunda linha de métricas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Mudança Média",
                self.metrics.format_percentage(market_summary['avg_change'])
            )
        
        with col2:
            st.metric(
                "Mudança Mediana",
                self.metrics.format_percentage(market_summary['median_change'])
            )
        
        with col3:
            st.metric(
                "Dominância Volume Alta",
                f"{market_summary['volume_dominance_up']:.1f}%"
            )
        
        with col4:
            volume_ratio = (market_summary['volume_gainers'] / 
                          market_summary['volume_losers'] 
                          if market_summary['volume_losers'] > 0 else 0)
            st.metric(
                "Ratio Vol. Alta/Baixa",
                f"{volume_ratio:.2f}x"
            )
    
    def render_main_tabs(self, df_filtered):
        """Renderizar abas principais do dashboard"""
        if df_filtered.empty:
            ErrorHandler.show_fallback_message("Nenhuma moeda encontrada com os filtros aplicados")
            return
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "🏆 Top Volume", 
            "📈 Maiores Variações", 
            "📊 Análise Técnica", 
            "🔍 Busca Detalhada"
        ])
        
        with tab1:
            self.render_volume_tab(df_filtered)
        
        with tab2:
            self.render_variations_tab(df_filtered)
        
        with tab3:
            self.render_analysis_tab(df_filtered)
        
        with tab4:
            self.render_search_tab(df_filtered)
    
    def render_volume_tab(self, df):
        """Renderizar aba de volume"""
        st.subheader("🏆 Top Criptomoedas por Volume de Negociação")
        
        # Gráfico de volume
        fig_volume = self.charts.create_volume_chart(df, top_n=20)
        if fig_volume:
            st.plotly_chart(fig_volume, use_container_width=True)
        
        # Tabela detalhada
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### 📋 Dados Detalhados")
            display_df = self.prepare_display_dataframe(df.head(20))
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        with col2:
            # Gráfico de distribuição
            market_summary = self.metrics.calculate_market_summary(df)
            fig_overview = self.charts.create_market_overview_chart(market_summary)
            if fig_overview:
                st.plotly_chart(fig_overview, use_container_width=True)
    
    def render_variations_tab(self, df):
        """Renderizar aba de variações"""
        st.subheader("📈 Maiores Variações nas Últimas 24 Horas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🚀 Maiores Altas")
            top_gainers = df.nlargest(10, 'changeRate')
            gainers_display = self.prepare_display_dataframe(top_gainers, ['symbol', 'last', 'changeRate', 'volValue'])
            st.dataframe(gainers_display, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("### 📉 Maiores Baixas")
            top_losers = df.nsmallest(10, 'changeRate')
            losers_display = self.prepare_display_dataframe(top_losers, ['symbol', 'last', 'changeRate', 'volValue'])
            st.dataframe(losers_display, use_container_width=True, hide_index=True)
        
        # Gráfico comparativo
        fig_gainers_losers = self.charts.create_top_gainers_losers_chart(df, n=10)
        if fig_gainers_losers:
            st.plotly_chart(fig_gainers_losers, use_container_width=True)
    
    def render_analysis_tab(self, df):
        """Renderizar aba de análise técnica"""
        st.subheader("📊 Análise Técnica e Distribuição")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de distribuição de mudanças
            fig_dist = self.charts.create_price_change_distribution(df)
            if fig_dist:
                st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Scatter plot preço vs volume
            fig_scatter = self.charts.create_price_vs_volume_scatter(df)
            if fig_scatter:
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Heatmap
        st.markdown("### 🔥 Heatmap de Performance")
        fig_heatmap = self.charts.create_volume_heatmap(df, top_n=20)
        if fig_heatmap:
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
    def render_search_tab(self, df):
        """Renderizar aba de busca detalhada"""
        st.subheader("🔍 Análise Detalhada por Moeda")
        
        # Seleção de moeda
        symbols = df['symbol'].tolist()
        selected_symbol = st.selectbox(
            "Selecione uma criptomoeda para análise detalhada:",
            symbols,
            help="Escolha uma moeda para ver gráficos e análises específicas"
        )
        
        if selected_symbol:
            # Dados da moeda selecionada
            coin_data = df[df['symbol'] == selected_symbol].iloc[0]
            
            # Métricas da moeda
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Preço Atual", f"${coin_data['last']:.6f}")
            
            with col2:
                st.metric(
                    "Mudança 24h", 
                    self.metrics.format_percentage(coin_data['changeRate']),
                    delta=self.metrics.format_percentage(coin_data['changeRate'])
                )
            
            with col3:
                st.metric("Volume", self.metrics.format_currency(coin_data['volValue']))
            
            with col4:
                st.metric("Alta/Baixa", f"${coin_data['high']:.6f} / ${coin_data['low']:.6f}")
            
            # Gráfico de candlestick (se disponível)
            try:
                with st.spinner("Carregando gráfico histórico..."):
                    end_time = int(time.time())
                    start_time = end_time - (24 * 60 * 60)  # 24 horas
                    
                    klines = self.api.get_klines(
                        selected_symbol, 
                        type_="1hour",
                        start_at=start_time,
                        end_at=end_time
                    )
                    
                    fig_candlestick = self.charts.create_candlestick_chart(
                        klines, 
                        selected_symbol
                    )
                    
                    if fig_candlestick:
                        st.plotly_chart(fig_candlestick, use_container_width=True)
                    else:
                        st.info("💡 Gráfico histórico não disponível para esta moeda")
            
            except Exception as e:
                st.warning(f"⚠️ Não foi possível carregar dados históricos: {str(e)}")
    
    def prepare_display_dataframe(self, df, columns=None):
        """Preparar DataFrame para exibição"""
        if df.empty:
            return df
        
        display_df = df.copy()
        
        # Selecionar colunas se especificado
        if columns:
            display_df = display_df[columns]
        else:
            display_df = display_df[['symbol', 'last', 'changeRate', 'volValue', 'high', 'low']]
        
        # Formatação
        display_df = display_df.rename(columns={
            'symbol': 'Símbolo',
            'last': 'Preço',
            'changeRate': 'Mudança 24h',
            'volValue': 'Volume',
            'high': 'Alta 24h',
            'low': 'Baixa 24h'
        })
        
        # Aplicar formatação
        if 'Preço' in display_df.columns:
            display_df['Preço'] = display_df['Preço'].apply(lambda x: f"${x:.6f}")
        
        if 'Mudança 24h' in display_df.columns:
            display_df['Mudança 24h'] = display_df['Mudança 24h'].apply(self.metrics.format_percentage)
        
        if 'Volume' in display_df.columns:
            display_df['Volume'] = display_df['Volume'].apply(self.metrics.format_currency)
        
        if 'Alta 24h' in display_df.columns:
            display_df['Alta 24h'] = display_df['Alta 24h'].apply(lambda x: f"${x:.6f}")
        
        if 'Baixa 24h' in display_df.columns:
            display_df['Baixa 24h'] = display_df['Baixa 24h'].apply(lambda x: f"${x:.6f}")
        
        return display_df
    
    def render_footer(self, df):
        """Renderizar rodapé com informações adicionais"""
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📊 KuCoin Crypto Dashboard**  
            Dashboard profissional para análise de criptomoedas
            """)
        
        with col2:
            st.markdown("""
            **🔄 Dados em Tempo Real**  
            Atualizações automáticas via API da KuCoin
            """)
        
        with col3:
            # Botão de download
            if not df.empty:
                DataExporter.create_download_button(
                    df.head(100), 
                    f"kucoin_crypto_data_{int(time.time())}.csv",
                    "📥 Baixar Top 100 CSV"
                )
        
        st.markdown("""
        <div style="text-align: center; padding: 1rem; color: #666; font-size: 0.8rem;">
            ⚠️ <strong>Aviso:</strong> Este dashboard é apenas para fins informativos. 
            Não constitui aconselhamento financeiro ou recomendação de investimento.
            <br>
            📡 Dados fornecidos pela API pública da KuCoin | 
            🔄 Cache: 5 minutos | 
            ⚡ Auto-refresh disponível
        </div>
        """, unsafe_allow_html=True)
    
    def check_auto_refresh(self):
        """Verificar e executar auto-refresh se necessário"""
        if SessionManager.should_auto_refresh(30):  # 30 segundos
            st.rerun()
    
    def run(self):
        """Executar aplicação principal"""
        # Renderizar cabeçalho
        self.render_header()
        
        # Carregar dados
        df = self.load_data()
        
        # Renderizar sidebar e obter filtros
        filters = self.render_sidebar(df)
        
        # Aplicar filtros
        df_filtered = UIComponents.apply_filters(df, filters) if not df.empty else df
        
        # Verificar se há dados
        if df.empty:
            st.error("❌ Não foi possível carregar dados da KuCoin")
            st.info("💡 Verifique sua conexão com a internet e tente novamente.")
            
            # Botão para tentar novamente
            if st.button("🔄 Tentar Novamente", type="primary"):
                CacheManager.clear_all_cache()
                st.rerun()
            
            return
        
        # Renderizar métricas
        self.render_metrics_overview(df_filtered)
        
        # Renderizar abas principais
        self.render_main_tabs(df_filtered)
        
        # Mostrar métricas de performance (opcional)
        if st.sidebar.checkbox("⚡ Mostrar Performance", value=False):
            PerformanceMonitor.show_performance_metrics()
        
        # Renderizar rodapé
        self.render_footer(df_filtered)
        
        # Verificar auto-refresh
        if st.session_state.get('auto_refresh', False):
            time.sleep(1)  # Pequena pausa para evitar loops muito rápidos
            if SessionManager.should_auto_refresh(30):
                st.rerun()


def main():
    """Função principal"""
    try:
        # Inicializar e executar aplicação
        app = CryptoDashboardApp()
        app.run()
        
    except Exception as e:
        st.error("❌ Erro crítico na aplicação")
        ErrorHandler.handle_api_error(e, "aplicação principal")
        
        # Opção de restart
        if st.button("🔄 Reiniciar Aplicação", type="primary"):
            st.cache_data.clear()
            st.rerun()


if __name__ == "__main__":
    main()
