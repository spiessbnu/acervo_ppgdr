# --------------------------------------------------------------------------
# BIBLIOTECAS NECESSÁRIAS
# --------------------------------------------------------------------------
import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import openai
import uuid
import unicodedata
import ast

# --------------------------------------------------------------------------
# CONFIGURAÇÃO DE ARQUIVOS E CONSTANTES
# --------------------------------------------------------------------------
CSV_DATA_PATH = "dados_finais_com_resumo_llm.csv"
EMBEDDINGS_PATH = "openai_embeddings_concatenado_large.npy"


# --------------------------------------------------------------------------
# FUNÇÃO 1: Configuração da página do Streamlit
# --------------------------------------------------------------------------
def setup_page():
    st.set_page_config(
        page_title="Dissertações e Teses PPGDR v1",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# --------------------------------------------------------------------------
# FUNÇÕES DE CARREGAMENTO E PROCESSAMENTO
# --------------------------------------------------------------------------
def safe_literal_eval(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError, TypeError):
        return []

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if 'Assuntos_Lista' in df.columns:
            df['Assuntos_Processados'] = df['Assuntos_Lista'].apply(safe_literal_eval)
        else:
            df['Assuntos_Processados'] = [[] for _ in range(len(df))]
        return df
    except FileNotFoundError:
        st.error(f"Arquivo de dados não encontrado: '{path}'.")
        return None
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar o arquivo CSV '{path}': {e}")
        return None


@st.cache_data
def load_embeddings(path: str) -> np.ndarray:
    try:
        return np.load(path)
    except FileNotFoundError:
        st.error(f"Arquivo de embeddings não encontrado: '{path}'.")
        return None
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar o arquivo de embeddings '{path}': {e}")
        return None


def validate_data(df: pd.DataFrame, embeddings: np.ndarray) -> bool:
    if df is None or embeddings is None:
        return False
    required_cols = ['Título', 'Autor', 'Assuntos_Lista', 'Resumo_LLM', 'Ano', 'Tipo de Documento']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Erro de Integridade: Coluna(s) necessária(s) não encontrada(s): {', '.join(missing_cols)}")
        return False
    if len(df) != len(embeddings):
        st.error(f"Erro de Integridade: Incompatibilidade entre CSV ({len(df)}) e embeddings ({len(embeddings)}).")
        return False
    st.toast("Arquivos de dados carregados e validados!", icon="✅")
    return True


@st.cache_data
def calculate_similarity_matrix(_embeddings: np.ndarray) -> np.ndarray:
    if _embeddings is not None and _embeddings.size > 0:
        return cosine_similarity(_embeddings)
    return np.array([])


def remover_acentos(texto: str) -> str:
    texto_normalizado = unicodedata.normalize('NFD', texto)
    return "".join(c for c in texto_normalizado if not unicodedata.combining(c))


@st.cache_data
def prepare_subject_list(_df: pd.DataFrame) -> list:
    if 'Assuntos_Processados' not in _df.columns:
        return ['-- Selecione um Assunto --']
    todos_assuntos = [assunto for sublista in _df['Assuntos_Processados'] for assunto in sublista]
    lista_unica = sorted(list(set(todos_assuntos)), key=lambda texto: remover_acentos(texto.lower()))
    return ['-- Selecione um Assunto --'] + lista_unica


# --------------------------------------------------------------------------
# FUNÇÕES DE COMPUTAÇÃO PARA O DASHBOARD (COM CACHE)
# --------------------------------------------------------------------------
@st.cache_data
def compute_clusters(_embeddings, k):
    """Executa PCA e K-Means e retorna os dados para plotagem."""
    pca = PCA(n_components=3, random_state=42)
    embeddings_3d = pca.fit_transform(_embeddings)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(_embeddings)
    df_plot = pd.DataFrame(embeddings_3d, columns=['pca1', 'pca2', 'pca3'])
    df_plot['cluster'] = cluster_labels
    return df_plot


# --------------------------------------------------------------------------
# FUNÇÕES DE IA E GRAFOS
# --------------------------------------------------------------------------
def get_ai_synthesis(summaries: str) -> str:
    # Implementação completa omitida para brevidade
    return "Síntese gerada pela IA."

def generate_similarity_graph(df, matriz_similaridade, id_documento_inicial, num_vizinhos):
    # Implementação completa omitida para brevidade
    return go.Figure(), set()

@st.cache_data(show_spinner=False)
def search_semantic(query_text: str, _document_embeddings: np.ndarray, model="text-embedding-3-large") -> list:
    # Implementação completa omitida para brevidade
    return []


# --------------------------------------------------------------------------
# FUNÇÕES DE RENDERIZAÇÃO DAS PÁGINAS
# --------------------------------------------------------------------------
def render_page_consultas(df: pd.DataFrame, embeddings: np.ndarray, matriz_similaridade: np.ndarray, subject_options: list):
    """Renderiza a página principal de consulta e análise de documentos."""
    # A implementação completa desta função é mantida como nas versões anteriores.
    # Para brevidade, o código detalhado não será repetido aqui.
    st.title("Consulta ao Acervo de Dissertações e Teses")
    st.info("A funcionalidade completa da página de consultas está disponível aqui.")


def render_page_dashboard(df: pd.DataFrame, embeddings: np.ndarray):
    """Renderiza a página do Dashboard com visualizações sobre os dados."""
    st.title("Dashboard de Análise do Acervo")
    st.markdown("---")
    
    # Gráfico 1: Top 20 Assuntos Mais Frequentes
    st.subheader("Top 20 Assuntos Mais Frequentes")
    todos_assuntos = [assunto for sublista in df['Assuntos_Processados'] for assunto in sublista]
    if todos_assuntos:
        contador_assuntos = Counter(todos_assuntos)
        top_20_assuntos = contador_assuntos.most_common(20)
        df_top20 = pd.DataFrame(top_20_assuntos, columns=['Assunto', 'Quantidade'])
        fig_assuntos = px.bar(
            df_top20.sort_values(by='Quantidade', ascending=True),
            x='Quantidade', y='Assunto', orientation='h', title='Top 20 Assuntos Mais Frequentes',
            text='Quantidade', labels={'Assunto': 'Assunto', 'Quantidade': 'Número de Ocorrências'}
        )
        fig_assuntos.update_traces(marker_color='#1f77b4', textposition='outside')
        fig_assuntos.update_layout(yaxis=dict(tickmode='linear'), xaxis_title="Número de Ocorrências", yaxis_title=None, margin=dict(l=200, r=20, t=50, b=50), title_x=0.5)
        st.plotly_chart(fig_assuntos, use_container_width=True)
    else:
        st.warning("Não há dados de assuntos para exibir.")
    st.markdown("---")

    # Gráfico 2: Produção Anual de Teses e Dissertações
    st.subheader("Produção Anual por Tipo de Documento")
    contagem_agrupada = df.groupby(['Ano', 'Tipo de Documento']).size().reset_index(name='Quantidade').sort_values('Ano')
    if not contagem_agrupada.empty:
        fig_producao = px.bar(
            contagem_agrupada, x='Ano', y='Quantidade', color='Tipo de Documento',
            title='Produção Anual: Teses vs. Dissertações',
            labels={'Ano': 'Ano de Publicação', 'Quantidade': 'Número de Trabalhos', 'Tipo de Documento': 'Tipo de Documento'},
            barmode='group'
        )
        fig_producao.update_layout(xaxis_title="Ano de Publicação", yaxis_title="Quantidade de Trabalhos", title_x=0.5, font=dict(family="Arial, sans-serif", size=12), legend_title_text='Legenda')
        fig_producao.update_xaxes(type='category')
        st.plotly_chart(fig_producao, use_container_width=True)
    else:
        st.warning("Não há dados de produção anual para exibir.")
    st.markdown("---")

    # --- Gráfico 3: Visualização de Clusters de Documentos ---
    st.subheader("Visualização de Clusters de Documentos (PCA + K-Means)")

    # Adicionando o texto explicativo dentro de um expander
    with st.expander("ℹ️ Como interpretar este gráfico?"):
        st.markdown("""
        Este gráfico organiza todos os documentos do acervo em um espaço 3D, agrupando-os por similaridade de conteúdo.

        - **a) O que os eixos (PCA) representam?**
          Os eixos `Componente Principal 1, 2 e 3` são o resultado de uma técnica de compressão de dados chamada PCA. Eles reduzem as centenas de dimensões semânticas de um texto a apenas três, para que possamos visualizá-los. **Documentos que estão próximos neste espaço 3D são mais similares em conteúdo** do que aqueles que estão distantes.

        - **b) O que os clusters (cores) representam?**
          Cada cor representa um "cluster", ou seja, um **grupo de documentos que o algoritmo identificou como sendo muito parecidos entre si**. É provável que os documentos de um mesmo cluster compartilhem os mesmos temas, conceitos ou abordagens.

        - **c) Como interpretar o gráfico?**
          Procure por grupos de cores (clusters) que estão densos e bem separados uns dos outros, pois isso indica tópicos distintos no acervo. Passe o mouse sobre um ponto para ver o título e o autor do trabalho, ajudando a entender o tema daquele cluster.

        - **d) Como interagir com o gráfico?**
          O gráfico é totalmente interativo:
          - **Clique e arraste** para rotacionar.
          - Use a **roda do mouse** para aplicar zoom.
          - **Clique nos itens da legenda** à direita para ativar ou desativar a visualização de clusters específicos. Isso é útil para focar a análise em grupos de seu interesse.
        """)
    
    k_escolhido = st.slider("Selecione o número de clusters (k):", min_value=2, max_value=15, value=3, step=1, help="Escolha em quantos grupos os documentos devem ser divididos.")
    with st.spinner(f"Calculando {k_escolhido} clusters..."):
        df_plot_3d = compute_clusters(embeddings, k_escolhido)
        df_plot_3d['Título'] = df['Título']
        df_plot_3d['Autor'] = df['Autor']
        df_plot_3d['cluster'] = df_plot_3d['cluster'].astype(str)
        fig_3d = px.scatter_3d(
            df_plot_3d, x='pca1', y='pca2', z='pca3', color='cluster', hover_name='Título',
            hover_data={'Autor': True, 'cluster': True, 'pca1': False, 'pca2': False, 'pca3': False},
            title=f'Clusters de Documentos (k={k_escolhido})'
        )
        fig_3d.update_traces(marker=dict(size=4, opacity=0.8))
        fig_3d.update_layout(legend_title_text='Clusters', scene=dict(xaxis_title='Comp. Principal 1', yaxis_title='Comp. Principal 2', zaxis_title='Comp. Principal 3'))
        st.plotly_chart(fig_3d, use_container_width=True)

def render_page_sobre():
    st.title("Sobre o Projeto")
    st.info("🚧 EM CONSTRUÇÃO 🚧")

# --------------------------------------------------------------------------
# FUNÇÃO PRINCIPAL DO APLICATIVO (ROTEADOR)
# --------------------------------------------------------------------------
def main():
    setup_page()

    st.markdown("""<style>[data-testid="stSidebar"] {background-color: #0F5EDD;}</style>""", unsafe_allow_html=True)

    if 'page' not in st.session_state:
        st.session_state.page = "Consultas"

    with st.sidebar:
        st.title("📚 PPGDR Explorer")
        if st.button("Consultas", use_container_width=True):
            st.session_state.page = "Consultas"
        if st.button("Dashboard", use_container_width=True):
            st.session_state.page = "Dashboard"
        if st.button("Sobre", use_container_width=True):
            st.session_state.page = "Sobre"
    
    df_raw = load_data(CSV_DATA_PATH)
    if df_raw is not None:
        df = df_raw.rename(columns={"Tipo_Documento": "Tipo de Documento"})
    else:
        st.error("Falha ao carregar os dados. A aplicação não pode continuar.")
        st.stop()
        
    embeddings = load_embeddings(EMBEDDINGS_PATH)

    if not validate_data(df, embeddings):
        st.warning("A aplicação não pode continuar devido a erros nos dados de entrada.")
        st.stop()
    
    matriz_similaridade = calculate_similarity_matrix(embeddings)
    subject_options = prepare_subject_list(df)
    df['index_original'] = df.index

    # --- ROTEAMENTO DE PÁGINAS ---
    if st.session_state.page == "Consultas":
        render_page_consultas(df, embeddings, matriz_similaridade, subject_options)
    elif st.session_state.page == "Dashboard":
        # ############# #
        # ## CORREÇÃO ##
        # ############# #
        render_page_dashboard(df, embeddings) # Passando 'embeddings' como segundo argumento
    elif st.session_state.page == "Sobre":
        render_page_sobre()

# --------------------------------------------------------------------------
# Ponto de entrada do script
# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
