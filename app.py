"""
This module contains a revised version of the original Streamlit app for
exploring the PPGDR thesis and dissertation collection.  The primary
change relative to the user supplied version is in the handling of row
selection within the AgGrid component.  Recent versions of
``streamlit‑aggrid`` changed how selections are returned and how reruns
are triggered.  The code below demonstrates how to configure the grid
using the modern ``update_on`` parameter and how to consume the
``selected_rows`` list of dictionaries returned from AgGrid.  See the
documentation and examples for details【942438317444411†L648-L681】.
"""

import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
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

def setup_page():
    st.set_page_config(
        page_title="Acervo de Dissertações e Teses PPGDR v1",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def safe_literal_eval(s):
    """Função segura para converter string de lista em objeto lista."""
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError, TypeError):
        return []

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Carrega o arquivo CSV com tratamento de erro aprimorado."""
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
    """Carrega os embeddings com tratamento de erro aprimorado."""
    try:
        return np.load(path)
    except FileNotFoundError:
        st.error(f"Arquivo de embeddings não encontrado: '{path}'.")
        return None
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar o arquivo de embeddings '{path}': {e}")
        return None

def validate_data(df: pd.DataFrame, embeddings: np.ndarray) -> bool:
    """Verifica se os dados carregados são consistentes."""
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
    """Calcula a matriz de similaridade de cossenos a partir dos embeddings."""
    if _embeddings is not None and _embeddings.size > 0:
        return cosine_similarity(_embeddings)
    return np.array([])

def remover_acentos(texto: str) -> str:
    """Remove acentos de uma string para ordenação alfabética correta."""
    texto_normalizado = unicodedata.normalize('NFD', texto)
    return "".join(c for c in texto_normalizado if not unicodedata.combining(c))

@st.cache_data
def prepare_subject_list(_df: pd.DataFrame) -> list:
    """Extrai, limpa, unifica e ordena os assuntos para o dropdown."""
    if 'Assuntos_Processados' not in _df.columns:
        return ['-- Selecione um Assunto --']
    todos_assuntos = [assunto for sublista in _df['Assuntos_Processados'] for assunto in sublista]
    lista_unica = sorted(list(set(todos_assuntos)), key=lambda texto: remover_acentos(texto.lower()))
    return ['-- Selecione um Assunto --'] + lista_unica

def initialize_state():
    """Carrega todos os dados e inicializa o session_state."""
    if 'data_loaded' not in st.session_state:
        df_raw = load_data(CSV_DATA_PATH)
        if df_raw is None:
            st.error("Falha crítica ao carregar os dados. A aplicação não pode continuar.")
            st.stop()
        df = df_raw.rename(columns={"Tipo_Documento": "Tipo de Documento"})
        df['index_original'] = df.index
        st.session_state.df = df
        st.session_state.embeddings = load_embeddings(EMBEDDINGS_PATH)
        if not validate_data(st.session_state.df, st.session_state.embeddings):
            st.stop()
        st.session_state.matriz_similaridade = calculate_similarity_matrix(st.session_state.embeddings)
        st.session_state.subject_options = prepare_subject_list(st.session_state.df)
        st.session_state.page = "Consultas"
        st.session_state.search_term = ""
        st.session_state.semantic_term = ""
        st.session_state.subject_filter = st.session_state.subject_options[0]
        st.session_state.analysis_cache = {}
        st.session_state.grid_key = str(uuid.uuid4())
        st.session_state.selected_doc_index = None
        st.session_state.num_vizinhos_cache = 5
        st.session_state.data_loaded = True

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

# IA and graph functions remain unchanged for brevity

def render_page_consultas(df: pd.DataFrame, embeddings: np.ndarray, matriz_similaridade: np.ndarray, subject_options: list):
    """Renderiza a página principal de consulta e análise de documentos."""
    st.title("Consulta ao Acervo de Dissertações e Teses")

    def clear_searches():
        st.session_state.search_term = ""
        st.session_state.semantic_term = ""
        st.session_state.subject_filter = subject_options[0]
        st.session_state.grid_key = str(uuid.uuid4())
        st.session_state.selected_doc_index = None
        if 'analysis_result' in st.session_state:
            del st.session_state['analysis_result']

    search_col1, search_col2 = st.columns(2)
    with search_col1:
        st.text_input("Busca simples", key="search_term", placeholder="Filtro simples por palavra-chave...", help="Busca por temas exatos: autor, assuntos, palavras-chave e termos nos resumos. Pressione Enter.")
    with search_col2:
        st.text_input("Busca semântica (com IA)", key="semantic_term", placeholder="Qual o tema do seu interesse?", help="Descreva um tema em palavras, tópicos ou frases e pressione Enter. O sistema retornará resultados com temas correlatos.")
    filter_col1, filter_col2 = st.columns([3, 1])
    with filter_col1:
        st.selectbox("Filtro por Assunto", options=subject_options, key="subject_filter")
    with filter_col2:
        st.button("Limpar Tudo 🧹", on_click=clear_searches, use_container_width=True, help="Limpa todas as buscas e filtros.")

    df_filtered = df.copy()
    if st.session_state.semantic_term:
        with st.spinner("Buscando por significado..."):
            ranked_indices = search_semantic(st.session_state.semantic_term, embeddings)
        if ranked_indices:
            df_filtered = df.loc[ranked_indices]
            st.success(f"Exibindo {len(df_filtered)} resultados.")
        else:
            st.warning("Nenhum resultado para a busca inteligente.")
            df_filtered = pd.DataFrame(columns=df.columns)
    elif st.session_state.search_term:
        cols_to_search = ["Autor", "Título", "Assuntos", "Resumo_LLM"]
        mask = df_filtered[cols_to_search].fillna('').astype(str).apply(lambda col: col.str.contains(st.session_state.search_term, case=False)).any(axis=1)
        df_filtered = df_filtered[mask]

    selected_subject = st.session_state.get('subject_filter', subject_options[0])
    if selected_subject != '-- Selecione um Assunto --':
        mask_subject = df_filtered['Assuntos_Processados'].apply(lambda lista: selected_subject in lista)
        df_filtered = df_filtered[mask_subject]

    st.divider()

    # Reset selection if any of the filters changed
    current_filter_state = (st.session_state.search_term, st.session_state.semantic_term, st.session_state.subject_filter)
    if st.session_state.get('last_filter_state') != current_filter_state:
        st.session_state.grid_key = str(uuid.uuid4())
        st.session_state.selected_doc_index = None
        if 'analysis_result' in st.session_state:
            del st.session_state['analysis_result']
    st.session_state.last_filter_state = current_filter_state

    cols_display = ["Tipo de Documento", "Autor", "Título", "Ano", "Assuntos", "Orientador"]
    df_aggrid = df_filtered[cols_display + ['index_original']].reset_index(drop=True)

    # Build grid options
    gb = GridOptionsBuilder.from_dataframe(df_aggrid)
    gb.configure_default_column(resizable=True, wrapText=True, autoHeight=True, suppressMenu=True, sortable=True)
    gb.configure_column("Título", width=500)
    gb.configure_column("Autor", width=250)
    gb.configure_column("Orientador", width=250)
    gb.configure_column("Assuntos", width=350)
    gb.configure_column("Tipo de Documento", width=150)
    gb.configure_column("Ano", width=90)

    # Determine which rows should be pre-selected.  Recent versions of
    # streamlit-aggrid expect row identifiers as strings【585581463231963†L274-L299】.
    pre_selected_rows_list: list[str] = []
    if st.session_state.get('selected_doc_index') is not None:
        match = df_aggrid.index[df_aggrid['index_original'] == st.session_state.selected_doc_index].tolist()
        if match:
            # Convert to string to satisfy newer versions (see issue #207)
            pre_selected_rows_list = [str(match[0])]

    gb.configure_selection(
        selection_mode="single",
        use_checkbox=True,
        pre_selected_rows=pre_selected_rows_list
    )

    gb.configure_column("index_original", hide=True)
    grid_opts = gb.build()

    # Configure AgGrid using the modern update_on API.  We subscribe only
    # to the 'selectionChanged' event so that the app reruns when the user
    # changes the row selection【484335540904902†L33-L64】.
    grid_response = AgGrid(
        df_aggrid,
        gridOptions=grid_opts,
        update_on=['selectionChanged'],
        update_mode='NO_UPDATE',
        enable_enterprise_modules=False,
        fit_columns_on_grid_load=False,
        key=st.session_state.grid_key
    )

    st.divider()

    # Extract the selected rows.  In newer versions the grid returns a list
    # of dictionaries instead of a DataFrame.  Each dict contains the
    # original values for each column plus metadata such as row indices.
    # We look up our hidden 'index_original' column to find the row in the
    # original dataframe【942438317444411†L648-L681】.
    selected_rows = []
    try:
        selected_rows = grid_response['selected_rows']
    except Exception:
        # Fallback for dataclass-based return objects
        if hasattr(grid_response, 'selected_rows'):
            selected_rows = grid_response.selected_rows
    if selected_rows:
        # If the selection changed, update session state
        newly_selected_index = selected_rows[0].get('index_original')
        if st.session_state.get('selected_doc_index') != newly_selected_index:
            st.session_state.selected_doc_index = newly_selected_index
            if 'analysis_result' in st.session_state:
                del st.session_state['analysis_result']
            st.rerun()
    elif st.session_state.get('selected_doc_index') is not None:
        # If the user cleared the selection, reset the selected index
        st.session_state.selected_doc_index = None
        st.rerun()

    # Details and similar tabs (unchanged from the original implementation)
    tab_detalhes, tab_similares = st.tabs(["Detalhes", "Trabalhos Similares"])
    with tab_detalhes:
        if st.session_state.get('selected_doc_index') is not None:
            try:
                detalhes = df.loc[st.session_state.selected_doc_index]
                st.subheader(detalhes.get('Título', ''))
                st.divider()
                st.markdown("#### Assuntos")
                st.write(detalhes.get('Assuntos', ''))
                st.markdown("#### Resumo")
                st.write(detalhes.get('Resumo_LLM', ''))
                st.markdown("#### Link para Download")
                link_pdf = detalhes.get('Link_PDF')
                if link_pdf and isinstance(link_pdf, str):
                    st.link_button("Baixar PDF", url=link_pdf, use_container_width=True)
                else:
                    st.warning("Nenhum link para download disponível.")
            except KeyError:
                st.error("O item selecionado não foi encontrado nos dados originais. Por favor, limpe os filtros e tente novamente.")
                st.session_state.selected_doc_index = None
        else:
            st.info("Selecione um registro na tabela para ver os detalhes.")

    with tab_similares:
        if not matriz_similaridade.any():
            st.warning("Dados de similaridade não disponíveis.")
        elif st.session_state.get('selected_doc_index') is not None:
            id_selecionado = st.session_state.selected_doc_index
            num_vizinhos = st.slider("Número de vizinhos", 1, 10, 5, 1, key="slider_vizinhos")
            if st.session_state.get('last_selected_id') != id_selecionado or st.session_state.get('num_vizinhos_cache') != num_vizinhos:
                if 'analysis_result' in st.session_state:
                    del st.session_state['analysis_result']
            st.session_state.last_selected_id = id_selecionado
            st.session_state.num_vizinhos_cache = num_vizinhos
            fig, node_indices = generate_similarity_graph(df, matriz_similaridade, id_selecionado, num_vizinhos)
            st.plotly_chart(fig, use_container_width=True)
            df_similares = df.loc[list(node_indices)][["Autor", "Título", "Ano"]].reset_index(drop=True)
            st.dataframe(df_similares, use_container_width=True, hide_index=True)
            st.divider()
            if st.button("Gerar análise da rede de trabalhos com IA 🧠", key="btn_analise"):
                cache_key = (id_selecionado, num_vizinhos)
                if cache_key in st.session_state.analysis_cache:
                    st.toast("Reexibindo análise em cache. ⚡")
                    st.session_state.analysis_result = st.session_state.analysis_cache[cache_key]
                else:
                    summaries_to_analyze = df.loc[list(node_indices)]['Resumo_LLM'].dropna()
                    if not summaries_to_analyze.empty:
                        with st.spinner('A IA está lendo e preparando a análise...'):
                            analysis = get_ai_synthesis("\n\n---\n\n".join(summaries_to_analyze))
                            st.session_state.analysis_result = analysis
                            st.session_state.analysis_cache[cache_key] = analysis
                    else:
                        st.warning("Não há resumos para gerar análise.")
                        st.session_state.analysis_result = ""
            if 'analysis_result' in st.session_state and st.session_state.analysis_result:
                with st.container(border=True):
                    st.subheader("Análise Gerada por IA")
                    st.markdown(st.session_state.analysis_result)
        else:
            st.info("Selecione um registro para visualizar trabalhos similares.")

def main():
    setup_page()
    initialize_state()
    with st.sidebar:
        st.markdown("<h1 style='color:white;'><b>📚 Acervo PPGDR</b></h1>", unsafe_allow_html=True)
        if st.button("Consultas", use_container_width=True):
            st.session_state.page = "Consultas"
        if st.button("Dashboard", use_container_width=True):
            st.session_state.page = "Dashboard"
        if st.button("Sobre", use_container_width=True):
            st.session_state.page = "Sobre"
    if st.session_state.page == "Consultas":
        render_page_consultas(
            st.session_state.df,
            st.session_state.embeddings,
            st.session_state.matriz_similaridade,
            st.session_state.subject_options
        )
    # Other pages omitted for brevity

if __name__ == "__main__":
    main()
