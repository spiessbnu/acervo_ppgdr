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
        page_title="Acervo de Dissertações e Teses PPGDR v1",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# --------------------------------------------------------------------------
# FUNÇÕES DE CARREGAMENTO E PROCESSAMENTO
# --------------------------------------------------------------------------
def safe_literal_eval(s):
    """Função segura para converter string de lista em objeto lista."""
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError, TypeError):
        return []

@st.cache_data
def load_data(path: str) -> pd.DataFrame | None:
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
def load_embeddings(path: str) -> np.ndarray | None:
    """Carrega os embeddings com tratamento de erro aprimorado."""
    try:
        return np.load(path)
    except FileNotFoundError:
        st.error(f"Arquivo de embeddings não encontrado: '{path}'.")
        return None
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar o arquivo de embeddings '{path}': {e}")
        return None

def validate_data(
    df: pd.DataFrame | None,
    embeddings: np.ndarray | None,
    require_embeddings: bool = True
) -> bool:
    """Verifica se os dados carregados são consistentes."""
    if df is None:
        return False

    required_cols = ['Título', 'Autor', 'Assuntos_Lista', 'Resumo_LLM', 'Ano', 'Tipo de Documento']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Erro de Integridade: Coluna(s) necessária(s) não encontrada(s): {', '.join(missing_cols)}")
        return False

    if require_embeddings:
        if embeddings is None:
            st.error("Embeddings não carregados.")
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
# FUNÇÃO DE IA PARA GERAR SÍNTESE (VERSÃO APRIMORADA)
# --------------------------------------------------------------------------
def get_ai_synthesis(summaries: str) -> str:
    """
    Chama a API da OpenAI com um prompt aprimorado para gerar uma síntese analítica dos textos.
    """
    try:
        client = openai.OpenAI(api_key=st.secrets["openai"]["api_key"])
        prompt_template = """
        Você é um especialista em análise de conteúdo e síntese acadêmica.
        Sua missão é analisar o conjunto de resumos de trabalhos acadêmicos fornecido abaixo.
        Leia todos os textos e identifique as conexões, os padrões e os temas centrais que os unem.
        Não resuma cada trabalho individualmente. Em vez disso, crie uma análise unificada que revele o panorama geral da pesquisa.

        CONTEXTO (Resumos a serem analisados):
        ---
        {summaries}
        ---

        Sua resposta deve seguir rigorosamente o seguinte formato, sem adicionar nenhuma introdução, saudação ou texto extra:

        **Síntese Analítica:**
        [Aqui, escreva um parágrafo denso e analítico que conecte as ideias principais dos textos. Destaque as convergências, possíveis divergências e a contribuição coletiva do conjunto para o campo de estudo.]

        **Temas Principais:**
        [Aqui, liste de 3 a 5 dos temas mais proeminentes e recorrentes encontrados nos textos, em formato de lista com marcadores. Seja conciso.]
        """
        prompt = prompt_template.format(summaries=summaries)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você é um especialista em análise de conteúdo e síntese acadêmica. Responda em português do Brasil."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Ocorreu um erro ao chamar a API da OpenAI: {e}")
        return "Falha ao gerar a análise. Verifique a configuração da chave de API ou se o serviço está disponível."

def generate_similarity_graph(df, matriz_similaridade, id_documento_inicial, num_vizinhos):
    """Gera um grafo de similaridade e retorna a figura e os IDs dos nós."""
    nos_da_rede = {id_documento_inicial}
    vizinhos_l1 = np.argsort(matriz_similaridade[id_documento_inicial])[-num_vizinhos-1:-1][::-1]
    nos_da_rede.update(vizinhos_l1)

    G = nx.Graph()
    for node_id in nos_da_rede:
        node_info = df.iloc[node_id]
        level = 0 if node_id == id_documento_inicial else 1
        G.add_node(node_id, title=node_info['Título'], author=node_info['Autor'], level=level)

    for vizinho_id in vizinhos_l1:
        similaridade = matriz_similaridade[id_documento_inicial, vizinho_id]
        G.add_edge(id_documento_inicial, vizinho_id, weight=similaridade)

    pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)

    edge_trace = go.Scatter(x=[], y=[], line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')
    for edge in G.edges():
        x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None); edge_trace['y'] += (y0, y1, None)

    node_trace = go.Scatter(x=[], y=[], mode='markers+text', text=[], hovertext=[], hovertemplate="%{hovertext}", marker=dict(color=[], size=[], line_width=2))
    cores_niveis = {0: 'crimson', 1: 'royalblue'}
    for node in G.nodes():
        x, y = pos[node]; info = G.nodes[node]; level = info['level']
        node_trace['x'] += (x,); node_trace['y'] += (y,)
        node_trace['marker']['color'] += (cores_niveis[level],)
        if level == 0:
            size = 35; similarity_text = "Nó Central"
        else:
            similarity_score = matriz_similaridade[node, id_documento_inicial]
            size = 15 + (similarity_score ** 3 * 40); similarity_text = f"Similaridade: {similarity_score:.3f}"
        node_trace['marker']['size'] += (size,)
        hover_text = f"<b>{info['title']}</b><br>Autor: {info['author']}<br>{similarity_text}"
        node_trace['hovertext'] += (hover_text,)
        label_texto = info['title'][:30] + '...' if len(info['title']) > 30 else info['title']
        node_trace['text'] += (label_texto,)

    node_trace.textposition = 'top center'; node_trace.textfont = dict(size=9, color='#333')
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title='',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    return fig, nos_da_rede

@st.cache_data(show_spinner=False)
def search_semantic(query_text: str, _document_embeddings: np.ndarray, model="text-embedding-3-large") -> list:
    if not query_text.strip():
        return []
    try:
        client = openai.OpenAI(api_key=st.secrets["openai"]["api_key"])
        query_embedding = client.embeddings.create(input=[query_text], model=model).data[0].embedding
        similarities = cosine_similarity([query_embedding], _document_embeddings).flatten()
        return [i for i in np.argsort(-similarities) if similarities[i] > 0.2][:20]
    except Exception as e:
        st.error(f"Erro na busca inteligente: {e}")
        return []

# --------------------------------------------------------------------------
# FUNÇÕES DE RENDERIZAÇÃO DAS PÁGINAS
# --------------------------------------------------------------------------
def render_page_consultas(df: pd.DataFrame, embeddings: np.ndarray, matriz_similaridade: np.ndarray, subject_options: list):
    """Renderiza a página principal de consulta e análise de documentos."""
    st.title("Consulta ao Acervo de Dissertações e Teses")

    # Estado inicial
    if 'search_term' not in st.session_state: st.session_state.search_term = ""
    if 'semantic_term' not in st.session_state: st.session_state.semantic_term = ""
    if 'subject_filter' not in st.session_state: st.session_state.subject_filter = subject_options[0]
    if 'analysis_cache' not in st.session_state: st.session_state.analysis_cache = {}
    if 'grid_key' not in st.session_state: st.session_state.grid_key = str(uuid.uuid4())
    if 'selected_id' not in st.session_state: st.session_state.selected_id = None
    if 'num_vizinhos_cache' not in st.session_state: st.session_state.num_vizinhos_cache = None

    def clear_searches():
        st.session_state.search_term = ""
        st.session_state.semantic_term = ""
        st.session_state.subject_filter = subject_options[0]
        st.session_state.grid_key = str(uuid.uuid4())
        st.session_state.pop('analysis_result', None)
        st.session_state.selected_id = None

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

    # Filtragem
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
        cols_all = set(df_filtered.columns)
        cols_to_search = [c for c in ["Autor", "Título", "Assuntos", "Resumo_LLM"] if c in cols_all]
        if cols_to_search:
            mask = df_filtered[cols_to_search].fillna('').astype(str).apply(
                lambda col: col.str.contains(st.session_state.search_term, case=False)
            ).any(axis=1)
            df_filtered = df_filtered[mask]

    selected_subject = st.session_state.get('subject_filter', subject_options[0])
    if selected_subject != '-- Selecione um Assunto --':
        mask_subject = df_filtered['Assuntos_Processados'].apply(lambda lista: selected_subject in lista)
        df_filtered = df_filtered[mask_subject]

    st.divider()

    # Reset de grid/seleção quando filtros mudam
    current_filter_state = (st.session_state.search_term, st.session_state.semantic_term, st.session_state.subject_filter)
    if st.session_state.get('last_filter_state') != current_filter_state:
        st.session_state.grid_key = str(uuid.uuid4())
        st.session_state.pop('analysis_result', None)
        st.session_state.selected_id = None
    st.session_state.last_filter_state = current_filter_state

    # Tabela (AgGrid)
    cols_display = ["Tipo de Documento", "Autor", "Título", "Ano", "Assuntos", "Orientador"]
