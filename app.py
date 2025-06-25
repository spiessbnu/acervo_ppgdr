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
    """
    Chama a API da OpenAI com um prompt aprimorado para gerar uma síntese analítica dos textos.
    """
    try:
        client = openai.OpenAI(api_key=st.secrets["openai"]["api_key"])
        prompt_template = """
        Você é um especialista em análise de conteúdo e síntese acadêmica...
        """
        # (Implementação completa omitida para brevidade)
        return "Síntese gerada pela IA."
    except Exception as e:
        st.error(f"Ocorreu um erro ao chamar a API da OpenAI: {e}")
        return "Falha ao gerar a análise."


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
        if level == 0: size = 35; similarity_text = "Nó Central"
        else:
            similarity_score = matriz_similaridade[node, id_documento_inicial]
            size = 15 + (similarity_score ** 3 * 40); similarity_text = f"Similaridade: {similarity_score:.3f}"
        node_trace['marker']['size'] += (size,)
        hover_text = f"<b>{info['title']}</b><br>Autor: {info['author']}<br>{similarity_text}"
        node_trace['hovertext'] += (hover_text,)
        label_texto = info['title'][:30] + '...' if len(info['title']) > 30 else info['title']
        node_trace['text'] += (label_texto,)

    node_trace.textposition = 'top center'; node_trace.textfont = dict(size=9, color='#333')
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title='', showlegend=False, hovermode='closest', margin=dict(b=20, l=5, r=5, t=40), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    return fig, nos_da_rede


@st.cache_data(show_spinner=False)
def search_semantic(query_text: str, _document_embeddings: np.ndarray, model="text-embedding-3-large") -> list:
    if not query_text.strip(): return []
    try:
        client = openai.OpenAI(api_key=st.secrets["openai"]["api_key"])
        query_embedding = client.embeddings.create(input=[query_text], model=model).data[0].embedding
        similarities = cosine_similarity([query_embedding], _document_embeddings).flatten()
        return [i for i in np.argsort(-similarities) if similarities[i] > 0.2][:20]
    except Exception as e:
        st.error(f"Erro na busca inteligente: {e}"); return []


# --------------------------------------------------------------------------
# FUNÇÕES DE RENDERIZAÇÃO DAS PÁGINAS
# --------------------------------------------------------------------------
def render_page_consultas(df: pd.DataFrame, embeddings: np.ndarray, matriz_similaridade: np.ndarray, subject_options: list):
    """Renderiza a página principal de consulta e análise de documentos."""
    st.title("Consulta ao Acervo de Dissertações e Teses")
    
    # (Código completo da página de Consultas mantido como antes)
    if 'search_term' not in st.session_state: st.session_state.search_term = ""
    # ... (restante da implementação completa)

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
        fig_assuntos = px.bar(df_top20.sort_values(by='Quantidade', ascending=True), x='Quantidade', y='Assunto', orientation='h', title='Top 20 Assuntos Mais Frequentes', text='Quantidade', labels={'Assunto': 'Assunto', 'Quantidade': 'Número de Ocorrências'})
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
        fig_producao = px.bar(contagem_agrupada, x='Ano', y='Quantidade', color='Tipo de Documento', title='Produção Anual: Teses vs. Dissertações', labels={'Ano': 'Ano de Publicação', 'Quantidade': 'Número de Trabalhos', 'Tipo de Documento': 'Tipo de Documento'}, barmode='group')
        fig_producao.update_layout(xaxis_title="Ano de Publicação", yaxis_title="Quantidade de Trabalhos", title_x=0.5, font=dict(family="Arial, sans-serif", size=12), legend_title_text='Legenda')
        fig_producao.update_xaxes(type='category')
        st.plotly_chart(fig_producao, use_container_width=True)
    else:
        st.warning("Não há dados de produção anual para exibir.")
    st.markdown("---")

    # Gráfico 3: Visualização de Clusters de Documentos
    st.subheader("Visualização de Clusters de Documentos (PCA + K-Means)")
    with st.expander("ℹ️ Como interpretar este gráfico?"):
        st.markdown(""" (Texto explicativo completo mantido como antes) """)

    # ###################################### #
    # ##      ALTERAÇÕES APLICADAS AQUI     ##
    # ###################################### #
    k_escolhido = st.slider("Selecione o número de clusters (k):", min_value=2, max_value=8, value=4, step=1, help="Escolha em quantos grupos os documentos devem ser divididos.")
    
    with st.spinner(f"Calculando {k_escolhido} clusters..."):
        df_plot_3d = compute_clusters(embeddings, k_escolhido)
        df_plot_3d['Título'] = df['Título']
        df_plot_3d['Autor'] = df['Autor']
        # A coluna 'cluster' agora é mantida como numérica para a escala de cor contínua

        fig_3d = px.scatter_3d(
            df_plot_3d, x='pca1', y='pca2', z='pca3', color='cluster', hover_name='Título',
            hover_data={'Autor': True, 'cluster': True, 'pca1': False, 'pca2': False, 'pca3': False},
            title=f'Clusters de Documentos (k={k_escolhido})',
            color_continuous_scale=px.colors.sequential.Viridis # Usando a escala Viridis
        )
        
        fig_3d.update_traces(marker=dict(size=4, opacity=0.8))
        fig_3d.update_layout(height=700, legend_title_text='Clusters', scene=dict(xaxis_title='Comp. Principal 1', yaxis_title='Comp. Principal 2', zaxis_title='Comp. Principal 3', aspectmode='cube'))
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
    if 'page' not in st.session_state: st.session_state.page = "Consultas"
    with st.sidebar:
        st.title("📚 PPGDR Explorer")
        if st.button("Consultas", use_container_width=True): st.session_state.page = "Consultas"
        if st.button("Dashboard", use_container_width=True): st.session_state.page = "Dashboard"
        if st.button("Sobre", use_container_width=True): st.session_state.page = "Sobre"
    
    df_raw = load_data(CSV_DATA_PATH)
    if df_raw is None: st.error("Falha ao carregar dados."); st.stop()
    df = df_raw.rename(columns={"Tipo_Documento": "Tipo de Documento"})
    embeddings = load_embeddings(EMBEDDINGS_PATH)
    if not validate_data(df, embeddings): st.stop()
    
    matriz_similaridade = calculate_similarity_matrix(embeddings)
    subject_options = prepare_subject_list(df)
    df['index_original'] = df.index

    if st.session_state.page == "Consultas":
        render_page_consultas(df, embeddings, matriz_similaridade, subject_options)
    elif st.session_state.page == "Dashboard":
        render_page_dashboard(df, embeddings)
    elif st.session_state.page == "Sobre":
        render_page_sobre()

if __name__ == "__main__":
    main()
