# --------------------------------------------------------------------------
# BIBLIOTECAS NECESSÁRIAS
# --------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import ast
from collections import Counter

from st_aggrid import (
    AgGrid,
    GridOptionsBuilder,
    GridUpdateMode,
    DataReturnMode,
)
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import openai
import uuid

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
# FUNÇÕES DE CARREGAMENTO E PROCESSAMENTO (sem alterações)
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
        if 'index_original' not in df.columns:
            df['index_original'] = df.index
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
    if df is None or embeddings is None: return False
    required_cols = ['Título', 'Autor', 'Resumo_LLM', 'Ano', 'Tipo de Documento']
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
    if 'Assuntos_Processados' not in _df.columns: return ['-- Selecione um Assunto --']
    todos_assuntos = [assunto for sublista in _df['Assuntos_Processados'] for assunto in sublista]
    lista_unica = sorted(list(set(todos_assuntos)), key=lambda texto: remover_acentos(texto.lower()))
    return ['-- Selecione um Assunto --'] + lista_unica

@st.cache_data
def compute_clusters(_embeddings, k):
    pca = PCA(n_components=3, random_state=42)
    embeddings_3d = pca.fit_transform(_embeddings)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(_embeddings)
    df_plot = pd.DataFrame(embeddings_3d, columns=['pca1', 'pca2', 'pca3'])
    df_plot['cluster'] = cluster_labels
    return df_plot

def get_ai_synthesis(summaries: str) -> str:
    try:
        client = openai.OpenAI(api_key=st.secrets["openai"]["api_key"])
        prompt_template = """
        Você é um especialista em análise de conteúdo e síntese acadêmica.
        Sua missão é analisar o conjunto de resumos de trabalhos acadêmicos fornecido abaixo e identificar conexões, padrões e temas centrais.
        CONTEXTO DOS RESUMOS:
        ---
        {summaries}
        ---
        **Síntese Analítica:**
        [parágrafo analítico]
        **Temas Principais:**
        - [tema 1]
        - [tema 2]
        - [tema 3-5]
        """
        prompt = prompt_template.format(summaries=summaries)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você é um especialista em análise de conteúdo. Responda em português do Brasil."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Erro na API da OpenAI: {e}"); return "Falha ao gerar a análise."

def generate_similarity_graph(df, matriz_similaridade, id_documento_inicial, num_vizinhos):
    nos_da_rede = {id_documento_inicial}; vizinhos_l1 = np.argsort(matriz_similaridade[id_documento_inicial])[-num_vizinhos-1:-1][::-1]
    nos_da_rede.update(vizinhos_l1)
    G = nx.Graph()
    for node_id in nos_da_rede:
        node_info = df.iloc[node_id]; level = 0 if node_id == id_documento_inicial else 1
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
# PÁGINAS (Refatoradas para o novo fluxo)
# --------------------------------------------------------------------------

def render_details_view(df: pd.DataFrame, matriz_similaridade: np.ndarray):
    def go_back_to_grid():
        st.session_state.view_mode = 'grid'

    st.button("⬅️ Voltar para a Tabela", on_click=go_back_to_grid)
    
    selected_item = st.session_state.get('selected_item_cache', pd.DataFrame())

    if not selected_item.empty:
        idx_original = selected_item.iloc[0]['index_original']
        detalhes = df.loc[idx_original]

        st.title(detalhes.get('Título', ''))
        st.caption(f"**Autor(a):** {detalhes.get('Autor', '')} | **Ano:** {detalhes.get('Ano', '')} | **Tipo:** {detalhes.get('Tipo de Documento', '')}")
        st.divider()

        st.subheader("Resumo")
        st.write(detalhes.get('Resumo_LLM', ''))
        st.subheader("Assuntos"); st.write(', '.join(detalhes.get('Assuntos_Processados', [])))
        
        link_pdf = detalhes.get('Link_PDF')
        if link_pdf and isinstance(link_pdf, str) and 'http' in link_pdf:
            st.link_button("Baixar PDF do Documento", url=link_pdf, use_container_width=True)
        else:
            st.warning("Nenhum link para download disponível.")
        
        st.divider()

        st.subheader("Trabalhos Similares")
        id_selecionado = idx_original
        num_vizinhos = st.slider("Número de trabalhos similares para exibir", 1, 10, 5, 1, key=f"slider_vizinhos_{id_selecionado}")
        fig, node_indices = generate_similarity_graph(df, matriz_similaridade, id_selecionado, num_vizinhos)
        st.plotly_chart(fig, use_container_width=True)
        df_similares = df.loc[list(node_indices)][["Autor", "Título", "Ano"]].reset_index(drop=True)
        st.dataframe(df_similares, use_container_width=True, hide_index=True)
        st.divider()
        
        if st.button("Gerar análise da rede de trabalhos com IA 🧠", key=f"btn_analise_{id_selecionado}"):
            cache_key = (id_selecionado, num_vizinhos)
            if cache_key in st.session_state.analysis_cache:
                analysis = st.session_state.analysis_cache[cache_key]; st.toast("Reexibindo análise em cache. ⚡")
            else:
                summaries_to_analyze = df.loc[list(node_indices)]['Resumo_LLM'].dropna()
                if not summaries_to_analyze.empty:
                    with st.spinner('A IA está lendo e preparando a análise...'):
                        analysis = get_ai_synthesis("\n\n---\n\n".join(summaries_to_analyze))
                        st.session_state.analysis_cache[cache_key] = analysis
                else:
                    analysis = "Não há resumos disponíveis para gerar análise."; st.warning(analysis)
            with st.container(border=True):
                st.subheader("Análise Gerada por IA"); st.markdown(analysis)
    else:
        st.error("Nenhum item selecionado. Por favor, volte à tabela e selecione um item.")

def render_grid_view(df: pd.DataFrame, subject_options: list):
    st.title("Consulta ao Acervo de Dissertações e Teses")

    def clear_all_filters():
        st.session_state.search_term = ""
        st.session_state.semantic_term = ""
        st.session_state.subject_filter = subject_options[0]
        st.session_state.selected_item_cache = pd.DataFrame() # Também limpa o item selecionado
        if 'semantic_query_input' in st.session_state:
            st.session_state.semantic_query_input = ""

    search_col1, search_col2 = st.columns(2)
    with search_col1:
        st.text_input("Busca simples por palavra-chave", key="search_term", placeholder="Filtre por autor, título, resumo...")
    with search_col2:
        with st.form(key="semantic_form"):
            semantic_input = st.text_input("Busca semântica (com IA)", placeholder="Qual o tema do seu interesse?", key="semantic_query_input")
            semantic_submitted = st.form_submit_button("Buscar com IA 🧠")
            if semantic_submitted and semantic_input:
                st.session_state.selected_item_cache = pd.DataFrame()
                st.session_state.semantic_term = semantic_input
                st.session_state.search_term = ""
                st.session_state.subject_filter = subject_options[0]
                st.rerun()

    filter_col1, filter_col2 = st.columns([3, 1])
    with filter_col1:
        st.selectbox("Filtro por Assunto", options=subject_options, key="subject_filter")
    with filter_col2:
        st.button("Limpar Filtros 🧹", on_click=clear_all_filters, use_container_width=True, type="primary")

    df_filtered = df.copy()
    if st.session_state.semantic_term:
        with st.spinner("Buscando por significado..."):
            ranked_indices = search_semantic(st.session_state.semantic_term, df.embeddings.to_list())
        if ranked_indices:
            df_filtered = df.loc[ranked_indices]
        else:
            df_filtered = pd.DataFrame(columns=df.columns); st.warning("Nenhum resultado para a busca semântica.")
    elif st.session_state.search_term:
        term = st.session_state.search_term
        df_search = df_filtered.copy()
        df_search['Assuntos_str_search'] = df_search['Assuntos_Processados'].apply(lambda x: ', '.join(map(str, x)))
        cols_to_search = ["Autor", "Título", "Resumo_LLM", "Orientador", "Assuntos_str_search"]
        mask = df_search[[c for c in cols_to_search if c in df_search.columns]].fillna('').astype(str).apply(
            lambda col: col.str.contains(term, case=False, na=False)
        ).any(axis=1)
        df_filtered = df_filtered[mask]
    
    selected_subject = st.session_state.get('subject_filter', subject_options[0])
    if selected_subject != '-- Selecione um Assunto --':
        mask_subject = df_filtered['Assuntos_Processados'].apply(lambda lista: selected_subject in lista)
        df_filtered = df_filtered[mask_subject]

    st.divider()

    df_para_exibir = df_filtered.copy()
    if 'Assuntos_Processados' in df_para_exibir.columns:
        df_para_exibir["Assuntos"] = df_para_exibir["Assuntos_Processados"].apply(
            lambda x: ', '.join(map(str, x)) if isinstance(x, list) else str(x)
        )
    
    cols_display = ["Tipo de Documento", "Autor", "Título", "Ano", "Orientador", "Assuntos"]
    cols_display_existentes = [c for c in cols_display if c in df_para_exibir.columns]
    df_aggrid = df_para_exibir[cols_display_existentes + ['index_original']].fillna('')

    gb = GridOptionsBuilder.from_dataframe(df_aggrid)
    gb.configure_default_column(resizable=True, wrapText=True, autoHeight=True, suppressMenu=True, sortable=True)
    gb.configure_selection(selection_mode='single', use_checkbox=False)
    gb.configure_column("Título", width=600); gb.configure_column("Autor", width=250)
    gb.configure_column("Orientador", width=250); gb.configure_column("Assuntos", width=350)
    gb.configure_column("Tipo de Documento", width=150); gb.configure_column("Ano", width=90)
    gb.configure_column("index_original", hide=True)
    grid_opts = gb.build()
    
    grid_response = AgGrid(
        df_aggrid, gridOptions=grid_opts, update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.AS_INPUT, fit_columns_on_grid_load=False,
        enable_enterprise_modules=False, key="main_interactive_grid", theme='streamlit'
    )
    
    selected_rows = grid_response.get("selected_rows", [])

    if st.button("Visualizar Detalhes do Item Selecionado", use_container_width=True, type="primary"):
        if len(selected_rows) == 1:
            st.session_state.selected_item_cache = pd.DataFrame(selected_rows).copy()
            st.session_state.view_mode = 'details'
            st.rerun()
        else:
            st.warning("Por favor, selecione uma única linha na tabela antes de clicar em 'Visualizar'.")

def render_page_consultas(df: pd.DataFrame, embeddings: np.ndarray, matriz_similaridade: np.ndarray, subject_options: list):
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = 'grid'
    
    if st.session_state.view_mode == 'grid':
        df_with_embeds = df.copy()
        df_with_embeds['embeddings'] = list(embeddings)
        render_grid_view(df_with_embeds, subject_options)
    elif st.session_state.view_mode == 'details':
        render_details_view(df, matriz_similaridade)

def render_page_dashboard(df: pd.DataFrame, embeddings: np.ndarray):
    st.title("Dashboard de Análise do Acervo")
    st.markdown("---")
    st.subheader("Top 20 Assuntos Mais Frequentes")
    todos_assuntos = [assunto for sublista in df['Assuntos_Processados'] for assunto in sublista]
    if todos_assuntos:
        contador_assuntos = Counter(todos_assuntos)
        df_top20 = pd.DataFrame(contador_assuntos.most_common(20), columns=['Assunto', 'Quantidade'])
        fig_assuntos = px.bar(df_top20.sort_values(by='Quantidade', ascending=True), x='Quantidade', y='Assunto', orientation='h', title=' ', text='Quantidade')
        fig_assuntos.update_traces(marker_color='#1f77b4', textposition='outside')
        fig_assuntos.update_layout(yaxis=dict(tickmode='linear'), xaxis_title="Ocorrências", yaxis_title=None, margin=dict(l=200, r=20, t=50, b=50), title_x=0.5)
        st.plotly_chart(fig_assuntos, use_container_width=True)
    st.markdown("---")
    st.subheader("Produção Anual por Tipo de Documento")
    contagem_agrupada = df.groupby(['Ano', 'Tipo de Documento']).size().reset_index(name='Quantidade').sort_values('Ano')
    if not contagem_agrupada.empty:
        fig_producao = px.bar(contagem_agrupada, x='Ano', y='Quantidade', color='Tipo de Documento', title=' ', barmode='group')
        fig_producao.update_layout(xaxis_title="Ano", yaxis_title="Quantidade", title_x=0.5, legend_title_text='Tipo')
        fig_producao.update_xaxes(type='category')
        st.plotly_chart(fig_producao, use_container_width=True)
    st.markdown("---")
    st.subheader("Visualização de Clusters de Documentos (PCA + K-Means)")
    with st.expander("ℹ️ Como interpretar este gráfico?"):
        st.markdown("Este gráfico organiza todos os documentos do acervo em um espaço 3D, agrupando-os por similaridade de conteúdo.")
    k_escolhido = st.slider("Selecione o número de clusters (k):", min_value=2, max_value=8, value=4, step=1)
    with st.spinner(f"Calculando {k_escolhido} clusters..."):
        df_plot_3d = compute_clusters(embeddings, k_escolhido)
        df_plot_3d['Título'] = df['Título']
        df_plot_3d['Autor'] = df['Autor']
        df_plot_3d['cluster'] = df_plot_3d['cluster'].astype(str)
        cores_viridis_discreto = px.colors.sample_colorscale("Viridis", 8)
        fig_3d = px.scatter_3d(
            df_plot_3d, x='pca1', y='pca2', z='pca3', color='cluster', hover_name='Título',
            hover_data={'Autor': True, 'cluster': True, 'pca1': False, 'pca2': False, 'pca3': False},
            title=f'Clusters de Documentos (k={k_escolhido})',
            color_discrete_sequence=cores_viridis_discreto
        )
        fig_3d.update_traces(marker=dict(size=4, opacity=0.8))
        fig_3d.update_layout(height=700, legend_title_text='Clusters',
                             scene=dict(xaxis_title='Comp. Principal 1', yaxis_title='Comp. Principal 2', zaxis_title='Comp. Principal 3', aspectmode='cube'))
        st.plotly_chart(fig_3d, use_container_width=True)

def render_page_sobre():
    st.title("Sobre o projeto")
    st.markdown("""
    Esta aplicação foi desenvolvida como uma interface inteligente para explorar o acervo de dissertações e teses do PPGDR. 
    Ela utiliza técnicas de Processamento de Linguagem Natural (PLN) e Inteligência Artificial (IA) para facilitar a descoberta de conhecimento e a análise de tendências.
    **Versão 1.7 - 09/25**
    """)
    st.divider()
    with st.container(border=True):
        st.subheader("🔎 1. Explore e Selecione na Tela de Consultas")
        st.markdown("Use as buscas ou filtros. Para analisar um item, **clique na linha desejada** na tabela e, em seguida, **clique no botão 'Visualizar Detalhes do Item Selecionado'**.")
    with st.container(border=True):
        st.subheader("⬅️ 2. Navegue de Volta")
        st.markdown("Na tela de detalhes, use o botão **'Voltar para a Tabela'** para retornar à sua busca.")
    with st.container(border=True):
        st.subheader("📊 3. Visualize o Panorama no Dashboard")
        st.markdown("Explore gráficos interativos sobre a produção anual, os assuntos mais frequentes e uma visualização 3D dos clusters temáticos de todo o acervo.")
    st.divider()
    col1, col2 = st.columns([2, 1])
    with col1:
        st.caption("""
            **Autoria do Aplicativo:** Maiko R. Spiess  
            **Fonte dos Dados:** Biblioteca Universitária FURB  
            **Data da Base de Conhecimento:** 08/2025
        """)
    with col2:
        st.link_button("Visite nosso site!", "https://www.net-dr.org", use_container_width=True)

# --------------------------------------------------------------------------
# FUNÇÃO PRINCIPAL (ROTEADOR)
# --------------------------------------------------------------------------
def main():
    setup_page()
    st.markdown("""<style>[data-testid="stSidebar"] {background-color: #0F5EDD;}</style>""", unsafe_allow_html=True)
    if 'page' not in st.session_state: st.session_state.page = "Consultas"

    with st.sidebar:
        st.markdown("<h1 style='color:white;'><b>📚 Acervo PPGDR</b></h1>", unsafe_allow_html=True)
        if st.button("Consultas", use_container_width=True): st.session_state.view_mode = 'grid'
        if st.button("Dashboard", use_container_width=True): st.session_state.page = "Dashboard"
        if st.button("Sobre", use_container_width=True): st.session_state.page = "Sobre"
        st.divider()
        try:
            st.image("NET-01.png", use_container_width=True)
        except Exception:
            st.warning("Logo não encontrado.")

    # Inicialização do estado da view da página de consulta
    if st.session_state.page == "Consultas" and 'view_mode' not in st.session_state:
        st.session_state.view_mode = 'grid'

    df_raw = load_data(CSV_DATA_PATH)
    if df_raw is None:
        st.error("Falha fatal ao carregar o arquivo de dados. A aplicação não pode continuar."); st.stop()

    df = df_raw.rename(columns={"Tipo_Documento": "Tipo de Documento"})
    embeddings = load_embeddings(EMBEDDINGS_PATH)
    
    if not validate_data(df, embeddings):
        st.error("Falha na validação dos dados. A aplicação não pode continuar."); st.stop()

    matriz_similaridade = calculate_similarity_matrix(embeddings)
    subject_options = prepare_subject_list(df)

    if st.session_state.page == "Consultas":
        render_page_consultas(df, embeddings, matriz_similaridade, subject_options)
    elif st.session_state.page == "Dashboard":
        render_page_dashboard(df, embeddings)
    elif st.session_state.page == "Sobre":
        render_page_sobre()

if __name__ == "__main__":
    main()
