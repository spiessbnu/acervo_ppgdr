# --------------------------------------------------------------------------
# BIBLIOTECAS NECESSÁRIAS (Unificadas)
# --------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from collections import Counter
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
# FUNÇÕES DE CARREGAMENTO E PROCESSAMENTO DE DADOS (Centralizadas)
# --------------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Carrega o arquivo CSV com tratamento de erro e pré-processamento."""
    try:
        df = pd.read_csv(path)
        if 'Assuntos_Lista' in df.columns:
            df['Assuntos_Processados'] = df['Assuntos_Lista'].apply(
                lambda s: ast.literal_eval(s) if isinstance(s, str) else []
            )
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
    """Carrega os embeddings com tratamento de erro."""
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
    if df is None or embeddings is None: return False
    required_cols = ['Título', 'Autor', 'Resumo_LLM']
    if any(col not in df.columns for col in required_cols):
        st.error("Erro de Integridade: Colunas essenciais não encontradas no CSV.")
        return False
    if len(df) != len(embeddings):
        st.error(f"Erro de Integridade: Incompatibilidade entre CSV ({len(df)}) e embeddings ({len(embeddings)}).")
        return False
    st.toast("Dados carregados e validados!", icon="✅")
    return True

@st.cache_data
def calculate_similarity_matrix(_embeddings: np.ndarray) -> np.ndarray:
    """Calcula a matriz de similaridade de cossenos."""
    if _embeddings is not None: return cosine_similarity(_embeddings)
    return np.array([])

def remover_acentos(texto: str) -> str:
    """Remove acentos para ordenação alfabética correta."""
    nfkd_form = unicodedata.normalize('NFD', texto)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

@st.cache_data
def prepare_subject_list(_df: pd.DataFrame) -> list:
    """Extrai, unifica e ordena os assuntos para o dropdown."""
    if 'Assuntos_Processados' not in _df.columns: return ['-- Selecione um Assunto --']
    todos_assuntos = [assunto for sublista in _df['Assuntos_Processados'] for assunto in sublista]
    lista_unica = sorted(list(set(todos_assuntos)), key=lambda texto: remover_acentos(texto.lower()))
    return ['-- Selecione um Assunto --'] + lista_unica

# --------------------------------------------------------------------------
# FUNÇÕES GERAIS (IA, GRAFOS, CLUSTERS)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def search_semantic(query_text: str, _document_embeddings: np.ndarray, model="text-embedding-3-large") -> list:
    """Gera o embedding para a query e retorna uma lista ordenada de índices de documentos."""
    if not query_text.strip(): return []
    try:
        client = openai.OpenAI(api_key=st.secrets["openai"]["api_key"])
        query_embedding = client.embeddings.create(input=[query_text], model=model).data[0].embedding
        similarities = cosine_similarity([query_embedding], _document_embeddings).flatten()
        return [i for i in np.argsort(-similarities) if similarities[i] > 0.2][:20]
    except Exception as e:
        st.error(f"Erro na busca inteligente: {e}"); return []

def get_ai_synthesis(summaries: str) -> str:
    """Chama a API da OpenAI para gerar uma síntese analítica."""
    try:
        client = openai.OpenAI(api_key=st.secrets["openai"]["api_key"])
        
        # --- NOVO PROMPT INTEGRADO AQUI ---
        prompt_template = """
        Você é um(a) analista especializado(a) em revisão de literatura e síntese de evidências.
        Receberá um conjunto de resumos (abstracts) e deve produzir uma análise que preserve
        as especificidades de cada estudo e, ao mesmo tempo, construa uma visão unificada do campo.

        INSTRUÇÕES (qualidade e segurança)
        - Escreva em português do Brasil, com tom técnico, claro e imparcial.
        - Não invente dados; se algo não constar no resumo, escreva “não informado”.
        - Evite jargão excessivo; explique termos técnicos apenas quando estritamente necessário.
        - Aponte sobreposições entre estudos e evite redundâncias.
        - Não exponha raciocínio passo a passo; entregue apenas as seções pedidas.

        ESCOPO ANALÍTICO MÍNIMO
        - Nível micro: para cada estudo, considere (quando disponível) questão de pesquisa,
          enquadramento teórico/conceitual, método/dados e achados/limitações.
        - Nível macro: identifique padrões, convergências/divergências e lacunas plausíveis,
          integrando os achados em uma narrativa coesa.

        FORMATO DE SAÍDA (estritamente nesta ordem; não use tabelas)

        Síntese analítica
        (2–3 parágrafos densos integrando o conjunto: panorama, fios condutores conceituais,
        escopo empírico, principais resultados e tensões metodológicas/teóricas. Não liste; sintetize.
        Ao citar achados específicos, identifique-os pelo conteúdo — tema, método, amostra — sem transcrever longos trechos.)

        Temas principais
        (- Liste 3–5 temas.
         - Para cada tema: título do tema e 1–2 frases explicando por que é central no corpus.
         - Em seguida, descreva a evidência típica/achados recorrentes em 1 frase, evitando redundâncias.)

        Convergências e Divergências
        (- Convergências: 2–5 enunciados curtos (uma frase cada) que expressem acordos recorrentes entre os estudos.
         - Divergências: 1 enunciado curto, identificando (se existirem) diferenças sobre recorte temático e metodológico.
             Exemplo: um ou mais resumos são muito distintos dos demais; apenas descreva, não critique.
         - Quando apropriado, sinalize “não informado” ou “incerto” para evitar extrapolações.)

        ENTRADA
        CONJUNTO DE RESUMOS:
        ---
        {summaries}
        ---

        RESTRIÇÕES FINAIS
        - Parafraseie; não copie longos trechos dos resumos.
        - Não inclua nada além das três seções especificadas.
        """
        prompt = prompt_template.format(summaries=summaries)
        response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": "Você é um especialista em análise de conteúdo e síntese acadêmica. Responda em português do Brasil."}, {"role": "user", "content": prompt}], temperature=0.6)
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Erro na API da OpenAI: {e}"); return "Falha ao gerar a análise."

def generate_similarity_graph(df, matriz_similaridade, id_documento_inicial, num_vizinhos):
    """Gera um grafo de similaridade e retorna a figura e os IDs dos nós."""
    nos_da_rede = {id_documento_inicial}; vizinhos_l1 = np.argsort(matriz_similaridade[id_documento_inicial])[-num_vizinhos-1:-1][::-1]; nos_da_rede.update(vizinhos_l1)
    G = nx.Graph()
    for node_id in nos_da_rede:
        node_info = df.iloc[node_id]; level = 0 if node_id == id_documento_inicial else 1
        G.add_node(node_id, title=node_info['Título'], author=node_info['Autor'], level=level)
    for vizinho_id in vizinhos_l1:
        similaridade = matriz_similaridade[id_documento_inicial, vizinho_id]; G.add_edge(id_documento_inicial, vizinho_id, weight=similaridade)
    pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
    edge_trace = go.Scatter(x=[], y=[], line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')
    edge_label_trace = go.Scatter(x=[], y=[], mode='text', text=[], textposition='middle center', hoverinfo='none', textfont=dict(size=9, color='firebrick'))
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]; edge_trace['x'] += (x0, x1, None); edge_trace['y'] += (y0, y1, None)
        edge_label_trace['x'] += ((x0 + x1) / 2,); edge_label_trace['y'] += ((y0 + y1) / 2,); edge_label_trace['text'] += (f"{edge[2]['weight']:.2f}",)
    node_trace = go.Scatter(x=[], y=[], mode='markers+text', text=[], hovertext=[], hovertemplate="%{hovertext}", marker=dict(color=[], size=[], line_width=2))
    cores_niveis = {0: 'crimson', 1: 'royalblue'}
    for node in G.nodes():
        x, y = pos[node]; info = G.nodes[node]; level = info['level']; node_trace['x'] += (x,); node_trace['y'] += (y,); node_trace['marker']['color'] += (cores_niveis[level],)
        if level == 0: size = 35; similarity_text = "Nó Central"
        else: similarity_score = matriz_similaridade[node, id_documento_inicial]; size = 15 + (similarity_score ** 3 * 40); similarity_text = f"Similaridade: {similarity_score:.3f}"
        node_trace['marker']['size'] += (size,); hover_text = f"<b>{info['title']}</b><br>Autor: {info['author']}<br>{similarity_text}"; node_trace['hovertext'] += (hover_text,)
        label_texto = info['title'][:30] + '...' if len(info['title']) > 30 else info['title']; node_trace['text'] += (label_texto,)
    node_trace.textposition = 'top center'; node_trace.textfont = dict(size=9, color='#333')
    fig = go.Figure(data=[edge_trace, node_trace, edge_label_trace], layout=go.Layout(title={'text': f'Rede de Similaridade para: "{df.iloc[id_documento_inicial]["Título"][:60]}..."', 'font': {'size': 16}}, showlegend=False, hovermode='closest', margin=dict(b=20, l=5, r=5, t=40), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    return fig, nos_da_rede

@st.cache_data
def compute_clusters(_embeddings, k):
    """Calcula PCA e K-Means para visualização de clusters."""
    pca = PCA(n_components=3, random_state=42); embeddings_3d = pca.fit_transform(_embeddings)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto'); cluster_labels = kmeans.fit_predict(_embeddings)
    df_plot = pd.DataFrame(embeddings_3d, columns=['pca1', 'pca2', 'pca3']); df_plot['cluster'] = cluster_labels
    return df_plot

# --------------------------------------------------------------------------
# FUNÇÃO PARA RENDERIZAR A PÁGINA 'CONSULTAS'
# --------------------------------------------------------------------------
def render_page_consultas(df, embeddings, matriz_similaridade, subject_options):
    st.title("🔎 Consulta ao Acervo de Dissertações e Teses")
    st.markdown("Utilize os filtros abaixo para encontrar trabalhos ou selecione um item na tabela para ver detalhes e análises de similaridade.")
    
    if 'grid_key' not in st.session_state: st.session_state.grid_key = str(uuid.uuid4())
    if 'analysis_cache' not in st.session_state: st.session_state.analysis_cache = {}

    def clear_searches():
        st.session_state.search_term = ""
        st.session_state.semantic_term = ""
        st.session_state.subject_filter = subject_options[0]
        st.session_state.grid_key = str(uuid.uuid4())
        if 'analysis_result' in st.session_state: del st.session_state['analysis_result']
        if 'selected_id' in st.session_state: del st.session_state['selected_id']

    search_col1, search_col2 = st.columns(2)
    with search_col1:
        st.text_input("Busca simples por palavra-chave", key="search_term", placeholder="Filtre por autor, título, resumo...")
    with search_col2:
        st.text_input("Busca semântica (com IA)", key="semantic_term", placeholder="Qual o tema do seu interesse?")

    filter_col1, filter_col2 = st.columns([3, 1])
    with filter_col1:
        st.selectbox("Filtro por Assunto", options=subject_options, key="subject_filter", index=subject_options.index(st.session_state.get('subject_filter', subject_options[0])))
    with filter_col2:
        st.button("Limpar Filtros 🧹", on_click=clear_searches, use_container_width=True, type="primary")
    
    st.divider()

    df_filtered = df.copy()
    if st.session_state.get('semantic_term'):
        with st.spinner("Buscando por significado..."):
            ranked_indices = search_semantic(st.session_state.semantic_term, embeddings)
        if ranked_indices: df_filtered = df.loc[ranked_indices]
        else: st.warning("Nenhum resultado para a busca semântica."); df_filtered = pd.DataFrame(columns=df.columns)
    elif st.session_state.get('search_term'):
        term = st.session_state.search_term
        cols_to_search = ["Autor", "Título", "Resumo_LLM", "Orientador"]
        df_filtered['Assuntos_str_search'] = df_filtered['Assuntos_Processados'].apply(lambda x: ', '.join(map(str, x)))
        mask = df_filtered[cols_to_search + ['Assuntos_str_search']].fillna('').astype(str).apply(lambda col: col.str.contains(term, case=False, na=False)).any(axis=1)
        df_filtered = df_filtered[mask]
    
    selected_subject = st.session_state.get('subject_filter', subject_options[0])
    if selected_subject != '-- Selecione um Assunto --':
        mask_subject = df_filtered['Assuntos_Processados'].apply(lambda lista: selected_subject in lista)
        df_filtered = df_filtered[mask_subject]

    cols_display = ["Tipo de Documento", "Autor", "Título", "Ano", "Orientador"]
    df_aggrid = df_filtered[cols_display + ['index_original']].copy()
    df_aggrid["Assuntos"] = df_filtered['Assuntos_Processados'].apply(lambda x: ', '.join(x))
    
    gb = GridOptionsBuilder.from_dataframe(df_aggrid[cols_display + ["Assuntos", 'index_original']])
    gb.configure_default_column(resizable=True, wrapText=True, autoHeight=True, sortable=True); gb.configure_column("Título", width=500); gb.configure_column("Autor", width=250); gb.configure_column("Orientador", width=250); gb.configure_column("Assuntos", width=350); gb.configure_column("Tipo de Documento", width=150); gb.configure_column("Ano", width=90); gb.configure_selection(selection_mode="single", use_checkbox=True); gb.configure_column("index_original", hide=True)
    grid_opts = gb.build()

    grid_response = AgGrid(df_aggrid, gridOptions=grid_opts, update_mode=GridUpdateMode.SELECTION_CHANGED, enable_enterprise_modules=False, fit_columns_on_grid_load=False, key=st.session_state.grid_key)
    st.divider()

    selected_rows = pd.DataFrame(grid_response.get("selected_rows"))
    tab_detalhes, tab_similares = st.tabs(["📄 Detalhes", "🔗 Trabalhos Similares"])

    with tab_detalhes:
        if not selected_rows.empty:
            detalhes = df.loc[selected_rows.iloc[0]['index_original']]
            st.subheader(detalhes.get('Título', '')); st.markdown(f"**Autor:** {detalhes.get('Autor', '')} | **Ano:** {detalhes.get('Ano', '')}"); st.markdown("---"); st.markdown("#### Assuntos"); st.write(', '.join(detalhes.get('Assuntos_Processados', []))); st.markdown("#### Resumo"); st.write(detalhes.get('Resumo_LLM', ''))
            link_pdf = detalhes.get('Link_PDF')
            if link_pdf and isinstance(link_pdf, str): st.link_button("🔗 Baixar PDF", url=link_pdf, use_container_width=True)
        else: st.info("Selecione um registro na tabela para ver os detalhes.")
            
    with tab_similares:
        if not selected_rows.empty:
            id_selecionado = selected_rows.iloc[0]['index_original']
            num_vizinhos = st.slider("Número de trabalhos similares para exibir:", 1, 10, 5, 1, key=f"slider_{id_selecionado}")
            
            # --- LÓGICA DE DETECÇÃO DE MUDANÇA DE CONTEXTO ---
            contexto_atual = (id_selecionado, num_vizinhos)
            if contexto_atual != st.session_state.get('ultimo_contexto', None):
                if 'analysis_result' in st.session_state:
                    del st.session_state['analysis_result']
                st.session_state['ultimo_contexto'] = contexto_atual

            fig, node_indices = generate_similarity_graph(df, matriz_similaridade, id_selecionado, num_vizinhos)
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("Documentos incluídos no grafo:")
            df_similares = df.loc[list(node_indices)][["Autor", "Título", "Ano"]].reset_index(drop=True)
            st.dataframe(df_similares, use_container_width=True, hide_index=True)
            st.divider()

            if st.button("Gerar Análise com IA 🧠", key=f"btn_analise_{id_selecionado}"):
                cache_key = (id_selecionado, num_vizinhos)
                if cache_key in st.session_state.analysis_cache: 
                    st.session_state.analysis_result = st.session_state.analysis_cache[cache_key]
                    st.toast("Reexibindo análise previamente gerada. ⚡")
                else:
                    summaries_to_analyze = df.loc[list(node_indices)]['Resumo_LLM'].dropna()
                    if not summaries_to_analyze.empty:
                        with st.spinner('A IA está lendo e preparando a análise...'):
                            analysis = get_ai_synthesis("\n\n---\n\n".join(summaries_to_analyze))
                            st.session_state.analysis_result = analysis
                            st.session_state.analysis_cache[cache_key] = analysis
                    else: 
                        st.session_state.analysis_result = "Não há resumos disponíveis para gerar a análise."
            
            if 'analysis_result' in st.session_state and st.session_state.analysis_result:
                with st.container(border=True): 
                    st.subheader("Análise Gerada por IA")
                    st.markdown(st.session_state.analysis_result)
        else: 
            st.info("Selecione um registro na tabela para visualizar trabalhos similares.")

# --------------------------------------------------------------------------
# FUNÇÃO PARA RENDERIZAR A PÁGINA 'DASHBOARD'
# --------------------------------------------------------------------------
def render_page_dashboard(df, embeddings):
    st.title("📊 Dashboard de Análise do Acervo")
    st.markdown("---")
    st.subheader("Top 20 Assuntos Mais Frequentes")
    todos_assuntos = [assunto for sublista in df['Assuntos_Processados'] for assunto in sublista]
    if todos_assuntos:
        contador_assuntos = Counter(todos_assuntos)
        df_top20 = pd.DataFrame(contador_assuntos.most_common(20), columns=['Assunto', 'Quantidade'])
        fig_assuntos = px.bar(df_top20.sort_values(by='Quantidade', ascending=True), x='Quantidade', y='Assunto', orientation='h', text='Quantidade')
        fig_assuntos.update_traces(marker_color='#1f77b4', textposition='outside'); fig_assuntos.update_layout(yaxis_title=None, xaxis_title="Ocorrências", margin=dict(l=250))
        st.plotly_chart(fig_assuntos, use_container_width=True)
    st.markdown("---")
    st.subheader("Produção Anual por Tipo de Documento")
    df_renamed = df.rename(columns={"Tipo_Documento": "Tipo de Documento"})
    contagem_agrupada = df_renamed.groupby(['Ano', 'Tipo de Documento']).size().reset_index(name='Quantidade').sort_values('Ano')
    if not contagem_agrupada.empty:
        fig_producao = px.bar(contagem_agrupada, x='Ano', y='Quantidade', color='Tipo de Documento', barmode='group')
        fig_producao.update_layout(xaxis_title="Ano", yaxis_title="Quantidade", legend_title_text='Tipo'); fig_producao.update_xaxes(type='category')
        st.plotly_chart(fig_producao, use_container_width=True)
    st.markdown("---")
    st.subheader("Visualização de Clusters de Documentos (PCA + K-Means)")
    with st.expander("ℹ️ Como interpretar este gráfico?"):
        st.markdown("Este gráfico organiza todos os documentos do acervo em um espaço 3D, agrupando-os por similaridade de conteúdo. Cada ponto é um documento. Documentos com temas semelhantes tendem a aparecer mais próximos uns dos outros. Os grupos (clusters) são coloridos para destacar as principais concentrações temáticas do acervo.")
    k_escolhido = st.slider("Selecione o número de clusters (grupos):", min_value=2, max_value=8, value=4, step=1)
    with st.spinner(f"Calculando {k_escolhido} clusters..."):
        df_plot_3d = compute_clusters(embeddings, k_escolhido)
        df_plot_3d['Título'] = df['Título']; df_plot_3d['Autor'] = df['Autor']; df_plot_3d['cluster'] = df_plot_3d['cluster'].astype(str)
        fig_3d = px.scatter_3d(df_plot_3d, x='pca1', y='pca2', z='pca3', color='cluster', hover_name='Título', hover_data={'Autor': True, 'cluster': True, 'pca1': False, 'pca2': False, 'pca3': False}, title=f'Clusters de Documentos (k={k_escolhido})', color_discrete_sequence=px.colors.qualitative.Plotly)
        fig_3d.update_traces(marker=dict(size=4, opacity=0.8)); fig_3d.update_layout(height=700, legend_title_text='Clusters', scene=dict(xaxis_title='Comp. 1', yaxis_title='Comp. 2', zaxis_title='Comp. 3'))
        st.plotly_chart(fig_3d, use_container_width=True)

# --------------------------------------------------------------------------
# FUNÇÃO PARA RENDERIZAR A PÁGINA 'SOBRE'
# --------------------------------------------------------------------------
def render_page_sobre():
    st.title("ℹ️ Sobre o Projeto")
    st.markdown("""Esta aplicação foi desenvolvida como uma interface inteligente para explorar o acervo de dissertações e teses do Programa de Pós-Graduação em Desenvolvimento Regional (PPGDR). Ela utiliza técnicas de Processamento de Linguagem Natural (PLN) e Inteligência Artificial (IA) para facilitar a descoberta de conhecimento e a análise de tendências no acervo.""")
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True): st.subheader("🔎 Explore e Selecione"); st.markdown("Na página de **Consulta**, use as buscas ou filtros para encontrar trabalhos. Clique em uma linha na tabela para carregar os detalhes e visualizar documentos similares.")
    with col2:
        with st.container(border=True): st.subheader("🧠 Descubra Conexões com a IA"); st.markdown("Ao selecionar um trabalho, a aba 'Trabalhos Similares' mostra um grafo de documentos próximos e permite gerar uma síntese analítica da rede com IA.")
    with col3:
        with st.container(border=True): st.subheader("📊 Visualize o Panorama"); st.markdown("Acesse o **Dashboard** para explorar gráficos interativos sobre a produção anual, os assuntos mais frequentes e uma visualização 3D dos clusters temáticos.")
    st.divider()
    st.info("""**Detalhes Técnicos:** Os resumos foram vetorizados com `text-embedding-3-large` (OpenAI). A busca semântica compara o significado da sua pergunta com o conteúdo dos documentos. As visualizações utilizam Plotly e NetworkX.""", icon="🤖")
    st.caption("""**Autoria do Aplicativo:** Maiko R. Spiess | **Fonte dos Dados:** Biblioteca Universitária FURB | **Data da Base de Conhecimento:** 08/2025""")

# --------------------------------------------------------------------------
# FUNÇÃO PRINCIPAL (ROTEADOR)
# --------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Acervo PPGDR", page_icon="📚", layout="wide")

    st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                background-color: #0F5EDD;
            }
            [data-testid="stSidebar"] h1 {
                color: white;
            }
            [data-testid="stSidebar"] .stButton button {
                color: #0F1116; 
            }
        </style>
    """, unsafe_allow_html=True)

    df = load_data(CSV_DATA_PATH)
    embeddings = load_embeddings(EMBEDDINGS_PATH)

    if not validate_data(df, embeddings):
        st.warning("A aplicação não pode continuar devido a erros nos dados de entrada.")
        st.stop()
    
    df = df.rename(columns={"Tipo_Documento": "Tipo de Documento"})
    df['index_original'] = df.index
    matriz_similaridade = calculate_similarity_matrix(embeddings)
    subject_options = prepare_subject_list(df)

    if 'page' not in st.session_state:
        st.session_state.page = "Consultas"

    with st.sidebar:
        st.title("📚 Acervo PPGDR")
        
        if st.button("Consultas", use_container_width=True):
            st.session_state.page = "Consultas"
        if st.button("Dashboard", use_container_width=True):
            st.session_state.page = "Dashboard"
        if st.button("Sobre", use_container_width=True):
            st.session_state.page = "Sobre"
        
        st.divider()
        st.image("NET-01.png", use_container_width=True)

    if st.session_state.page == "Consultas":
        if 'search_term' not in st.session_state: st.session_state.search_term = ""
        if 'semantic_term' not in st.session_state: st.session_state.semantic_term = ""
        if 'subject_filter' not in st.session_state: st.session_state.subject_filter = subject_options[0]
        render_page_consultas(df, embeddings, matriz_similaridade, subject_options)
    
    elif st.session_state.page == "Dashboard":
        render_page_dashboard(df, embeddings)

    elif st.session_state.page == "Sobre":
        render_page_sobre()

# --------------------------------------------------------------------------
# Ponto de entrada do script
# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
