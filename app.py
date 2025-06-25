# --------------------------------------------------------------------------
# BIBLIOTECAS NECESSÁRIAS
# --------------------------------------------------------------------------
import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import openai

# --------------------------------------------------------------------------
# FUNÇÃO 1: Configuração da página do Streamlit
# --------------------------------------------------------------------------
def setup_page():
    st.set_page_config(
        page_title="Visualizador de Acervo Acadêmico v6",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# --------------------------------------------------------------------------
# FUNÇÕES DE CARREGAMENTO E PROCESSAMENTO COM CACHE
# --------------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Carrega o arquivo CSV principal e retorna um DataFrame."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"Erro: O arquivo de dados '{path}' não foi encontrado.")
        return pd.DataFrame()

@st.cache_data
def load_embeddings(path: str) -> np.ndarray:
    """Carrega os embeddings de um arquivo .npy."""
    try:
        return np.load(path)
    except FileNotFoundError:
        st.error(f"Erro: O arquivo de embeddings '{path}' não foi encontrado.")
        return np.array([])

@st.cache_data
def calculate_similarity_matrix(_embeddings: np.ndarray) -> np.ndarray:
    """Calcula a matriz de similaridade de cossenos a partir dos embeddings."""
    if _embeddings.size > 0:
        return cosine_similarity(_embeddings)
    return np.array([])

# --------------------------------------------------------------------------
# FUNÇÃO DE IA PARA GERAR SÍNTESE
# --------------------------------------------------------------------------
def get_ai_synthesis(summaries: str) -> str:
    """Chama a API da OpenAI para gerar uma síntese dos textos."""
    try:
        client = openai.OpenAI(api_key=st.secrets["openai"]["api_key"])
        prompt_template = """
        Você é um assistente de pesquisa acadêmica altamente qualificado. Sua tarefa é analisar um conjunto de resumos de trabalhos acadêmicos e gerar uma análise concisa.

        Com base nos seguintes resumos:
        ---
        {summaries}
        ---

        Por favor, gere uma resposta estritamente no seguinte formato:

        **Síntese Temática:**
        [Um parágrafo coeso que resume os principais temas, metodologias e conclusões encontrados no conjunto de textos.]

        **Termos Comuns:**
        - [Termo 1]
        - [Termo 2]
        - [Termo 3]
        - [Termo 4]
        - [Termo 5]
        """
        prompt = prompt_template.format(summaries=summaries)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Você é um assistente de pesquisa acadêmica."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Erro ao contatar a API de IA: {e}")
        return "Não foi possível gerar a análise."

# --------------------------------------------------------------------------
# FUNÇÃO PARA GERAR O GRAFO DE SIMILARIDADE
# --------------------------------------------------------------------------
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
    edge_label_trace = go.Scatter(x=[], y=[], mode='text', text=[], textposition='middle center', hoverinfo='none', textfont=dict(size=9, color='firebrick'))
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None); edge_trace['y'] += (y0, y1, None)
        edge_label_trace['x'] += ((x0 + x1) / 2,); edge_label_trace['y'] += ((y0 + y1) / 2,)
        edge_label_trace['text'] += (f"{edge[2]['weight']:.2f}",)
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
            size = 15 + (similarity_score ** 3 * 40)
            similarity_text = f"Similaridade: {similarity_score:.3f}"
        node_trace['marker']['size'] += (size,)
        hover_text = f"<b>{info['title']}</b><br>Autor: {info['author']}<br>{similarity_text}"
        node_trace['hovertext'] += (hover_text,)
        label_texto = info['title'][:30] + '...' if len(info['title']) > 30 else info['title']
        node_trace['text'] += (label_texto,)
    node_trace.textposition = 'top center'; node_trace.textfont = dict(size=9, color='#333')
    fig = go.Figure(data=[edge_trace, node_trace, edge_label_trace], layout=go.Layout(
        title={'text': f'<br>Rede de Similaridade para: "{df.iloc[id_documento_inicial]["Título"][:60]}..."', 'font': {'size': 16}},
        showlegend=False, hovermode='closest', margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    return fig, nos_da_rede

# --------------------------------------------------------------------------
# FUNÇÃO DE BUSCA SEMÂNTICA
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def search_semantic(query_text: str, _document_embeddings: np.ndarray, model="text-embedding-3-large") -> list:
    """Gera o embedding para a query e retorna uma lista ordenada de índices de documentos."""
    if not query_text.strip():
        return []
    try:
        client = openai.OpenAI(api_key=st.secrets["openai"]["api_key"])
        query_embedding = client.embeddings.create(input=[query_text], model=model).data[0].embedding
        similarities = cosine_similarity([query_embedding], _document_embeddings).flatten()
        ranked_indices = np.argsort(-similarities)
        return [i for i in ranked_indices if similarities[i] > 0.2]
    except Exception as e:
        st.error(f"Erro na busca inteligente: {e}")
        return []

# --------------------------------------------------------------------------
# FUNÇÃO PRINCIPAL DO APLICATIVO
# --------------------------------------------------------------------------
def main():
    setup_page()
    st.title("Visualizador de Acervo Acadêmico")
    st.markdown("Use a busca para encontrar trabalhos e selecione um na tabela para ver detalhes e similares.")

    if 'analysis_cache' not in st.session_state: st.session_state.analysis_cache = {}
    
    df = load_data("dados_finais_com_resumo_llm.csv")
    embeddings = load_embeddings("openai_embeddings_concatenado_large.npy")
    matriz_similaridade = calculate_similarity_matrix(embeddings)
    
    if df.empty: return

    df = df.rename(columns={"Tipo_Documento": "Tipo de Documento"})
    df['index_original'] = df.index
    
    st.subheader("Ferramentas de Busca")
    col1, col2 = st.columns(2)
    with col1:
        simple_search_term = st.text_input(
            "Busca simples",
            placeholder="Filtro por palavra-chave...",
            help="Busca por correspondência exata de texto."
        )
    with col2:
        semantic_search_term = st.text_input(
            "Busca inteligente (com IA)",
            placeholder="Qual o tema do seu interesse?",
            help="Descreva um tema e pressione Enter. A IA buscará por significado."
        )
    
    if semantic_search_term:
        with st.spinner("Buscando por significado... A IA está processando seu pedido."):
            ranked_indices = search_semantic(semantic_search_term, embeddings)
        if ranked_indices:
            df_display = df.loc[ranked_indices]
            st.success(f"Exibindo os {len(df_display)} resultados mais relevantes para '{semantic_search_term}'.")
        else:
            st.warning("Nenhum resultado relevante encontrado para a busca inteligente.")
            df_display = pd.DataFrame(columns=df.columns)
    elif simple_search_term:
        cols_to_search = ["Autor", "Título", "Assuntos", "Resumo_LLM"]
        mask = df[cols_to_search].fillna('').astype(str).apply(
            lambda col: col.str.contains(simple_search_term, case=False)
        ).any(axis=1)
        df_display = df[mask]
    else:
        df_display = df

    st.divider()
    
    cols_display = ["Tipo de Documento", "Autor", "Título", "Ano", "Assuntos", "Orientador"]
    df_aggrid = df_display[cols_display + ['index_original']]
    gb = GridOptionsBuilder.from_dataframe(df_aggrid)
    gb.configure_selection(selection_mode="single", use_checkbox=True)
    gb.configure_column("index_original", hide=True)
    grid_opts = gb.build()

    grid_response = AgGrid(
        df_aggrid, gridOptions=grid_opts, update_mode=GridUpdateMode.SELECTION_CHANGED,
        enable_enterprise_modules=False, fit_columns_on_grid_load=True, key='data_grid'
    )
    st.divider()

    selected_rows = grid_response.get("selected_rows")
    tab_detalhes, tab_similares = st.tabs(["Detalhes", "Trabalhos Similares"])

    with tab_detalhes:
        if selected_rows is not None and not selected_rows.empty:
            detalhes = df.loc[selected_rows.iloc[0]['index_original']]
            st.markdown("#### Assuntos"); st.write(detalhes.get('Assuntos', 'N/A'))
            st.markdown("#### Resumo"); st.write(detalhes.get('Resumo_LLM', 'N/A'))
            st.markdown("#### Link para Download")
            link_pdf = detalhes.get('Link_PDF')
            if link_pdf and isinstance(link_pdf, str):
                st.link_button("Baixar PDF", url=link_pdf, use_container_width=True)
            else:
                st.warning("Nenhum link para download disponível.")
        else:
            st.info("Selecione um registro na tabela para ver os detalhes.")

    with tab_similares:
        if embeddings.size == 0 or matriz_similaridade.size == 0:
            st.warning("Não foi possível carregar os dados de similaridade.")
        elif selected_rows is not None and not selected_rows.empty:
            id_selecionado = selected_rows.iloc[0]['index_original']
            st.caption("Ajuste a quantidade de trabalhos similares a serem exibidos.")
            texto_ajuda = (
                "Este indicador de similaridade é calculado com base em embeddings de texto, que representam os resumos como vetores numéricos."
                " Utilizamos modelos da OpenAI para comparar esses vetores por similaridade de cosseno, estimando a proximidade de conteúdo entre os textos."
            )
            num_vizinhos = st.slider(
                "Número de vizinhos", min_value=1, max_value=10, value=5, step=1, help=texto_ajuda
            )

            if (st.session_state.get('selected_id') != id_selecionado or 
                st.session_state.get('num_vizinhos_cache') != num_vizinhos):
                if 'analysis_result' in st.session_state: del st.session_state['analysis_result']
                st.session_state.selected_id = id_selecionado
                st.session_state.num_vizinhos_cache = num_vizinhos
            
            fig, node_indices = generate_similarity_graph(df, matriz_similaridade, id_selecionado, num_vizinhos)
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("Documentos incluídos no grafo:")
            df_similares = df.loc[list(node_indices)][["Autor", "Título", "Ano"]].reset_index(drop=True)
            st.dataframe(df_similares, use_container_width=True, hide_index=True)
            st.divider()

            if st.button("Gerar Análise com IA 🧠", key="btn_analise"):
                cache_key = (id_selecionado, num_vizinhos)
                if cache_key in st.session_state.analysis_cache:
                    st.toast("Reexibindo análise previamente gerada. ⚡")
                    st.session_state.analysis_result = st.session_state.analysis_cache[cache_key]
                else:
                    with st.spinner('A IA está lendo e preparando a análise...'):
                        summaries_to_analyze = df.loc[list(node_indices)]['Resumo_LLM'].dropna()
                        full_text_summaries = "\n\n---\n\n".join(summaries_to_analyze)
                        analysis = get_ai_synthesis(full_text_summaries)
                        st.session_state.analysis_result = analysis
                        st.session_state.analysis_cache[cache_key] = analysis

            if 'analysis_result' in st.session_state and st.session_state.analysis_result:
                with st.container(border=True):
                    st.subheader("Análise Gerada por IA")
                    st.markdown(st.session_state.analysis_result)
        else:
            st.info("Selecione um registro na tabela para visualizar trabalhos similares.")

# --------------------------------------------------------------------------
# Ponto de entrada do script
# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
