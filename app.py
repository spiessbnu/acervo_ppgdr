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
import uuid
import unicodedata # Nova biblioteca para normalização de texto
import ast # Nova biblioteca para conversão segura de string para lista

# --------------------------------------------------------------------------
# FUNÇÃO 1: Configuração da página do Streamlit
# --------------------------------------------------------------------------
def setup_page():
    st.set_page_config(
        page_title="Visualizador de Acervo Acadêmico v8",
        page_icon="📚",
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

# --- NOVO: FUNÇÃO AUXILIAR PARA PREPARAR A LISTA DE ASSUNTOS ---
def remover_acentos(texto: str) -> str:
    """Remove acentos de uma string para ordenação alfabética correta."""
    texto_normalizado = unicodedata.normalize('NFD', texto)
    return "".join(c for c in texto_normalizado if not unicodedata.combining(c))

@st.cache_data
def prepare_subject_list(_df: pd.DataFrame) -> list:
    """Extrai, limpa, unifica e ordena os assuntos para o dropdown."""
    if 'Assuntos_Lista' not in _df.columns:
        return []
    
    # Converte a string de lista para uma lista real de forma segura
    def safe_literal_eval(s):
        try: return ast.literal_eval(s)
        except (ValueError, SyntaxError): return []
    
    assuntos_processados = _df['Assuntos_Lista'].apply(safe_literal_eval)
    
    # Cria uma lista única de todos os assuntos
    todos_assuntos = [assunto for sublista in assuntos_processados for assunto in sublista]
    lista_unica = sorted(list(set(todos_assuntos)), key=lambda texto: remover_acentos(texto.lower()))
    
    return ['-- Selecione um Assunto --'] + lista_unica

# (As outras funções: get_ai_synthesis, generate_similarity_graph, search_semantic permanecem as mesmas)
def get_ai_synthesis(summaries: str) -> str:
    try:
        client = openai.OpenAI(api_key=st.secrets["openai"]["api_key"])
        prompt_template = "Você é um assistente de pesquisa... {summaries} ... **Síntese Temática:**\n[...]\n\n**Termos Comuns:**\n- [...]"
        prompt = prompt_template.format(summaries=summaries)
        response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": "Você é um assistente de pesquisa acadêmica."}, {"role": "user", "content": prompt}], temperature=0.5)
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Erro ao contatar a API de IA: {e}"); return "Não foi possível gerar a análise."
def generate_similarity_graph(df, matriz_similaridade, id_documento_inicial, num_vizinhos):
    nos_da_rede = {id_documento_inicial}; vizinhos_l1 = np.argsort(matriz_similaridade[id_documento_inicial])[-num_vizinhos-1:-1][::-1]; nos_da_rede.update(vizinhos_l1)
    G = nx.Graph(); 
    for node_id in nos_da_rede: G.add_node(node_id, title=df.iloc[node_id]['Título'], author=df.iloc[node_id]['Autor'], level=0 if node_id == id_documento_inicial else 1)
    for vizinho_id in vizinhos_l1: G.add_edge(id_documento_inicial, vizinho_id, weight=matriz_similaridade[id_documento_inicial, vizinho_id])
    pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
    edge_trace = go.Scatter(x=[], y=[], line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')
    edge_label_trace = go.Scatter(x=[], y=[], mode='text', text=[], textposition='middle center', hoverinfo='none', textfont=dict(size=9, color='firebrick'))
    for edge in G.edges(data=True): x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]; edge_trace['x'] += (x0, x1, None); edge_trace['y'] += (y0, y1, None); edge_label_trace['x'] += ((x0 + x1) / 2,); edge_label_trace['y'] += ((y0 + y1) / 2,); edge_label_trace['text'] += (f"{edge[2]['weight']:.2f}",)
    node_trace = go.Scatter(x=[], y=[], mode='markers+text', text=[], hovertext=[], hovertemplate="%{hovertext}", marker=dict(color=[], size=[], line_width=2))
    cores_niveis = {0: 'crimson', 1: 'royalblue'}
    for node in G.nodes():
        x, y = pos[node]; info = G.nodes[node]; level = info['level']; node_trace['x'] += (x,); node_trace['y'] += (y,); node_trace['marker']['color'] += (cores_niveis[level],)
        if level == 0: size = 35; similarity_text = "Nó Central"
        else: similarity_score = matriz_similaridade[node, id_documento_inicial]; size = 15 + (similarity_score ** 3 * 40); similarity_text = f"Similaridade: {similarity_score:.3f}"
        node_trace['marker']['size'] += (size,); hover_text = f"<b>{info['title']}</b><br>Autor: {info['author']}<br>{similarity_text}"; node_trace['hovertext'] += (hover_text,)
        label_texto = info['title'][:30] + '...' if len(info['title']) > 30 else info['title']; node_trace['text'] += (label_texto,)
    node_trace.textposition = 'top center'; node_trace.textfont = dict(size=9, color='#333')
    fig = go.Figure(data=[edge_trace, node_trace, edge_label_trace], layout=go.Layout(title={'text': f'<br>Rede de Similaridade para: "{df.iloc[id_documento_inicial]["Título"][:60]}..."', 'font': {'size': 16}}, showlegend=False, hovermode='closest', margin=dict(b=20,l=5,r=5,t=40), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    return fig, nos_da_rede
@st.cache_data(show_spinner=False)
def search_semantic(query_text: str, _document_embeddings: np.ndarray, model="text-embedding-3-large") -> list:
    if not query_text.strip(): return []
    try:
        client = openai.OpenAI(api_key=st.secrets["openai"]["api_key"])
        query_embedding = client.embeddings.create(input=[query_text], model=model).data[0].embedding
        similarities = cosine_similarity([query_embedding], _document_embeddings).flatten()
        ranked_indices = np.argsort(-similarities)
        return [i for i in ranked_indices if similarities[i] > 0.2][:20]
    except Exception as e:
        st.error(f"Erro na busca inteligente: {e}"); return []
# --------------------------------------------------------------------------
# FUNÇÃO PRINCIPAL DO APLICATIVO
# --------------------------------------------------------------------------
def main():
    setup_page()
    st.title("Visualizador de Acervo Acadêmico")
    st.markdown("Use as ferramentas de busca e filtros para explorar o acervo.")

    # Inicialização dos estados da sessão
    for key in ['analysis_cache', 'search_term', 'semantic_term', 'subject_filter', 'grid_key', 'selected_id', 'num_vizinhos_cache']:
        if key not in st.session_state:
            st.session_state[key] = {} if key == 'analysis_cache' else None if key.endswith('_id') else 0 if key.endswith('_cache') else str(uuid.uuid4()) if key == 'grid_key' else ""

    df = load_data("dados_finais_com_resumo_llm.csv")
    embeddings = load_embeddings("openai_embeddings_concatenado_large.npy")
    matriz_similaridade = calculate_similarity_matrix(embeddings)
    
    if df.empty: return

    df = df.rename(columns={"Tipo_Documento": "Tipo de Documento"})
    df['index_original'] = df.index
    
    st.subheader("Ferramentas de Busca e Filtro")
    
    def clear_searches():
        st.session_state.search_term = ""
        st.session_state.semantic_term = ""
        st.session_state.subject_filter = subject_options[0] # Reseta para a opção padrão
        st.session_state.grid_key = str(uuid.uuid4())
        if 'analysis_result' in st.session_state: del st.session_state['analysis_result']
        if 'selected_id' in st.session_state: del st.session_state['selected_id']

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1: st.text_input("Busca simples", key="search_term")
    with col2: st.text_input("Busca inteligente (com IA)", key="semantic_term")
    with col3: st.button("Limpar Tudo 🧹", on_click=clear_searches, use_container_width=True, help="Limpa todas as buscas e filtros.")

    # NOVO: Filtro por Assunto
    subject_options = prepare_subject_list(df)
    st.selectbox("Filtro por Assunto", options=subject_options, key="subject_filter")

    # Lógica de reset ao iniciar uma nova busca ou filtro
    # ... (a lógica de reset se torna mais complexa, então a integramos ao fluxo de filtragem)

    # --- LÓGICA DE FILTRAGEM PROGRESSIVA ---
    df_filtered = df.copy() # Começa com o DataFrame completo

    # 1. Aplica a busca inteligente (tem prioridade)
    if st.session_state.semantic_term:
        with st.spinner("Buscando por significado..."):
            ranked_indices = search_semantic(st.session_state.semantic_term, embeddings)
        if ranked_indices:
            df_filtered = df.loc[ranked_indices]
            st.success(f"Exibindo {len(df_filtered)} resultados para '{st.session_state.semantic_term}'.")
        else:
            st.warning("Nenhum resultado para a busca inteligente."); df_filtered = pd.DataFrame(columns=df.columns)
    # 2. Se não houver busca inteligente, aplica a busca simples
    elif st.session_state.search_term:
        cols_to_search = ["Autor", "Título", "Assuntos", "Resumo_LLM"]
        mask = df_filtered[cols_to_search].fillna('').astype(str).apply(lambda col: col.str.contains(st.session_state.search_term, case=False)).any(axis=1)
        df_filtered = df_filtered[mask]
    
    # 3. Aplica o filtro por assunto sobre o resultado das buscas anteriores
    selected_subject = st.session_state.subject_filter
    if selected_subject != '-- Selecione um Assunto --':
        # Re-processa a coluna de assuntos no dataframe já filtrado, se necessário
        if 'Assuntos_Processados' not in df_filtered.columns:
             df_filtered['Assuntos_Processados'] = df_filtered['Assuntos_Lista'].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else [])
        
        mask_subject = df_filtered['Assuntos_Processados'].apply(lambda lista: selected_subject in lista)
        df_filtered = df_filtered[mask_subject]

    df_display = df_filtered
    
    st.divider()
    
    # Restante da UI (AgGrid, abas) continua aqui...
    cols_display = ["Tipo de Documento", "Autor", "Título", "Ano", "Assuntos", "Orientador"]
    df_aggrid = df_display[cols_display + ['index_original']]
    gb = GridOptionsBuilder.from_dataframe(df_aggrid)
    gb.configure_selection(selection_mode="single", use_checkbox=True); gb.configure_column("index_original", hide=True)
    grid_opts = gb.build()
    
    # AJUSTE: A chave da AgGrid é resetada para limpar a seleção após uma nova busca/filtro
    # A lógica de reset foi simplificada e agora o fluxo de filtragem é o ponto central.
    # Ao re-filtrar, a `df_aggrid` muda, o que força o update da tabela.
    # O reset da seleção é um efeito colateral desejado.
    grid_response = AgGrid(df_aggrid, gridOptions=grid_opts, update_mode=GridUpdateMode.SELECTION_CHANGED, 
                           enable_enterprise_modules=False, fit_columns_on_grid_load=True, key=f"grid_{len(df_display)}")
    st.divider()

    selected_rows = grid_response.get("selected_rows")
    # ... (o código das abas "Detalhes" e "Trabalhos Similares" permanece o mesmo)
    tab_detalhes, tab_similares = st.tabs(["Detalhes", "Trabalhos Similares"])
    with tab_detalhes:
        if selected_rows is not None and not selected_rows.empty:
            detalhes = df.loc[selected_rows.iloc[0]['index_original']]
            st.markdown("#### Assuntos"); st.write(detalhes.get('Assuntos', 'N/A'))
            st.markdown("#### Resumo"); st.write(detalhes.get('Resumo_LLM', 'N/A'))
            st.markdown("#### Link para Download")
            link_pdf = detalhes.get('Link_PDF')
            if link_pdf and isinstance(link_pdf, str): st.link_button("Baixar PDF", url=link_pdf, use_container_width=True)
            else: st.warning("Nenhum link para download disponível.")
        else:
            st.info("Selecione um registro na tabela para ver os detalhes.")
    with tab_similares:
        if embeddings.size == 0 or matriz_similaridade.size == 0:
            st.warning("Não foi possível carregar os dados de similaridade.")
        elif selected_rows is not None and not selected_rows.empty:
            id_selecionado = selected_rows.iloc[0]['index_original']
            # ... (código do slider, grafo, tabela e botão de IA sem alterações) ...
            st.caption("Ajuste a quantidade de trabalhos similares a serem exibidos.")
            texto_ajuda = ("Este indicador de similaridade é calculado com base em embeddings de texto...")
            num_vizinhos = st.slider("Número de vizinhos", 1, 10, 5, 1, help=texto_ajuda)
            if st.session_state.get('selected_id') != id_selecionado or st.session_state.get('num_vizinhos_cache') != num_vizinhos:
                if 'analysis_result' in st.session_state: del st.session_state['analysis_result']
                st.session_state.selected_id = id_selecionado; st.session_state.num_vizinhos_cache = num_vizinhos
            fig, node_indices = generate_similarity_graph(df, matriz_similaridade, id_selecionado, num_vizinhos)
            st.plotly_chart(fig, use_container_width=True)
            st.write("Documentos incluídos no grafo:")
            df_similares = df.loc[list(node_indices)][["Autor", "Título", "Ano"]].reset_index(drop=True)
            st.dataframe(df_similares, use_container_width=True, hide_index=True); st.divider()
            if st.button("Gerar Análise com IA 🧠", key="btn_analise"):
                cache_key = (id_selecionado, num_vizinhos)
                if cache_key in st.session_state.analysis_cache:
                    st.toast("Reexibindo análise previamente gerada. ⚡"); st.session_state.analysis_result = st.session_state.analysis_cache[cache_key]
                else:
                    summaries_to_analyze = df.loc[list(node_indices)]['Resumo_LLM'].dropna()
                    full_text_summaries = "\n\n---\n\n".join(summaries_to_analyze)
                    if not full_text_summaries.strip():
                        st.warning("Não há resumos disponíveis para gerar a análise."); st.session_state.analysis_result = ""
                    else:
                        with st.spinner('A IA está lendo e preparando a análise...'):
                            analysis = get_ai_synthesis(full_text_summaries); st.session_state.analysis_result = analysis
                            st.session_state.analysis_cache[cache_key] = analysis
            if 'analysis_result' in st.session_state and st.session_state.analysis_result:
                with st.container(border=True): st.subheader("Análise Gerada por IA"); st.markdown(st.session_state.analysis_result)
        else:
            st.info("Selecione um registro na tabela para visualizar trabalhos similares.")

# --------------------------------------------------------------------------
# Ponto de entrada do script
# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
