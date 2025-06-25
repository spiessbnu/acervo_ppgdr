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

# --------------------------------------------------------------------------
# FUNÇÃO 1: Configuração da página do Streamlit
# --------------------------------------------------------------------------
def setup_page():
    st.set_page_config(
        page_title="Visualização de Dados do Acervo v2",
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

# --------------------------------------------------------------------------
# FUNÇÃO PARA GERAR O GRAFO DE SIMILARIDADE
# --------------------------------------------------------------------------
def generate_similarity_graph(df, matriz_similaridade, id_documento_inicial, num_vizinhos, limiar_de_aresta):
    """Gera um grafo de similaridade interativo com Plotly."""
    
    # --- 1. Coleta dos Nós e Construção do Grafo ---
    nos_da_rede = {id_documento_inicial}
    # Encontra os N vizinhos mais próximos (excluindo ele mesmo)
    vizinhos_l1 = np.argsort(matriz_similaridade[id_documento_inicial])[-num_vizinhos-1:-1][::-1]
    nos_da_rede.update(vizinhos_l1)

    G = nx.Graph()
    # Adiciona os nós
    for node_id in nos_da_rede:
        node_info = df.iloc[node_id]
        level = 0 if node_id == id_documento_inicial else 1
        G.add_node(node_id, title=node_info['Título'], author=node_info['Autor'], level=level)

    # Adiciona arestas do nó central para seus vizinhos
    for vizinho_id in vizinhos_l1:
        similaridade = matriz_similaridade[id_documento_inicial, vizinho_id]
        if similaridade >= limiar_de_aresta:
            G.add_edge(id_documento_inicial, vizinho_id, weight=similaridade)
            
    # --- 2. Visualização Interativa com Plotly ---
    pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)

    edge_trace = go.Scatter(x=[], y=[], line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')
    edge_label_trace = go.Scatter(x=[], y=[], mode='text', text=[], textposition='middle center',
                                  hoverinfo='none', textfont=dict(size=9, color='firebrick'))

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
        edge_label_trace['x'] += tuple([(x0 + x1) / 2])
        edge_label_trace['y'] += tuple([(y0 + y1) / 2])
        edge_label_trace['text'] += tuple([f"{edge[2]['weight']:.2f}"])

    node_trace = go.Scatter(
        x=[], y=[],
        mode='markers+text',
        text=[],
        hovertext=[], # <-- A CORREÇÃO ESTÁ AQUI
        hovertemplate="%{hovertext}",
        marker=dict(color=[], size=[], line_width=2)
    )
    
    cores_niveis = {0: 'crimson', 1: 'royalblue'}
    for node in G.nodes():
        x, y = pos[node]
        info = G.nodes[node]
        level = info['level']
        
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['marker']['color'] += tuple([cores_niveis[level]])
        
        if level == 0:
            size = 35
            similarity_text = "Nó Central"
        else:
            similarity_score = matriz_similaridade[node, id_documento_inicial]
            size = 15 + (similarity_score ** 3 * 40)
            similarity_text = f"Similaridade: {similarity_score:.3f}"

        node_trace['marker']['size'] += tuple([size])
        
        hover_text = f"<b>{info['title']}</b><br>Autor: {info['author']}<br>{similarity_text}"
        node_trace['hovertext'] += tuple([hover_text])
        
        label_texto = info['title'][:30] + '...' if len(info['title']) > 30 else info['title']
        node_trace['text'] += tuple([label_texto])

    node_trace.textposition = 'top center'
    node_trace.textfont = dict(size=9, color='#333')

    fig = go.Figure(data=[edge_trace, node_trace, edge_label_trace],
                 layout=go.Layout(
                    title=f'<br>Rede de Similaridade para: "{df.iloc[id_documento_inicial]["Título"][:60]}..."',
                    titlefont_size=16, showlegend=False, hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig

# --------------------------------------------------------------------------
# FUNÇÃO 3: Corpo principal do aplicativo
# --------------------------------------------------------------------------
def main():
    setup_page()
    st.title("Visualizador de Acervo Acadêmico")
    st.markdown("Selecione uma linha na tabela para ver detalhes e trabalhos similares.")

    # --- CARREGAMENTO INICIAL DE TODOS OS DADOS ---
    df = load_data("dados_finais_com_resumo_llm.csv")
    embeddings = load_embeddings("openai_embeddings_concatenado_large.npy")
    matriz_similaridade = calculate_similarity_matrix(embeddings)
    
    if df.empty:
        return

    df = df.rename(columns={"Tipo_Documento": "Tipo de Documento"})
    df['index_original'] = df.index
    
    cols_display = ["Tipo de Documento", "Autor", "Título", "Ano", "Assuntos", "Orientador"]
    df_aggrid = df[cols_display + ['index_original']]

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

    # --- CRIAÇÃO DAS ABAS ---
    tab_detalhes, tab_similares = st.tabs(["Detalhes", "Trabalhos Similares"])

    # --- ABA 1: DETALHES ---
    with tab_detalhes:
        if selected_rows is not None and not selected_rows.empty:
            detalhes = df.loc[selected_rows.iloc[0]['index_original']]
            st.markdown("#### Assuntos")
            st.write(detalhes.get('Assuntos', 'Nenhum assunto listado.'))
            st.markdown("#### Resumo")
            st.write(detalhes.get('Resumo_LLM', 'Resumo não disponível.'))
            st.markdown("#### Link para Download")
            link_pdf = detalhes.get('Link_PDF')
            if link_pdf and isinstance(link_pdf, str):
                st.link_button("Baixar PDF", url=link_pdf, use_container_width=True)
            else:
                st.warning("Nenhum link para download disponível.")
        else:
            st.info("Selecione um registro na tabela acima para ver os detalhes.")

    # --- ABA 2: TRABALHOS SIMILARES ---
    with tab_similares:
        if embeddings.size == 0 or matriz_similaridade.size == 0:
            st.warning("Não foi possível carregar os dados de similaridade. Verifique os arquivos.")
        elif selected_rows is not None and not selected_rows.empty:
            id_selecionado = selected_rows.iloc[0]['index_original']
            
            st.subheader("Controles da Visualização")
            col1, col2 = st.columns(2)
            with col1:
                num_vizinhos = st.slider("Número de vizinhos", min_value=1, max_value=10, value=5, step=1)
            with col2:
                limiar_similaridade = st.slider("Limiar de similaridade", min_value=0.10, max_value=0.50, value=0.20, step=0.01)

            # Gera e exibe o gráfico
            fig = generate_similarity_graph(df, matriz_similaridade, id_selecionado, num_vizinhos, limiar_similaridade)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Selecione um registro na tabela para visualizar trabalhos similares.")

# --------------------------------------------------------------------------
# Ponto de entrada do script
# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
