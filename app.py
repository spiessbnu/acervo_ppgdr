# --------------------------------------------------------------------------
# BIBLIOTECAS NECESSÁRIAS
# --------------------------------------------------------------------------
import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px  # Adicionado para o novo gráfico
from collections import Counter  # Adicionado para contar os assuntos
from sklearn.metrics.pairwise import cosine_similarity
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
# FUNÇÕES DE CARREGAMENTO E PROCESSAMENTO (MAIS ROBUSTAS)
# --------------------------------------------------------------------------
def safe_literal_eval(s):
    """Função segura para converter string de lista em objeto lista."""
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError, TypeError):
        # Se a string for inválida ou vazia (NaN), retorna uma lista vazia
        return []

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Carrega o arquivo CSV com tratamento de erro aprimorado."""
    try:
        df = pd.read_csv(path)
        # Garante que a coluna de assuntos processados seja criada ao carregar os dados
        if 'Assuntos_Lista' in df.columns:
            df['Assuntos_Processados'] = df['Assuntos_Lista'].apply(safe_literal_eval)
        else:
            df['Assuntos_Processados'] = [[] for _ in range(len(df))]
        return df
    except FileNotFoundError:
        st.error(f"Arquivo de dados não encontrado: '{path}'. Verifique se o arquivo está no diretório correto.")
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
        st.error(f"Arquivo de embeddings não encontrado: '{path}'. Verifique se o arquivo está no diretório correto.")
        return None
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar o arquivo de embeddings '{path}': {e}")
        return None


def validate_data(df: pd.DataFrame, embeddings: np.ndarray) -> bool:
    """Verifica se os dados carregados são consistentes."""
    if df is None or embeddings is None:
        return False

    required_cols = ['Título', 'Autor', 'Assuntos_Lista', 'Resumo_LLM']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Erro de Integridade: Coluna(s) necessária(s) não encontrada(s) no CSV: {', '.join(missing_cols)}")
        return False

    if len(df) != len(embeddings):
        st.error(
            f"Erro de Integridade: O número de registros no CSV ({len(df)}) é diferente do "
            f"número de vetores de embeddings ({len(embeddings)}). Os arquivos não são compatíveis."
        )
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
        if level == 0: size = 35; similarity_text = "Nó Central"
        else:
            similarity_score = matriz_similaridade[node, id_documento_inicial]
            size = 15 + (similarity_score ** 3 * 40); similarity_text = f"Similaridade: {similarity_score:.3f}"
        node_trace['marker']['size'] += (size,)
        hover_text = f"<b>{info['title']}</b><br>Autor: {info['author']}<br>{similarity_text}"
        node_trace['hovertext'] += (hover_text,)
        label_texto = info['title'][:30] + '...' if len(info['title']) > 30 else info['title']
        node_trace['text'] += (label_texto,)

    node_trace.textposition = 'top center'
    node_trace.textfont = dict(size=9, color='#333')

    fig = go.Figure(
        data=[edge_trace, node_trace, edge_label_trace],
        layout=go.Layout(
            title={'text': f'<br>Rede de Similaridade para: "{df.iloc[id_documento_inicial]["Título"][:60]}..."', 'font': {'size': 16}},
            showlegend=False, hovermode='closest', margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    return fig, nos_da_rede


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


# --------------------------------------------------------------------------
# FUNÇÕES DE RENDERIZAÇÃO DAS PÁGINAS
# --------------------------------------------------------------------------

def render_page_consultas(df: pd.DataFrame, embeddings: np.ndarray, matriz_similaridade: np.ndarray, subject_options: list):
    """Renderiza a página principal de consulta e análise de documentos."""
    st.title("Consulta ao Acervo de Dissertações e Teses")
    
    # --- INICIALIZAÇÃO DE ESTADO (específico da página) ---
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
        if 'analysis_result' in st.session_state: del st.session_state['analysis_result']
        if 'selected_id' in st.session_state: del st.session_state['selected_id']

    # --- LAYOUT DOS FILTROS ---
    search_col1, search_col2 = st.columns(2)
    with search_col1:
        st.text_input("Busca simples", key="search_term", placeholder="Filtro por palavra-chave...")
    with search_col2:
        st.text_input("Busca inteligente (com IA)", key="semantic_term", placeholder="Qual o tema do seu interesse?", help="Descreva um tema e pressione Enter.")

    filter_col1, filter_col2 = st.columns([3, 1])
    with filter_col1:
        st.selectbox("Filtro por Assunto", options=subject_options, key="subject_filter")
    with filter_col2:
        st.button("Limpar Tudo 🧹", on_click=clear_searches, use_container_width=True, help="Limpa todas as buscas e filtros.")
    
    # --- LÓGICA DE FILTRAGEM ---
    df_filtered = df.copy()
    if st.session_state.semantic_term:
        with st.spinner("Buscando por significado..."):
            ranked_indices = search_semantic(st.session_state.semantic_term, embeddings)
        if ranked_indices:
            df_filtered = df.loc[ranked_indices]
            st.success(f"Exibindo {len(df_filtered)} resultados para '{st.session_state.semantic_term}'.")
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
    
    df_display = df_filtered
    st.divider()
    
    # --- RENDERIZAÇÃO DA TABELA (AgGrid) ---
    current_filter_state = (st.session_state.search_term, st.session_state.semantic_term, st.session_state.subject_filter)
    if st.session_state.get('last_filter_state') != current_filter_state:
        st.session_state.grid_key = str(uuid.uuid4())
        if 'analysis_result' in st.session_state: del st.session_state['analysis_result']
        if 'selected_id' in st.session_state: del st.session_state['selected_id']
    st.session_state.last_filter_state = current_filter_state

    cols_display = ["Tipo de Documento", "Autor", "Título", "Ano", "Assuntos", "Orientador"]
    df_aggrid = df_display[cols_display + ['index_original']]
    
    gb = GridOptionsBuilder.from_dataframe(df_aggrid)
    gb.configure_default_column(resizable=True, wrapText=True, autoHeight=True, suppressMenu=True, sortable=True)
    gb.configure_column("Título", width=500, minWidth=300)
    gb.configure_column("Autor", width=250, minWidth=150)
    gb.configure_column("Orientador", width=250, minWidth=150)
    gb.configure_column("Assuntos", width=350)
    gb.configure_column("Tipo de Documento", width=150)
    gb.configure_column("Ano", width=90, minWidth=70, maxWidth=100)
    gb.configure_selection(selection_mode="single", use_checkbox=True)
    gb.configure_column("index_original", hide=True)
    grid_opts = gb.build()

    grid_response = AgGrid(
        df_aggrid,
        gridOptions=grid_opts,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        enable_enterprise_modules=False,
        fit_columns_on_grid_load=False,
        key=st.session_state.grid_key
    )
    st.divider()

    # --- ABAS DE DETALHES E SIMILARES ---
    selected_rows = grid_response.get("selected_rows")
    tab_detalhes, tab_similares = st.tabs(["Detalhes", "Trabalhos Similares"])

    with tab_detalhes:
        if selected_rows is not None and not selected_rows.empty:
            detalhes = df.loc[selected_rows.iloc[0]['index_original']]
            st.subheader(detalhes.get('Título', 'Título não disponível'))
            st.divider()
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
            st.info("Selecione um registro na tabela para ver os detalhes.")
            
    with tab_similares:
        if not matriz_similaridade.any():
            st.warning("Não foi possível carregar os dados de similaridade.")
        elif selected_rows is not None and not selected_rows.empty:
            id_selecionado = selected_rows.iloc[0]['index_original']
            st.caption("Ajuste a quantidade de trabalhos similares a serem exibidos.")
            num_vizinhos = st.slider("Número de vizinhos", 1, 10, 5, 1, help="Define o número de documentos similares no grafo.")
            
            if st.session_state.get('selected_id') != id_selecionado or st.session_state.get('num_vizinhos_cache') != num_vizinhos:
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
                    summaries_to_analyze = df.loc[list(node_indices)]['Resumo_LLM'].dropna()
                    full_text_summaries = "\n\n---\n\n".join(summaries_to_analyze)
                    if not full_text_summaries.strip():
                        st.warning("Não há resumos disponíveis para gerar a análise.")
                        st.session_state.analysis_result = ""
                    else:
                        with st.spinner('A IA está lendo e preparando a análise...'):
                            analysis = get_ai_synthesis(full_text_summaries)
                            st.session_state.analysis_result = analysis
                            st.session_state.analysis_cache[cache_key] = analysis
            
            if 'analysis_result' in st.session_state and st.session_state.analysis_result:
                with st.container(border=True):
                    st.subheader("Análise Gerada por IA")
                    st.markdown(st.session_state.analysis_result)
        else:
            st.info("Selecione um registro na tabela para visualizar trabalhos similares.")

def render_page_dashboard(df: pd.DataFrame):
    """Renderiza a página do Dashboard com visualizações sobre os dados."""
    st.title("Dashboard de Análise do Acervo")
    st.markdown("---")
    
    # --- Gráfico 1: Top 20 Assuntos Mais Frequentes ---
    st.subheader("Top 20 Assuntos Mais Frequentes")

    # Criar uma lista única com todos os assuntos (achatar a lista de listas)
    todos_assuntos = [assunto for sublista in df['Assuntos_Processados'] for assunto in sublista]

    if not todos_assuntos:
        st.warning("Não há dados de assuntos para exibir.")
        return

    # Contar as 20 ocorrências mais comuns
    contador_assuntos = Counter(todos_assuntos)
    top_20_assuntos = contador_assuntos.most_common(20)

    # Criar um DataFrame com os 20 assuntos mais comuns para plotagem
    df_top20 = pd.DataFrame(top_20_assuntos, columns=['Assunto', 'Quantidade'])

    # Criação do Gráfico de Barras Horizontais com Plotly Express
    fig = px.bar(
        df_top20.sort_values(by='Quantidade', ascending=True),
        x='Quantidade',
        y='Assunto',
        orientation='h',
        title='Top 20 Assuntos Mais Frequentes nos Trabalhos',
        text='Quantidade',
        labels={'Assunto': 'Assunto', 'Quantidade': 'Número de Ocorrências'}
    )

    # Personalizar a aparência do gráfico
    fig.update_traces(
        marker_color='#1f77b4',
        textposition='outside'
    )

    # Ajustes finos no layout para melhor legibilidade
    fig.update_layout(
        yaxis=dict(tickmode='linear'),
        xaxis_title="Número de Ocorrências",
        yaxis_title=None,
        margin=dict(l=200, r=20, t=50, b=50), # Aumenta a margem esquerda para caber os textos
        title_x=0.5
    )

    # Exibir o gráfico no Streamlit
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    # Futuras visualizações podem ser adicionadas aqui


def render_page_sobre():
    """Renderiza a página Sobre (placeholder)."""
    st.title("Sobre o Projeto")
    st.info("🚧 EM CONSTRUÇÃO 🚧")
    st.write("Esta página conterá informações sobre o projeto, os dados e as tecnologias utilizadas.")

# --------------------------------------------------------------------------
# FUNÇÃO PRINCIPAL DO APLICATIVO (ROTEADOR)
# --------------------------------------------------------------------------
def main():
    setup_page()

    # --- APLICA ESTILO CSS PARA A BARRA LATERAL ---
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                background-color: #0F5EDD;
            }
        </style>
    """, unsafe_allow_html=True)

    # --- NAVEGAÇÃO NA BARRA LATERAL COM BOTÕES ---
    # Inicializa o estado da página se não existir
    if 'page' not in st.session_state:
        st.session_state.page = "Consultas"

    with st.sidebar:
        st.title("📚 PPGDR Explorer")
        
        # Botões para navegação que alteram o estado da sessão
        if st.button("Consultas", use_container_width=True):
            st.session_state.page = "Consultas"
        if st.button("Dashboard", use_container_width=True):
            st.session_state.page = "Dashboard"
        if st.button("Sobre", use_container_width=True):
            st.session_state.page = "Sobre"
    
    # --- CARREGAMENTO E VALIDAÇÃO DOS DADOS (feito uma vez) ---
    df = load_data(CSV_DATA_PATH)
    embeddings = load_embeddings(EMBEDDINGS_PATH)

    if not validate_data(df, embeddings):
        st.warning("A aplicação não pode continuar devido a erros nos dados de entrada. Por favor, corrija os problemas acima.")
        st.stop()
    
    # --- PREPARAÇÃO DOS DADOS (feito uma vez) ---
    matriz_similaridade = calculate_similarity_matrix(embeddings)
    subject_options = prepare_subject_list(df)
    df = df.rename(columns={"Tipo_Documento": "Tipo de Documento"})
    df['index_original'] = df.index

    # --- ROTEAMENTO DE PÁGINAS BASEADO NO ESTADO DA SESSÃO ---
    if st.session_state.page == "Consultas":
        render_page_consultas(df, embeddings, matriz_similaridade, subject_options)
    elif st.session_state.page == "Dashboard":
        render_page_dashboard(df)
    elif st.session_state.page == "Sobre":
        render_page_sobre()

# --------------------------------------------------------------------------
# Ponto de entrada do script
# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
