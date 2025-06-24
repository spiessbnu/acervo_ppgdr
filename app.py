import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ast
import unicodedata
import re
import numpy as np
import nltk
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import RSLPStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import networkx as nx
import pickle
import textwrap
import openai
from openai import OpenAI

# Configuração da página
st.set_page_config(
    page_title="Acervo PPGDR v1",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para o design branco com destaques azuis
st.markdown("""
<style>
    .main {
        background-color: white;
    }
    
    .stButton > button {
        background-color: #0F5EDD;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #0A4BC7;
        color: white;
    }
    
    .stSelectbox > div > div {
        background-color: white;
        border: 1px solid #0F5EDD;
    }
    
    .stTextInput > div > div > input {
        border: 1px solid #0F5EDD;
    }
    
    .metric-card {
        background-color: #F8FAFF;
        border: 1px solid #0F5EDD;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .header-title {
        color: #0F5EDD;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .section-header {
        color: #0F5EDD;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
        border-bottom: 2px solid #0F5EDD;
        padding-bottom: 0.5rem;
    }
    
    .expandable-content {
        background-color: #F8FAFF;
        border-left: 4px solid #0F5EDD;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    
    .similarity-score {
        background-color: #0F5EDD;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .embedding-badge {
        background-color: #28a745;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
        font-size: 0.7rem;
        font-weight: bold;
    }
    
    .api-badge {
        background-color: #17a2b8;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
        font-size: 0.7rem;
        font-weight: bold;
    }
    
    .summary-box {
        background-color: #F8FAFF;
        border: 2px solid #0F5EDD;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .cluster-info {
        background-color: #E3F2FD;
        border-left: 4px solid #0F5EDD;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 5px 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Funções auxiliares
@st.cache_data
def load_data():
    """Carrega os dados do CSV"""
    try:
        df = pd.read_csv('dados_finais_com_resumo_llm.csv')
        return df
    except FileNotFoundError:
        st.error("Arquivo 'dados_finais_com_resumo_llm.csv' não encontrado. Por favor, faça o upload do arquivo.")
        return pd.DataFrame()

@st.cache_data
def load_embeddings():
    """Carrega os embeddings pré-calculados"""
    try:
        # Carrega o arquivo correto de embeddings
        embeddings = np.load('openai_embeddings_concatenado_large.npy')
        st.success(f"✅ Embeddings carregados com sucesso! Dimensão: {embeddings.shape}")
        return embeddings
    except FileNotFoundError:
        st.warning("⚠️ Arquivo 'openai_embeddings_concatenado_large.npy' não encontrado. A busca semântica avançada não estará disponível.")
        return None
    except Exception as e:
        st.error(f"❌ Erro ao carregar embeddings: {str(e)}")
        return None

@st.cache_resource
def init_openai_client():
    """Inicializa o cliente OpenAI a partir de st.secrets."""
    try:
        key = st.secrets.get("openai_api_key", "").strip()
        if not key:
            st.warning("⚠️ OpenAI API key não configurada em st.secrets.")
            return None
        client = OpenAI(api_key=key)
        # Optional: testar conexão — mas cuidado com rate limits
        # client.models.list()
        return client
    except Exception as e:
        st.error(f"❌ Erro ao inicializar cliente OpenAI: {e}")
        return None

def safe_literal_eval(s):
    """Converte string em lista de forma segura"""
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError, TypeError):
        return []

def remover_acentos(texto):
    """Remove acentos de uma string para ordenação"""
    texto_normalizado = unicodedata.normalize('NFD', texto)
    return "".join(c for c in texto_normalizado if not unicodedata.combining(c))

def normalizar_string(texto):
    """Converte para minúsculas e remove acentos"""
    if not isinstance(texto, str):
        return ""
    texto_normalizado = unicodedata.normalize('NFD', texto.lower())
    return "".join(c for c in texto_normalizado if not unicodedata.combining(c))

def preprocessar_texto(texto, usar_stemmer=False):
    """Limpa, tokeniza, remove stopwords e opcionalmente aplica stemming"""
    if not isinstance(texto, str):
        return ""
    
    texto_normalizado = normalizar_string(texto)
    texto_limpo = re.sub(r'[^a-zA-Z\s]', '', texto_normalizado)
    tokens = word_tokenize(texto_limpo)
    
    try:
        stopwords_pt = set(stopwords.words('portuguese'))
    except LookupError:
        nltk.download('stopwords')
        stopwords_pt = set(stopwords.words('portuguese'))
    
    tokens_filtrados = [p for p in tokens if p not in stopwords_pt and len(p) > 1]
    
    if usar_stemmer:
        stemmer = RSLPStemmer()
        tokens_finais = [stemmer.stem(p) for p in tokens_filtrados]
    else:
        tokens_finais = tokens_filtrados
    
    return " ".join(tokens_finais)

def buscar_e_rankear(query_texto, usar_stemmer, vectorizer, tfidf_matrix):
    """Processa uma query, calcula a similaridade e retorna um ranking"""
    query_processada = preprocessar_texto(query_texto, usar_stemmer=usar_stemmer)
    if not query_processada:
        return []
    
    query_vetor = vectorizer.transform([query_processada])
    similaridades = cosine_similarity(query_vetor, tfidf_matrix).flatten()
    indices_rankeados = np.argsort(-similaridades)
    
    return [(i, similaridades[i]) for i in indices_rankeados if similaridades[i] > 0.01]

def buscar_semantica_openai_real(query_texto, client, embeddings_documentos, model="text-embedding-3-large"):
    """
    Busca semântica real usando API OpenAI para gerar embedding da query.
    """
    if not query_texto.strip() or embeddings_documentos is None or client is None:
        return []
    
    try:
        # Gera o embedding para a query do usuário usando a API OpenAI
        response = client.embeddings.create(
            input=[query_texto], 
            model=model
        )
        query_embedding = response.data[0].embedding
        
        # Calcula a similaridade de cosseno entre o vetor da query e todos os vetores dos documentos
        similaridades = cosine_similarity([query_embedding], embeddings_documentos).flatten()
        
        # Ordena os resultados pela similaridade (do maior para o menor)
        indices_rankeados = np.argsort(-similaridades)
        
        # Retorna o ranking, filtrando resultados com score muito baixo
        return [(i, similaridades[i]) for i in indices_rankeados if similaridades[i] > 0.2]
        
    except Exception as e:
        st.error(f"Erro na busca semântica com API OpenAI: {str(e)}")
        return []

def gerar_sumario_cluster(documentos_cluster, client, model="gpt-4o-mini"):
    """
    Gera um sumário temático para um cluster de documentos usando OpenAI.
    """
    if not documentos_cluster or client is None:
        return "Sumário não disponível."
    
    try:
        # Concatena o texto de todos os documentos do cluster
        textos_concatenados = "\n\n--- PRÓXIMO DOCUMENTO ---\n\n".join(
            [f"Título: {doc['Título']}\nResumo: {doc['Resumo_LLM']}" for doc in documentos_cluster]
        )
        
        # Prompt para sumarização temática
        prompt_sistema = """
        Você é um assistente de pesquisa acadêmica altamente qualificado, especializado em identificar e sintetizar os temas centrais de um conjunto de documentos.

        Sua tarefa é ler todos os textos fornecidos a seguir, que são de trabalhos acadêmicos agrupados por similaridade semântica.
        Analise-os e gere um único parágrafo coeso que resuma os principais temas, problemas de pesquisa e metodologias abordadas pelo grupo como um todo.
        Não descreva cada trabalho individualmente. Foque na essência que conecta todos eles.

        Após o parágrafo de síntese, identifique e liste de 3 a 5 palavras-chave ou conceitos que são os pilares deste agrupamento.

        Formate sua resposta exatamente da seguinte forma:

        **Síntese Temática:**
        [Seu parágrafo de resumo aqui]

        **Termos Comuns:**
        - [Termo 1]
        - [Termo 2]
        - [Termo 3]
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": textos_concatenados}
            ],
            temperature=0.1,
            max_tokens=500,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Erro ao gerar sumário: {str(e)}"

def calcular_matriz_similaridade(df, embeddings=None):
    """Calcula matriz de similaridade entre documentos"""
    if embeddings is not None:
        # Usa embeddings se disponíveis
        try:
            matriz_similaridade = cosine_similarity(embeddings)
            return matriz_similaridade
        except Exception as e:
            st.warning(f"Erro ao usar embeddings, usando TF-IDF: {str(e)}")
    
    # Fallback para TF-IDF
    vectorizer, tfidf_matrix, _ = preparar_busca_semantica(df)
    matriz_similaridade = cosine_similarity(tfidf_matrix)
    return matriz_similaridade

        
def criar_rede_similaridade(df, embeddings=None, threshold=0.3, max_nodes=20):
    """Cria visualização de rede de similaridade usando Plotly"""
    
    # Limita o número de documentos para melhor visualização
    df_sample = df.head(max_nodes).copy()
    
    # Calcula matriz de similaridade
    if embeddings is not None:
        embeddings_sample = embeddings[:max_nodes]
        matriz_sim = cosine_similarity(embeddings_sample)
        st.info(f"🚀 Usando embeddings reais para calcular similaridade (dimensão: {embeddings_sample.shape})")
    else:
        # Usa TF-IDF como fallback
        textos_combinados = []
        for idx, row in df_sample.iterrows():
            texto = str(row['Título'])
            if 'Resumo_LLM' in df_sample.columns and pd.notna(row['Resumo_LLM']):
                texto += " " + str(row['Resumo_LLM'])
            textos_combinados.append(texto)
        
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([preprocessar_texto(texto) for texto in textos_combinados])
        matriz_sim = cosine_similarity(tfidf_matrix)
        st.info("📊 Usando TF-IDF para calcular similaridade")
    
    # Cria grafo
    G = nx.Graph()
    
    # Adiciona nós
    for i, (idx, row) in enumerate(df_sample.iterrows()):
        titulo_curto = row['Título'][:50] + "..." if len(row['Título']) > 50 else row['Título']
        G.add_node(i, 
                  titulo=titulo_curto,
                  titulo_completo=row['Título'],
                  autor=row['Autor'],
                  ano=row['Ano'],
                  tipo=row['Tipo_Documento'],
                  resumo=row['Resumo_LLM'] if 'Resumo_LLM' in row else '',
                  idx_original=idx)
    
    # Adiciona arestas baseadas na similaridade
    edges_added = 0
    for i in range(len(df_sample)):
        for j in range(i+1, len(df_sample)):
            similaridade = matriz_sim[i][j]
            if similaridade > threshold:
                G.add_edge(i, j, weight=similaridade)
                edges_added += 1
    
    if edges_added == 0:
        st.warning(f"⚠️ Nenhuma conexão encontrada com threshold {threshold}. Tente diminuir o valor.")
        return None, None
    
    # Layout do grafo
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    
    # Prepara dados para Plotly
    edge_x = []
    edge_y = []
    edge_weights = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        weight = G[edge[0]][edge[1]]['weight']
        edge_weights.append(weight)
    
    # Cria trace das arestas com espessura baseada no peso
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='rgba(15, 94, 221, 0.3)'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Prepara dados dos nós
    node_x = []
    node_y = []
    node_text = []
    node_info = []
    node_color = []
    node_size = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Informações do nó
        node_data = G.nodes[node]
        node_text.append(f"{node_data['titulo']}<br>({node_data['ano']})")
        
        # Calcular conexões e similaridade média
        neighbors = list(G.neighbors(node))
        num_connections = len(neighbors)
        avg_similarity = np.mean([G[node][neighbor]['weight'] for neighbor in neighbors]) if neighbors else 0
        
        node_info.append(
            f"<b>{node_data['titulo_completo']}</b><br>"
            f"Autor: {node_data['autor']}<br>"
            f"Ano: {node_data['ano']}<br>"
            f"Tipo: {node_data['tipo']}<br>"
            f"Conexões: {num_connections}<br>"
            f"Similaridade média: {avg_similarity:.3f}"
        )
        
        # Cor baseada no tipo de documento
        if node_data['tipo'] == 'Tese':
            node_color.append('#0A4BC7')
        else:
            node_color.append('#0F5EDD')
        
        # Tamanho baseado no número de conexões
        node_size.append(max(15, 10 + num_connections * 3))
    
    # Cria trace dos nós
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=node_info,
        text=[t.split('<br>')[0] for t in node_text],  # Só o título para o texto
        textposition="middle center",
        textfont=dict(size=8, color='white'),
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color='white'),
            opacity=0.9
        )
    )
    
    # Cria figura
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text=f'Rede de Similaridade - {len(df_sample)} Trabalhos ({edges_added} conexões)',
                x=0.5,
                font=dict(size=16, color='#0F5EDD')
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=60),
            annotations=[
                dict(
                    text=f"Nós azuis: Dissertações | Nós azul escuro: Teses<br>"
                         f"Conexões indicam similaridade > {threshold}<br>"
                         f"Tamanho do nó = número de conexões",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(size=10, color='#666')
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
    )
    
    return fig, G

# Inicialização do NLTK
@st.cache_resource
def init_nltk():
    """Inicializa recursos do NLTK"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    try:
        nltk.data.find('corpora/rslp')
    except LookupError:
        nltk.download('rslp')

# Função principal
def main():
    # Inicializar NLTK
    init_nltk()
    
    # Título principal
    st.markdown('<h1 class="header-title">📚 Acervo PPGDR v1</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem;">Plataforma de Consulta do Acervo de Teses e Dissertações do PPGDR-FURB</p>', unsafe_allow_html=True)
    
    # Carregar dados
    df = load_data()
    embeddings = load_embeddings()
    client = init_openai_client()
    
    if df.empty:
        st.warning("Nenhum dado foi carregado. Por favor, verifique se o arquivo CSV está disponível.")
        return
    
    # Preparar dados
    if 'Assuntos_Processados' not in df.columns:
        df['Assuntos_Processados'] = df['Assuntos_Lista'].apply(safe_literal_eval)
    
    # Status dos embeddings e API
    if embeddings is not None:
        st.sidebar.markdown(f'<span class="embedding-badge">✅ Embeddings Ativos</span>', unsafe_allow_html=True)
        st.sidebar.markdown(f"Dimensão: {embeddings.shape}")
    else:
        st.sidebar.markdown("⚠️ Embeddings não disponíveis")
    
    if client is not None:
        st.sidebar.markdown(f'<span class="api-badge">✅ API OpenAI Ativa</span>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown("⚠️ API OpenAI não configurada")
    
    # Sidebar para navegação
    st.sidebar.markdown('<h2 style="color: #0F5EDD;">🔍 Navegação</h2>', unsafe_allow_html=True)
    
    # Opções de navegação
    opcoes = ["🏠 Página Principal", "📊 Dashboard", "🕸️ Rede de Similaridade"]
    escolha = st.sidebar.selectbox("Selecione uma opção:", opcoes)
    
    if escolha == "🏠 Página Principal":
        pagina_principal(df, embeddings, client)
    elif escolha == "📊 Dashboard":
        dashboard(df)
    elif escolha == "🕸️ Rede de Similaridade":
        pagina_rede_similaridade(df, embeddings, client)

def pagina_principal(df, embeddings, client):
    """Página principal com busca e visualização dos dados"""
    
    # Métricas gerais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #0F5EDD; margin: 0;">Total de Trabalhos</h3>
            <h2 style="margin: 0;">{}</h2>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        dissertacoes = len(df[df['Tipo_Documento'] == 'Dissertação'])
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #0F5EDD; margin: 0;">Dissertações</h3>
            <h2 style="margin: 0;">{}</h2>
        </div>
        """.format(dissertacoes), unsafe_allow_html=True)
    
    with col3:
        teses = len(df[df['Tipo_Documento'] == 'Tese'])
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #0F5EDD; margin: 0;">Teses</h3>
            <h2 style="margin: 0;">{}</h2>
        </div>
        """.format(teses), unsafe_allow_html=True)
    
    with col4:
        anos = df['Ano'].nunique()
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #0F5EDD; margin: 0;">Anos de Produção</h3>
            <h2 style="margin: 0;">{}</h2>
        </div>
        """.format(anos), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Seção de busca
    st.markdown('<h2 class="section-header">🔍 Busca no Acervo</h2>', unsafe_allow_html=True)
    
    # Abas para diferentes tipos de busca
    if embeddings is not None and client is not None:
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Busca por Assunto", 
            "🔤 Busca por Palavras-chave", 
            "🧠 Busca Semântica (TF-IDF)", 
            "🚀 Busca Semântica OpenAI"
        ])
        
        with tab4:
            busca_semantica_openai(df, embeddings, client)
    elif embeddings is not None:
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Busca por Assunto", 
            "🔤 Busca por Palavras-chave", 
            "🧠 Busca Semântica (TF-IDF)", 
            "🚀 Busca Semântica Avançada"
        ])
        
        with tab4:
            busca_semantica_avancada(df, embeddings)
    else:
        tab1, tab2, tab3 = st.tabs([
            "📋 Busca por Assunto", 
            "🔤 Busca por Palavras-chave", 
            "🧠 Busca Semântica"
        ])
    
    with tab1:
        busca_por_assunto(df)
    
    with tab2:
        busca_por_palavras_chave(df)
    
    with tab3:
        busca_semantica(df)
    
    st.markdown("---")
    
    # Exibição do dataframe completo
    st.markdown('<h2 class="section-header">📊 Dados Gerais do Acervo</h2>', unsafe_allow_html=True)
    
    # Filtros para o dataframe
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tipos_selecionados = st.multiselect(
            "Filtrar por Tipo de Documento:",
            options=df['Tipo_Documento'].unique(),
            default=df['Tipo_Documento'].unique()
        )
    
    with col2:
        anos_selecionados = st.multiselect(
            "Filtrar por Ano:",
            options=sorted(df['Ano'].unique()),
            default=sorted(df['Ano'].unique())
        )
    
    with col3:
        orientadores_selecionados = st.multiselect(
            "Filtrar por Orientador:",
            options=sorted(df['Orientador'].unique()) if 'Orientador' in df.columns else [],
            default=sorted(df['Orientador'].unique()) if 'Orientador' in df.columns else []
        )
    
    # Aplicar filtros
    df_filtrado = df[
        (df['Tipo_Documento'].isin(tipos_selecionados)) &
        (df['Ano'].isin(anos_selecionados))
    ]
    
    if 'Orientador' in df.columns and orientadores_selecionados:
        df_filtrado = df_filtrado[df_filtrado['Orientador'].isin(orientadores_selecionados)]
    
    # Exibir dataframe filtrado
    st.dataframe(
        df_filtrado[['Autor', 'Título', 'Ano', 'Tipo_Documento', 'Orientador'] if 'Orientador' in df.columns else ['Autor', 'Título', 'Ano', 'Tipo_Documento']],
        use_container_width=True,
        height=400
    )
    
    # Detalhes expandíveis para cada registro
    if not df_filtrado.empty:
        st.markdown('<h3 class="section-header">📖 Detalhes dos Trabalhos</h3>', unsafe_allow_html=True)
        
        for idx, row in df_filtrado.iterrows():
            with st.expander(f"+ Detalhes: {row['Título'][:100]}..."):
                exibir_detalhes_trabalho(row, df, embeddings)

def busca_por_assunto(df):
    """Implementa a busca por assunto usando dropdown"""
    
    # Gerar lista de assuntos únicos e ordenados
    todos_assuntos = [assunto for sublista in df['Assuntos_Processados'] for assunto in sublista]
    lista_unica = list(set(todos_assuntos))
    lista_ordenada = sorted(lista_unica, key=lambda texto: remover_acentos(texto.lower()))
    
    # Dropdown para seleção de assunto
    assunto_selecionado = st.selectbox(
        "Selecione um assunto:",
        options=['-- Selecione um Assunto --'] + lista_ordenada,
        key="busca_assunto"
    )
    
    if assunto_selecionado != '-- Selecione um Assunto --':
        # Filtrar dataframe pelo assunto selecionado
        filtro_booleano = df['Assuntos_Processados'].apply(lambda lista: assunto_selecionado in lista)
        df_filtrado = df[filtro_booleano]
        
        if not df_filtrado.empty:
            st.success(f"Encontrados {len(df_filtrado)} trabalho(s) para o assunto: '{assunto_selecionado}'")
            
            # Exibir resultados
            for idx, row in df_filtrado.iterrows():
                with st.expander(f"📄 {row['Título']}"):
                    exibir_detalhes_trabalho(row, df)
        else:
            st.warning("Nenhum trabalho encontrado para o assunto selecionado.")

def busca_por_palavras_chave(df):
    """Implementa a busca por palavras-chave"""
    
    # Campo de entrada para palavras-chave
    palavras_chave = st.text_input(
        "Digite palavras-chave para busca:",
        placeholder="Ex: desenvolvimento regional, sustentabilidade, economia",
        key="busca_palavras"
    )
    
    if palavras_chave:
        # Buscar nos títulos e resumos
        palavras_normalizadas = normalizar_string(palavras_chave)
        
        # Filtrar por título
        filtro_titulo = df['Título'].apply(
            lambda x: any(palavra in normalizar_string(str(x)) for palavra in palavras_normalizadas.split())
        )
        
        # Filtrar por resumo LLM
        filtro_resumo = pd.Series([False] * len(df))
        if 'Resumo_LLM' in df.columns:
            filtro_resumo = df['Resumo_LLM'].apply(
                lambda x: any(palavra in normalizar_string(str(x)) for palavra in palavras_normalizadas.split())
            )
        
        # Combinar filtros
        df_filtrado = df[filtro_titulo | filtro_resumo]
        
        if not df_filtrado.empty:
            st.success(f"Encontrados {len(df_filtrado)} trabalho(s) para as palavras-chave: '{palavras_chave}'")
            
            # Exibir resultados
            for idx, row in df_filtrado.iterrows():
                with st.expander(f"📄 {row['Título']}"):
                    exibir_detalhes_trabalho(row, df)
        else:
            st.warning("Nenhum trabalho encontrado para as palavras-chave especificadas.")

@st.cache_resource
def preparar_busca_semantica(df):
    """Prepara os dados para busca semântica"""
    
    # Combinar título e resumo LLM para busca semântica
    textos_combinados = []
    for idx, row in df.iterrows():
        texto = str(row['Título'])
        if 'Resumo_LLM' in df.columns and pd.notna(row['Resumo_LLM']):
            texto += " " + str(row['Resumo_LLM'])
        textos_combinados.append(texto)
    
    # Preprocessar textos
    textos_processados = [preprocessar_texto(texto) for texto in textos_combinados]
    
    # Criar matriz TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(textos_processados)
    
    return vectorizer, tfidf_matrix, textos_processados

def busca_semantica(df):
    """Implementa a busca semântica usando TF-IDF e similaridade de cosseno"""
    
    # Campo de entrada para busca semântica
    query_semantica = st.text_input(
        "Digite sua consulta para busca semântica:",
        placeholder="Ex: impactos ambientais do desenvolvimento urbano",
        key="busca_semantica"
    )
    
    if query_semantica:
        try:
            # Preparar dados para busca semântica
            vectorizer, tfidf_matrix, textos_processados = preparar_busca_semantica(df)
            
            # Realizar busca
            resultados = buscar_e_rankear(query_semantica, False, vectorizer, tfidf_matrix)
            
            if resultados:
                st.success(f"Encontrados {len(resultados)} trabalho(s) relevantes para: '{query_semantica}'")
                
                # Exibir resultados ordenados por relevância
                for i, (idx, score) in enumerate(resultados[:10]):  # Top 10 resultados
                    row = df.iloc[idx]
                    with st.expander(f"📄 {row['Título']} (Relevância: {score:.3f})"):
                        exibir_detalhes_trabalho(row, df)
            else:
                st.warning("Nenhum trabalho encontrado para a consulta especificada.")
                
        except Exception as e:
            st.error(f"Erro na busca semântica: {str(e)}")

def busca_semantica_avancada(df, embeddings):
    """Implementa a busca semântica usando embeddings pré-calculados (sem API)"""
    
    st.info("🚀 **Busca Semântica Avançada** - Usando embeddings pré-calculados")
    st.warning("⚠️ **Limitação**: Esta versão simula a busca semântica pois não há API OpenAI configurada.")
    
    # Campo de entrada para busca semântica avançada
    query_avancada = st.text_input(
        "Digite sua consulta para busca semântica avançada:",
        placeholder="Ex: desenvolvimento sustentável na região sul do Brasil",
        key="busca_semantica_avancada"
    )
    
    if query_avancada:
        try:
            # Usar TF-IDF como simulação
            vectorizer, tfidf_matrix, _ = preparar_busca_semantica(df)
            resultados = buscar_e_rankear(query_avancada, False, vectorizer, tfidf_matrix)
            
            if resultados:
                st.success(f"Encontrados {len(resultados)} trabalho(s) relevantes para: '{query_avancada}'")
                
                # Exibir resultados ordenados por relevância
                for i, (idx, score) in enumerate(resultados[:10]):
                    row = df.iloc[idx]
                    
                    # Criar card com score destacado
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.markdown(f"**{i+1}. {row['Título']}**")
                        st.markdown(f"*{row['Autor']} ({row['Ano']})*")
                    
                    with col2:
                        st.markdown(f'<span class="similarity-score">Similaridade: {score:.3f}</span>', 
                                  unsafe_allow_html=True)
                    
                    # Resumo formatado
                    if 'Resumo_LLM' in df.columns and pd.notna(row['Resumo_LLM']):
                        resumo_formatado = textwrap.fill(str(row['Resumo_LLM']), width=100)
                        st.markdown(f"**Resumo:** {resumo_formatado}")
                    
                    st.markdown("---")
                    
            else:
                st.warning("Nenhum trabalho encontrado para a consulta especificada.")
                
        except Exception as e:
            st.error(f"Erro na busca semântica avançada: {str(e)}")

def busca_semantica_openai(df, embeddings, client):
    """Implementa a busca semântica real usando API OpenAI"""
    
    st.info("🚀 **Busca Semântica OpenAI** - Usando API OpenAI para gerar embeddings da query em tempo real")
    
    # Campo de entrada para busca semântica OpenAI
    query_openai = st.text_input(
        "Digite sua consulta para busca semântica OpenAI:",
        placeholder="Ex: desenvolvimento sustentável na região sul do Brasil",
        key="busca_semantica_openai"
    )
    
    if query_openai:
        with st.spinner("Gerando embedding da query e calculando similaridades..."):
            try:
                # Realizar busca semântica real com API OpenAI
                resultados = buscar_semantica_openai_real(query_openai, client, embeddings)
                
                if resultados:
                    st.success(f"Encontrados {len(resultados)} trabalho(s) altamente relevantes para: '{query_openai}'")
                    
                    # Exibir resultados ordenados por relevância
                    for i, (idx, score) in enumerate(resultados[:10]):
                        row = df.iloc[idx]
                        
                        # Criar card com score destacado
                        col1, col2 = st.columns([4, 1])
                        
                        with col1:
                            st.markdown(f"**{i+1}. {row['Título']}**")
                            st.markdown(f"*{row['Autor']} ({row['Ano']})*")
                        
                        with col2:
                            st.markdown(f'<span class="similarity-score">Similaridade: {score:.3f}</span>', 
                                      unsafe_allow_html=True)
                        
                        # Resumo formatado
                        if 'Resumo_LLM' in df.columns and pd.notna(row['Resumo_LLM']):
                            resumo_formatado = textwrap.fill(str(row['Resumo_LLM']), width=100)
                            st.markdown(f"**Resumo:** {resumo_formatado}")
                        
                        st.markdown("---")
                        
                else:
                    st.warning("Nenhum trabalho encontrado para a consulta especificada.")
                    
            except Exception as e:
                st.error(f"Erro na busca semântica OpenAI: {str(e)}")

def pagina_rede_similaridade(df, embeddings, client):
    """Página da rede de similaridade com sumarização temática"""
    
    st.markdown('<h1 class="header-title">🕸️ Rede de Similaridade</h1>', unsafe_allow_html=True)
    st.markdown("Visualização interativa das conexões entre trabalhos baseada em similaridade semântica.")
    
    # Controles da rede
    col1, col2, col3 = st.columns(3)
    
    with col1:
        threshold = st.slider(
            "Threshold de Similaridade:",
            min_value=0.1,
            max_value=0.8,
            value=0.3,
            step=0.05,
            help="Conexões só aparecem se a similaridade for maior que este valor"
        )
    
    with col2:
        max_nodes = st.slider(
            "Máximo de Trabalhos:",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
            help="Número máximo de trabalhos a exibir na rede"
        )
    
    with col3:
        filtro_tipo = st.selectbox(
            "Filtrar por Tipo:",
            options=["Todos", "Dissertação", "Tese"],
            help="Filtrar trabalhos por tipo de documento"
        )
    
    # Aplicar filtro de tipo se necessário
    df_filtrado = df.copy()
    if filtro_tipo != "Todos":
        df_filtrado = df_filtrado[df_filtrado['Tipo_Documento'] == filtro_tipo]
    
    if len(df_filtrado) < 2:
        st.warning("Não há trabalhos suficientes para criar uma rede de similaridade.")
        return
    
    # Gerar e exibir a rede
    with st.spinner("Gerando rede de similaridade..."):
        try:
            fig, grafo = criar_rede_similaridade(df_filtrado, embeddings, threshold, max_nodes)
            
            if fig is not None and grafo is not None:
                st.plotly_chart(fig, use_container_width=True, height=600)
                
                # Informações adicionais
                st.markdown("### 📊 Informações da Rede")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Trabalhos Exibidos", min(max_nodes, len(df_filtrado)))
                
                with col2:
                    # Calcular número de conexões
                    conexoes = len(grafo.edges())
                    st.metric("Conexões", conexoes)
                
                with col3:
                    num_nodes = len(grafo.nodes())
                    densidade = (2 * conexoes) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
                    st.metric("Densidade da Rede", f"{densidade:.2%}")
                
                # Sumarização Temática
                if client is not None and len(grafo.nodes()) > 0:
                    st.markdown("### 🤖 Sumarização Temática do Cluster")
                    
                    if st.button("🚀 Gerar Sumário Temático do Cluster", type="primary"):
                        with st.spinner("Gerando sumário temático com OpenAI..."):
                            try:
                                # Preparar documentos do cluster
                                documentos_cluster = []
                                for node in grafo.nodes():
                                    node_data = grafo.nodes[node]
                                    idx_original = node_data['idx_original']
                                    row = df_filtrado.iloc[idx_original]
                                    documentos_cluster.append({
                                        'Título': row['Título'],
                                        'Resumo_LLM': row['Resumo_LLM'] if 'Resumo_LLM' in row else ''
                                    })
                                
                                # Gerar sumário
                                sumario = gerar_sumario_cluster(documentos_cluster, client)
                                
                                # Exibir sumário
                                st.markdown(f"""
                                <div class="summary-box">
                                    <h4 style="color: #0F5EDD; margin-top: 0;">📋 Sumário Temático do Cluster</h4>
                                    {sumario}
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Informações do cluster
                                st.markdown(f"""
                                <div class="cluster-info">
                                    <strong>Informações do Cluster:</strong><br>
                                    • Número de trabalhos analisados: {len(documentos_cluster)}<br>
                                    • Threshold de similaridade: {threshold}<br>
                                    • Conexões na rede: {conexoes}<br>
                                    • Modelo usado: GPT-4o-mini
                                </div>
                                """, unsafe_allow_html=True)
                                
                            except Exception as e:
                                st.error(f"Erro ao gerar sumário temático: {str(e)}")
                else:
                    if client is None:
                        st.info("🔧 **Sumarização Temática**: Configure a API OpenAI para habilitar esta funcionalidade.")
                    else:
                        st.info("📊 Gere uma rede de similaridade para usar a sumarização temática.")
                
                # Explicação
                st.markdown("""
                ### 💡 Como Interpretar a Rede
                
                - **Nós (círculos)**: Cada nó representa um trabalho (tese ou dissertação)
                - **Cores**: Azul claro = Dissertações, Azul escuro = Teses  
                - **Conexões (linhas)**: Indicam alta similaridade semântica entre trabalhos
                - **Tamanho dos nós**: Proporcional ao número de conexões (centralidade)
                - **Posicionamento**: Trabalhos similares tendem a ficar próximos
                - **Clusters**: Grupos de trabalhos conectados indicam temas relacionados
                
                ### 🤖 Sumarização Temática
                
                - **Funcionalidade**: Analisa todos os trabalhos da rede e gera um resumo temático
                - **Tecnologia**: Usa GPT-4o-mini para identificar temas centrais e termos comuns
                - **Resultado**: Síntese dos principais temas e palavras-chave do cluster
                
                **Dica**: Passe o mouse sobre os nós para ver detalhes dos trabalhos!
                """)
            
        except Exception as e:
            st.error(f"Erro ao gerar rede de similaridade: {str(e)}")

def exibir_detalhes_trabalho(row, df_completo, embeddings=None):
    """Exibe os detalhes de um trabalho específico"""
    
    # Informações básicas
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Autor:** {row['Autor']}")
        st.markdown(f"**Ano:** {row['Ano']}")
        st.markdown(f"**Tipo:** {row['Tipo_Documento']}")
        if 'Orientador' in row:
            st.markdown(f"**Orientador:** {row['Orientador']}")
    
    with col2:
        if 'Assuntos_Lista' in row:
            st.markdown("**Assuntos:**")
            assuntos = safe_literal_eval(row['Assuntos_Lista'])
            for assunto in assuntos[:5]:  # Mostrar apenas os primeiros 5
                st.markdown(f"• {assunto}")
    
    # Resumo LLM
    if 'Resumo_LLM' in row and pd.notna(row['Resumo_LLM']):
        st.markdown("**Resumo:**")
        st.markdown(f'<div class="expandable-content">{row["Resumo_LLM"]}</div>', unsafe_allow_html=True)
    
    # Trabalhos similares
    if embeddings is not None:
        st.markdown("**Trabalhos Similares:**")
        try:
            # Encontrar índice do trabalho atual
            idx_atual = df_completo.index[df_completo['Título'] == row['Título']].tolist()
            if idx_atual:
                idx_atual = idx_atual[0]
                
                # Calcular similaridade com todos os outros trabalhos
                matriz_sim = calcular_matriz_similaridade(df_completo, embeddings)
                similaridades = matriz_sim[idx_atual]
                
                # Encontrar os 3 mais similares (excluindo o próprio trabalho)
                indices_similares = np.argsort(-similaridades)[1:4]  # Exclui o primeiro (próprio trabalho)
                
                for i, idx_similar in enumerate(indices_similares):
                    if similaridades[idx_similar] > 0.3:  # Threshold mínimo
                        trabalho_similar = df_completo.iloc[idx_similar]
                        score = similaridades[idx_similar]
                        st.markdown(f"• **{trabalho_similar['Título']}** (Similaridade: {score:.3f})")
                        st.markdown(f"  *{trabalho_similar['Autor']} ({trabalho_similar['Ano']})*")
        except Exception as e:
            st.info(f"Análise de similaridade não disponível: {str(e)}")
    else:
        st.markdown("**Rede de Similaridade:**")
        st.info("Funcionalidade de rede de similaridade estará disponível quando os embeddings forem carregados.")

def dashboard(df):
    """Página do dashboard com visualizações"""
    
    st.markdown('<h1 class="header-title">📊 Dashboard do Acervo</h1>', unsafe_allow_html=True)
    
    # Gráfico 1: Quantidade de trabalhos por tipo
    st.markdown('<h2 class="section-header">📈 Distribuição por Tipo de Documento</h2>', unsafe_allow_html=True)
    
    contagem_docs = df['Tipo_Documento'].value_counts().reset_index()
    contagem_docs.columns = ['Tipo_Documento', 'Quantidade']
    
    fig1 = px.bar(
        contagem_docs,
        x='Tipo_Documento',
        y='Quantidade',
        title='Quantidade de Trabalhos por Tipo de Documento',
        text='Quantidade',
        color='Tipo_Documento',
        color_discrete_map={
            'Dissertação': '#0F5EDD',
            'Tese': '#0A4BC7'
        }
    )
    
    fig1.update_traces(textposition='outside', textfont_size=12)
    fig1.update_layout(
        showlegend=False,
        title_x=0.5,
        font=dict(family="Arial, sans-serif", size=12)
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Gráfico 2: Top 20 assuntos mais frequentes
    st.markdown('<h2 class="section-header">🏷️ Assuntos Mais Frequentes</h2>', unsafe_allow_html=True)
    
    todos_assuntos = [assunto for sublista in df['Assuntos_Processados'] for assunto in sublista]
    contador_assuntos = Counter(todos_assuntos)
    top_20_assuntos = contador_assuntos.most_common(20)
    
    df_top20 = pd.DataFrame(top_20_assuntos, columns=['Assunto', 'Quantidade'])
    
    fig2 = px.bar(
        df_top20.sort_values(by='Quantidade', ascending=True),
        x='Quantidade',
        y='Assunto',
        orientation='h',
        title='Top 20 Assuntos Mais Frequentes nos Trabalhos',
        text='Quantidade'
    )
    
    fig2.update_traces(marker_color='#0F5EDD', textposition='outside')
    fig2.update_layout(
        yaxis=dict(tickmode='linear'),
        xaxis_title="Número de Ocorrências",
        yaxis_title=None,
        margin=dict(l=150, r=20, t=50, b=50),
        title_x=0.5
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Gráfico 3: Produção anual
    st.markdown('<h2 class="section-header">📅 Produção Anual</h2>', unsafe_allow_html=True)
    
    contagem_agrupada = df.groupby(['Ano', 'Tipo_Documento']).size().reset_index(name='Quantidade')
    contagem_agrupada = contagem_agrupada.sort_values('Ano')
    
    fig3 = px.bar(
        contagem_agrupada,
        x='Ano',
        y='Quantidade',
        color='Tipo_Documento',
        title='Produção Anual: Teses vs. Dissertações',
        barmode='group',
        color_discrete_map={
            'Dissertação': '#0F5EDD',
            'Tese': '#0A4BC7'
        }
    )
    
    fig3.update_layout(
        title_x=0.5,
        font=dict(family="Arial, sans-serif", size=12),
        legend_title_text='Tipo de Documento'
    )
    
    fig3.update_xaxes(type='category')
    
    st.plotly_chart(fig3, use_container_width=True)
    
    # Estatísticas adicionais
    st.markdown('<h2 class="section-header">📊 Estatísticas Gerais</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #0F5EDD; margin: 0;">Período de Produção</h3>
            <h2 style="margin: 0;">{} - {}</h2>
        </div>
        """.format(df['Ano'].min(), df['Ano'].max()), unsafe_allow_html=True)
    
    with col2:
        media_anual = len(df) / (df['Ano'].max() - df['Ano'].min() + 1)
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #0F5EDD; margin: 0;">Média Anual</h3>
            <h2 style="margin: 0;">{:.1f}</h2>
        </div>
        """.format(media_anual), unsafe_allow_html=True)
    
    with col3:
        total_assuntos = len(set([assunto for sublista in df['Assuntos_Processados'] for assunto in sublista]))
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #0F5EDD; margin: 0;">Assuntos Únicos</h3>
            <h2 style="margin: 0;">{}</h2>
        </div>
        """.format(total_assuntos), unsafe_allow_html=True)

if __name__ == "__main__":
    main()

