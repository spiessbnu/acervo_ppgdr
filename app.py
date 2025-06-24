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
    """Inicializa o cliente OpenAI"""
    try:
        # Tenta carregar a chave da API dos secrets do Streamlit
        if "openai_api_key" in st.secrets:
            client = OpenAI(api_key=st.secrets["openai_api_key"])
            # Teste simples para verificar se a chave funciona
            try:
                client.models.list()
                return client
            except Exception as e:
                st.error(f"❌ Chave da API OpenAI inválida: {str(e)}")
                return None
        else:
            st.warning("⚠️ Chave da API OpenAI não configurada. Funcionalidades avançadas não estarão disponíveis.")
            return None
    except Exception as e:
        st.error(f"❌ Erro ao inicializar cliente OpenAI: {str(e)}")
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
          
(Content truncated due to size limit. Use line ranges to read in chunks)
