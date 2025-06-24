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
from pathlib import Path
from openai import OpenAI

# -----------------------------------------------------------------------------
# Configuração da Página e Estilo
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Acervo PPGDR",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Opcional: Carregar CSS externo, se existir
STYLE_PATH = Path(__file__).with_name("style.css") if "__file__" in locals() else None
if STYLE_PATH and STYLE_PATH.exists():
    st.markdown(f"<style>{STYLE_PATH.read_text()}</style>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Carregamento de Dados e Modelos (com cache)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Carregando dados do acervo...")
def load_data():
    """Carrega e pré-processa o DataFrame principal."""
    df = pd.read_csv("dados_finais_com_resumo_llm.csv")
    if "Assuntos_Processados" not in df.columns or df["Assuntos_Processados"].dtype != 'object':
        df["Assuntos_Processados"] = df["Assuntos_Lista"].apply(safe_literal_eval)
    # Garante um índice único e reiniciado
    df.reset_index(drop=True, inplace=True)
    return df

@st.cache_data(show_spinner="Carregando embeddings vetoriais...")
def load_embeddings():
    """Carrega os embeddings de um arquivo .npy."""
    try:
        return np.load("openai_embeddings_concatenado_large.npy")
    except FileNotFoundError:
        st.warning("Arquivo de embeddings não encontrado. A funcionalidade de similaridade semântica estará desabilitada.")
        return None

@st.cache_data(show_spinner="Calculando matriz de similaridade...")
def get_similarity_matrix(_embeddings):
    """Calcula e cacheia a matriz de similaridade de cossenos."""
    if _embeddings is not None:
        return cosine_similarity(_embeddings)
    return None

@st.cache_resource(show_spinner=False)
def init_nltk():
    """Baixa pacotes NLTK necessários se não existirem."""
    packages = ["punkt", "stopwords", "rslp"]
    for pkg_id in packages:
        try:
            nltk.data.find(f"tokenizers/{pkg_id}")
        except LookupError:
            nltk.download(pkg_id, quiet=True)

# -----------------------------------------------------------------------------
# Funções de Pré-processamento de Texto
# -----------------------------------------------------------------------------
def safe_literal_eval(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError, TypeError):
        return []

def normalizar_string(texto):
    if not isinstance(texto, str): return ""
    texto = unicodedata.normalize('NFD', texto.lower())
    return re.sub(r'[\u0300-\u036f]', '', texto)

def preprocessar_texto(texto, usar_stemmer=False):
    if not isinstance(texto, str): return ""
    texto_normalizado = normalizar_string(texto)
    texto_normalizado = re.sub(r"[^a-zA-Z\s]", "", texto_normalizado)
    tokens = word_tokenize(texto_normalizado, language='portuguese')
    stop_words = set(stopwords.words('portuguese'))
    tokens_filtrados = [t for t in tokens if t not in stop_words and len(t) > 1]
    if usar_stemmer:
        stemmer = RSLPStemmer()
        tokens_filtrados = [stemmer.stem(t) for t in tokens_filtrados]
    return " ".join(tokens_filtrados)

# -----------------------------------------------------------------------------
# Funções de Busca (Otimizadas)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Preparando modelo de busca TF-IDF...")
def get_tfidf_model(_df):
    """Cria e cacheia o vetorizador e a matriz TF-IDF."""
    textos = [
        preprocessar_texto(f"{row['Título']} {row.get('Resumo_LLM', '')}")
        for _, row in _df.iterrows()
    ]
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(textos)
    return vectorizer, tfidf_matrix

def buscar_por_similaridade_tfidf(query, vectorizer, tfidf_matrix):
    """Executa a busca TF-IDF e retorna os índices e scores."""
    if not query.strip(): return []
    query_preprocessed = preprocessar_texto(query)
    if not query_preprocessed: return []
    
    query_vec = vectorizer.transform([query_preprocessed])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Adiciona uma pequena tolerância para a busca
    ranked_indices = np.argsort(-similarities)
    return [
        (idx, similarities[idx])
        for idx in ranked_indices if similarities[idx] > 0.01
    ]

# -----------------------------------------------------------------------------
# Componentes da Interface (UI)
# -----------------------------------------------------------------------------
def exibir_detalhes_trabalho(row, df_full, similarity_matrix):
    """Mostra os detalhes de um trabalho selecionado."""
    with st.container(border=True):
        st.markdown(f"### {row['Título']}")
        
        meta_cols = st.columns(3)
        meta_cols[0].markdown(f"**Autor:** {row['Autor']}")
        meta_cols[1].markdown(f"**Ano:** {row['Ano']}")
        meta_cols[2].markdown(f"**Tipo:** {row['Tipo_Documento']}")
        
        if pd.notna(row.get("Resumo_LLM")) and row.get("Resumo_LLM"):
            with st.expander("📄 Ver resumo", expanded=False):
                st.write(row["Resumo_LLM"])

        if similarity_matrix is not None:
            with st.expander("🔗 Ver trabalhos similares", expanded=False):
                mostrar_similares_openai(row, df_full, similarity_matrix)
        
        with st.expander("🗂️ Ver metadados completos", expanded=False):
            st.json(row.to_dict(), expanded=False)

def mostrar_similares_openai(row, df_full, similarity_matrix):
    """Exibe os trabalhos mais similares com base nos embeddings da OpenAI."""
    try:
        idx = row.name
        sim_scores = similarity_matrix[idx]
        top_indices = np.argsort(-sim_scores)[1:4] # Exclui o próprio item
        
        st.write("Trabalhos com maior similaridade semântica (Embeddings):")
        for i in top_indices:
            # Adiciona um limiar para não mostrar resultados irrelevantes
            if sim_scores[i] > 0.30:
                similar_row = df_full.iloc[i]
                st.markdown(
                    f"- **{similar_row['Título']}** ({similar_row['Ano']}) - _Similaridade: {sim_scores[i]:.2%}_"
                )
    except Exception as e:
        st.error(f"Ocorreu um erro ao buscar trabalhos similares: {e}")

# -----------------------------------------------------------------------------
# Abas da Aplicação
# -----------------------------------------------------------------------------
def aba_pesquisa(df, embeddings, similarity_matrix, tfidf_vectorizer, tfidf_matrix):
    """Renderiza a aba principal de pesquisa e visualização."""
    st.markdown("## 📖 Acervo de Trabalhos Acadêmicos")
    
    # Métricas gerais
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Trabalhos", len(df))
    col2.metric("Período Coberto", f"{df['Ano'].min()} - {df['Ano'].max()}")
    col3.metric("Total de Assuntos", len(set(a for sub in df["Assuntos_Processados"] for a in sub)))
    
    st.divider()

    # --- Filtros e Pesquisa ---
    with st.expander("🔎 **Filtros e Pesquisa**", expanded=True):
        
        modo_busca = st.radio(
            "Modo de busca:",
            ["Palavras-chave no Título/Resumo", "Assunto Específico", "Busca Semântica (TF-IDF)"],
            horizontal=True,
            key="modo_busca"
        )

        df_filtrado = df.copy()

        if modo_busca == "Assunto Específico":
            todos_assuntos = sorted({a for sub in df["Assuntos_Processados"] for a in sub})
            assunto_selecionado = st.selectbox(
                "Selecione o assunto:",
                options=["-- Nenhum --"] + todos_assuntos,
                index=0
            )
            if assunto_selecionado != "-- Nenhum --":
                df_filtrado = df[df["Assuntos_Processados"].apply(lambda lst: assunto_selecionado in lst)]
        
        else: # Palavras-chave ou Semântica
            query = st.text_input(
                "Digite os termos da sua pesquisa:",
                key="query_text",
                placeholder="Ex: desenvolvimento regional sustentável"
            )
            if query:
                if modo_busca == "Palavras-chave no Título/Resumo":
                    q_norm = normalizar_string(query)
                    mask = (
                        df["Título"].str.lower().str.contains(q_norm, na=False) |
                        df["Resumo_LLM"].str.lower().str.contains(q_norm, na=False)
                    )
                    df_filtrado = df[mask]
                
                elif modo_busca == "Busca Semântica (TF-IDF)":
                    resultados = buscar_por_similaridade_tfidf(query, tfidf_vectorizer, tfidf_matrix)
                    indices = [i for i, _ in resultados[:50]] # Limita a 50 resultados
                    df_filtrado = df.iloc[indices]

    # --- Exibição dos Resultados ---
    st.markdown("### Resultados")
    
    if df_filtrado.empty:
        st.info("Nenhum trabalho encontrado com os filtros aplicados.")
    else:
        # Colunas a serem exibidas no dataframe principal
        cols_display = ['Título', 'Autor', 'Ano', 'Tipo_Documento']
        df_display = df_filtrado[cols_display].copy()

        # Usar on_select para uma experiência interativa
        selecao = st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row"
        )
        
        st.markdown(f"Exibindo **{len(df_filtrado)}** de **{len(df)}** trabalhos.")
        
        st.divider()

        # --- Exibição dos Detalhes ---
        if selecao.selection.rows:
            # Obter o índice original do DataFrame filtrado
            selected_row_index_in_filtered = selecao.selection.rows[0]
            # Obter o índice original do DataFrame completo
            original_index = df_filtrado.index[selected_row_index_in_filtered]
            
            st.session_state['selected_item_index'] = original_index
            
        # Exibe o item selecionado (se houver)
        if 'selected_item_index' in st.session_state and st.session_state['selected_item_index'] is not None:
            try:
                # Recupera a linha completa do dataframe original
                selected_row = df.loc[st.session_state['selected_item_index']]
                st.markdown("### Detalhes do Trabalho Selecionado")
                exibir_detalhes_trabalho(selected_row, df, similarity_matrix)
            except (KeyError, IndexError):
                st.error("O trabalho selecionado não foi encontrado. Por favor, selecione outro.")
                del st.session_state['selected_item_index']


def aba_dashboard(df):
    """Renderiza a aba de dashboard com gráficos."""
    st.markdown("## 📊 Dashboard do Acervo")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📄 Produção por Tipo de Documento")
        tipo_counts = df["Tipo_Documento"].value_counts().reset_index()
        fig1 = px.pie(
            tipo_counts,
            names='Tipo_Documento',
            values='count',
            title="Distribuição percentual"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("#### 📈 Produção Anual")
        ano_counts = df["Ano"].value_counts().sort_index()
        fig2 = px.line(
            ano_counts,
            x=ano_counts.index,
            y=ano_counts.values,
            title="Trabalhos publicados por ano",
            labels={'x': 'Ano', 'y': 'Quantidade'}
        )
        fig2.update_traces(mode='lines+markers')
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### 🏷️ Assuntos Mais Frequentes")
    todos_assuntos = Counter(a for sub in df["Assuntos_Processados"] for a in sub)
    df_top_assuntos = pd.DataFrame(todos_assuntos.most_common(20), columns=["Assunto", "Frequência"])
    
    fig3 = px.bar(
        df_top_assuntos.sort_values("Frequência"),
        x="Frequência",
        y="Assunto",
        orientation='h',
        title="Top 20 Assuntos"
    )
    st.plotly_chart(fig3, use_container_width=True)


# -----------------------------------------------------------------------------
# Execução Principal
# -----------------------------------------------------------------------------
def main():
    """Função principal que organiza e executa o app."""
    # Inicializações
    init_nltk()
    
    # Carregamento de dados e modelos
    df = load_data()
    embeddings = load_embeddings()
    similarity_matrix = get_similarity_matrix(embeddings)
    tfidf_vectorizer, tfidf_matrix = get_tfidf_model(df)
    
    # Inicializa o estado da sessão se necessário
    if 'selected_item_index' not in st.session_state:
        st.session_state['selected_item_index'] = None
    
    # Definição das abas da interface
    tab1, tab2 = st.tabs(["**Visão Geral e Pesquisa**", "**Dashboard**"])

    with tab1:
        aba_pesquisa(df, embeddings, similarity_matrix, tfidf_vectorizer, tfidf_matrix)
    
    with tab2:
        aba_dashboard(df)


if __name__ == "__main__":
    main()
