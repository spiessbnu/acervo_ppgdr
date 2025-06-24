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
# Configuração da página
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Acervo PPGDR v2",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Estilo (CSS)
# -----------------------------------------------------------------------------
STYLE_PATH = Path(__file__).with_name("style.css")
if STYLE_PATH.exists():
    st.markdown(STYLE_PATH.read_text(), unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Utilitários
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("dados_finais_com_resumo_llm.csv")
    if "Assuntos_Processados" not in df.columns:
        df["Assuntos_Processados"] = df["Assuntos_Lista"].apply(safe_literal_eval)
    return df

@st.cache_data(show_spinner=False)
def load_embeddings():
    return np.load("openai_embeddings_concatenado_large.npy")

@st.cache_resource(show_spinner=False)
def init_openai_client():
    key = st.secrets.get("openai_api_key", "").strip()
    return OpenAI(api_key=key) if key else None

@st.cache_resource(show_spinner=False)
def init_nltk():
    for pkg in ["punkt", "stopwords", "rslp"]:
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg)

@st.cache_data(show_spinner=False)
def matriz_similaridade_cached(emb):
    return cosine_similarity(emb)

# -----------------------------------------------------------------------------
# Pré‑processamento
# -----------------------------------------------------------------------------

def safe_literal_eval(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError, TypeError):
        return []

def normalizar_string(texto):
    if not isinstance(texto, str):
        return ""
    t = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in t if not unicodedata.combining(c))

def preprocessar_texto(texto, usar_stemmer=False):
    if not isinstance(texto, str):
        return ""
    texto = re.sub(r"[^a-zA-Z\s]", "", normalizar_string(texto))
    tokens = word_tokenize(texto)
    stop = set(stopwords.words("portuguese"))
    tokens = [t for t in tokens if t not in stop and len(t) > 1]
    if usar_stemmer:
        stmr = RSLPStemmer()
        tokens = [stmr.stem(t) for t in tokens]
    return " ".join(tokens)

# -----------------------------------------------------------------------------
# Similaridade / Busca
# -----------------------------------------------------------------------------

def preparar_busca_semantica(df):
    textos = []
    for _, r in df.iterrows():
        txt = f"{r['Título']} {r.get('Resumo_LLM', '')}"
        textos.append(preprocessar_texto(txt))
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    mat = vec.fit_transform(textos)
    return vec, mat

def buscar_e_rankear(query, vec, mat):
    q = preprocessar_texto(query)
    if not q:
        return []
    q_vec = vec.transform([q])
    sims = cosine_similarity(q_vec, mat).flatten()
    idxs = np.argsort(-sims)
    return [(i, sims[i]) for i in idxs if sims[i] > 0.01]

# -----------------------------------------------------------------------------
# Interface – componentes reutilizáveis
# -----------------------------------------------------------------------------

def mostrar_resultados(df_result, df_full, embeddings):
    if df_result.empty:
        st.info("Nenhum trabalho corresponde ao filtro/consulta.")
        return

    col_list, col_detail = st.columns([1, 2], gap="large")

    with col_list:
        idx_sel = st.radio(
            "Resultados:",
            options=df_result.index,
            format_func=lambda i: f"{df_result.loc[i, 'Título'][:80]}… ({df_result.loc[i,'Ano']})",
            label_visibility="collapsed",
        )

    with col_detail:
        exibir_detalhes_trabalho(df_result.loc[idx_sel], df_full, embeddings)

def mostrar_similares(row, df_full, embeddings):
    try:
        idx = row.name
        sim = matriz_similaridade_cached(embeddings)[idx]
        top = np.argsort(-sim)[1:4]
        for j in top:
            if sim[j] > 0.30:
                r = df_full.iloc[j]
                st.write(f"- **{r['Título']}** ({r['Ano']}) – similaridade {sim[j]:.3f}")
    except Exception as e:
        st.error(f"Erro na similaridade: {e}")


def exibir_detalhes_trabalho(row, df_full, embeddings):
    st.markdown(f"### {row['Título']}")
    st.markdown(f"**Autor:** {row['Autor']} | **Ano:** {row['Ano']} | **Tipo:** {row['Tipo_Documento']}")

    if st.toggle("📄 Mostrar resumo", key=f"res_{row.name}"):
        st.write(row.get("Resumo_LLM", "_Resumo indisponível_"))

    if embeddings is not None and st.toggle("🔗 Mostrar trabalhos similares", key=f"sim_{row.name}"):
        mostrar_similares(row, df_full, embeddings)

    if st.toggle("🗂️ Metadados completos", key=f"meta_{row.name}"):
        st.json(row.to_dict(), expanded=False)

# -----------------------------------------------------------------------------
# Páginas
# -----------------------------------------------------------------------------

def pagina_principal(df, embeddings):
    st.markdown("## 🔍 Pesquisa no Acervo")

    query = st.text_input("Digite sua busca:")
    modo = st.radio(
        "Modo de busca:",
        ["Assunto", "Palavras-chave", "Semântica TF‑IDF"],
        horizontal=True,
    )

    if modo == "Assunto":
        assuntos = sorted({a for sub in df["Assuntos_Processados"] for a in sub})
        assunto = st.selectbox("Selecione o assunto:", ["--"] + assuntos)
        if assunto != "--":
            df_r = df[df["Assuntos_Processados"].apply(lambda lst: assunto in lst)]
            mostrar_resultados(df_r, df, embeddings)
    elif modo == "Palavras-chave" and query:
        q_norm = normalizar_string(query)
        mask = df["Título"].str.lower().str.contains(q_norm, na=False) |
               df.get("Resumo_LLM", "").str.lower().str.contains(q_norm, na=False)
        mostrar_resultados(df[mask], df, embeddings)
    elif modo == "Semântica TF‑IDF" and query:
        vec, mat = preparar_busca_semantica(df)
        res = buscar_e_rankear(query, vec, mat)[:50]
        mostrar_resultados(df.iloc[[i for i, _ in res]], df, embeddings)


def dashboard(df):
    st.markdown("## 📊 Dashboard")

    # Tipo documento
    fig1 = px.bar(
        df["Tipo_Documento"].value_counts().reset_index(),
        x="index", y="Tipo_Documento", text="Tipo_Documento",
        labels={"index": "Tipo", "Tipo_Documento": "Qtd"},
    )
    fig1.update_traces(textposition="outside")
    st.plotly_chart(fig1, use_container_width=True)

    # Top assuntos
    assuntos = Counter(a for sub in df["Assuntos_Processados"] for a in sub)
    top = pd.DataFrame(assuntos.most_common(20), columns=["Assunto", "Qtd"])
    fig2 = px.bar(top.sort_values("Qtd"), x="Qtd", y="Assunto", orientation="h")
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    init_nltk()

    df = load_data()
    embeddings = None
    try:
        embeddings = load_embeddings()
    except Exception:
        pass

    pagina = st.sidebar.selectbox("Página", ["Pesquisa", "Dashboard"])

    if pagina == "Pesquisa":
        pagina_principal(df, embeddings)
    else:
        dashboard(df)


if __name__ == "__main__":
    main()
