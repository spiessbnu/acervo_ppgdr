# pages/1_Dashboard.py

# --------------------------------------------------------------------------
# BIBLIOTECAS NECESSÁRIAS
# --------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import ast

# --------------------------------------------------------------------------
# CONFIGURAÇÃO DE ARQUIVOS E CONSTANTES
# --------------------------------------------------------------------------
CSV_DATA_PATH = "dados_finais_com_resumo_llm.csv"
EMBEDDINGS_PATH = "openai_embeddings_concatenado_large.npy"

# --------------------------------------------------------------------------
# FUNÇÕES DE CARREGAMENTO E PROCESSAMENTO (Duplicadas para independência da página)
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

@st.cache_data
def compute_clusters(_embeddings, k):
    """Calcula PCA e K-Means para visualização de clusters."""
    pca = PCA(n_components=3, random_state=42)
    embeddings_3d = pca.fit_transform(_embeddings)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(_embeddings)
    df_plot = pd.DataFrame(embeddings_3d, columns=['pca1', 'pca2', 'pca3'])
    df_plot['cluster'] = cluster_labels
    return df_plot

# --------------------------------------------------------------------------
# FUNÇÃO PARA RENDERIZAR A PÁGINA DO DASHBOARD
# --------------------------------------------------------------------------
def render_page_dashboard(df: pd.DataFrame, embeddings: np.ndarray):
    st.title("📊 Dashboard de Análise do Acervo")
    st.markdown("---")

    # Gráfico 1: Top 20 Assuntos
    st.subheader("Top 20 Assuntos Mais Frequentes")
    todos_assuntos = [assunto for sublista in df['Assuntos_Processados'] for assunto in sublista]
    if todos_assuntos:
        contador_assuntos = Counter(todos_assuntos)
        df_top20 = pd.DataFrame(contador_assuntos.most_common(20), columns=['Assunto', 'Quantidade'])
        fig_assuntos = px.bar(
            df_top20.sort_values(by='Quantidade', ascending=True),
            x='Quantidade', y='Assunto', orientation='h', text='Quantidade'
        )
        fig_assuntos.update_traces(marker_color='#1f77b4', textposition='outside')
        fig_assuntos.update_layout(
            yaxis_title=None, xaxis_title="Ocorrências", margin=dict(l=250)
        )
        st.plotly_chart(fig_assuntos, use_container_width=True)
    st.markdown("---")

    # Gráfico 2: Produção Anual
    st.subheader("Produção Anual por Tipo de Documento")
    df_renamed = df.rename(columns={"Tipo_Documento": "Tipo de Documento"})
    contagem_agrupada = df_renamed.groupby(['Ano', 'Tipo de Documento']).size().reset_index(name='Quantidade').sort_values('Ano')
    if not contagem_agrupada.empty:
        fig_producao = px.bar(contagem_agrupada, x='Ano', y='Quantidade', color='Tipo de Documento', barmode='group')
        fig_producao.update_layout(xaxis_title="Ano", yaxis_title="Quantidade", legend_title_text='Tipo')
        fig_producao.update_xaxes(type='category')
        st.plotly_chart(fig_producao, use_container_width=True)
    st.markdown("---")

    # Gráfico 3: Clusters 3D
    st.subheader("Visualização de Clusters de Documentos (PCA + K-Means)")
    with st.expander("ℹ️ Como interpretar este gráfico?"):
        st.markdown("Este gráfico organiza todos os documentos do acervo em um espaço 3D, agrupando-os por similaridade de conteúdo. Cada ponto é um documento. Documentos com temas semelhantes tendem a aparecer mais próximos uns dos outros. Os grupos (clusters) são coloridos para destacar as principais concentrações temáticas do acervo.")
    
    k_escolhido = st.slider("Selecione o número de clusters (grupos):", min_value=2, max_value=8, value=4, step=1)
    with st.spinner(f"Calculando {k_escolhido} clusters..."):
        df_plot_3d = compute_clusters(embeddings, k_escolhido)
        df_plot_3d['Título'] = df['Título']
        df_plot_3d['Autor'] = df['Autor']
        df_plot_3d['cluster'] = df_plot_3d['cluster'].astype(str)
        
        fig_3d = px.scatter_3d(
            df_plot_3d, x='pca1', y='pca2', z='pca3', color='cluster', hover_name='Título',
            hover_data={'Autor': True, 'cluster': True, 'pca1': False, 'pca2': False, 'pca3': False},
            title=f'Clusters de Documentos (k={k_escolhido})',
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig_3d.update_traces(marker=dict(size=4, opacity=0.8))
        fig_3d.update_layout(height=700, legend_title_text='Clusters',
                             scene=dict(xaxis_title='Comp. 1', yaxis_title='Comp. 2', zaxis_title='Comp. 3'))
        st.plotly_chart(fig_3d, use_container_width=True)

# --------------------------------------------------------------------------
# PONTO DE ENTRADA DA PÁGINA
# --------------------------------------------------------------------------
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Dashboard")
    
    df = load_data(CSV_DATA_PATH)
    embeddings = load_embeddings(EMBEDDINGS_PATH)
    
    if df is not None and embeddings is not None:
        render_page_dashboard(df, embeddings)
    else:
        st.error("Falha ao carregar os dados necessários para o dashboard.")
