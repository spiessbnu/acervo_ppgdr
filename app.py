import streamlit as st
import pandas as pd

# Configuração da página do Streamlit
def setup_page():
    st.set_page_config(
        page_title="Visualização de Dados",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# Função para carregar e armazenar em cache o DataFrame
@st.cache_data  # use @st.cache se sua versão do Streamlit for anterior à 1.18
def load_data(path: str) -> pd.DataFrame:
    """
    Lê o arquivo CSV e retorna um DataFrame pandas.

    Parâmetros:
    - path: caminho para o arquivo CSV

    Retorna:
    - pd.DataFrame com os dados carregados
    """
    df = pd.read_csv(path)
    return df

# Função principal do app
def main():
    setup_page()
    st.title("Aplicativo Inicial em Streamlit")
    st.markdown("Veja abaixo os dados carregados do arquivo CSV:")

    # Carrega os dados
    arquivo = "dados_finais_com_resumo_llm.csv"
    df = load_data(arquivo)

    # Renomeia a coluna
    df = df.rename(columns={"Tipo_Documento": "Tipo de Documento"})

    # Define e reordena as colunas a exibir
    cols_to_show = [
        "Tipo de Documento",  # coluna principal
        "Autor",             # adicionar Autor
        "Título",            # adicionar Título
        "Ano",
        "Assuntos",
        "Orientador"
    ]
    df_display = df.reset_index(drop=True)[cols_to_show]

    # Exibe o DataFrame sem índice e com colunas filtradas
    st.dataframe(df_display)

if __name__ == "__main__":
    main()
