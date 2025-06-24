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
        "Tipo de Documento",
        "Autor",
        "Título",
        "Ano",
        "Assuntos",
        "Orientador"
    ]
    df_display = df.reset_index(drop=True)[cols_to_show]

    # Exibe o DataFrame com seleção de linha
    st.markdown("Selecione um registro diretamente na tabela abaixo:")
    selected = st.experimental_data_editor(
        df_display,
        num_rows="dynamic",
        use_container_width=True,
        row_selectable="single"
    )

    # Detalhes do registro selecionado
    st.markdown("---")
    st.subheader("Detalhes do Registro Selecionado")

    if not selected.empty:
        # Obtém o índice do registro selecionado
        sel_idx = selected.index[0]
        detalhes = df.reset_index(drop=True).loc[sel_idx]

        # Exibe todas as colunas originais do registro selecionado
        st.markdown("**Informações completas do registro:**")
        for col, val in detalhes.items():
            st.write(f"- **{col}**: {val}")
    else:
        st.info("Nenhum registro selecionado.")

if __name__ == "__main__":
    main()
