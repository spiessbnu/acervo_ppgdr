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
    """
    return pd.read_csv(path)

# Função principal do app
def main():
    setup_page()
    st.title("Aplicativo Inicial em Streamlit")
    st.markdown("Veja abaixo os dados carregados do arquivo CSV:")

    # Carrega os dados e prepara o DataFrame
    df = load_data("dados_finais_com_resumo_llm.csv")
    df = df.rename(columns={"Tipo_Documento": "Tipo de Documento"})
    cols_to_show = [
        "Tipo de Documento",
        "Autor",
        "Título",
        "Ano",
        "Assuntos",
        "Orientador"
    ]
    df_display = df.reset_index(drop=True)[cols_to_show]

    # Exibe o DataFrame com seleção de linha (Streamlit >=1.35)
    st.markdown("Selecione um registro na tabela abaixo:")
    event = st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row"
    )

    # Botão para exibir detalhes
    st.markdown("---")
    if st.button("Exibir detalhes do registro selecionado"):
        selected_rows = event.selection.rows
        if selected_rows:
            sel_idx = selected_rows[0]
            detalhes = df.loc[sel_idx]
            st.subheader("Detalhes do Registro")
            for col, val in detalhes.items():
                st.write(f"- **{col}**: {val}")
        else:
            st.warning("Nenhum registro foi selecionado. Por favor, selecione uma linha na tabela.")

if __name__ == "__main__":
    main()
