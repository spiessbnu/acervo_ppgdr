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

    # Carrega e prepara o DataFrame
    df = load_data("dados_finais_com_resumo_llm.csv")
    df = df.rename(columns={"Tipo_Documento": "Tipo de Documento"})
    cols_to_show = ["Tipo de Documento", "Autor", "Título", "Ano", "Assuntos", "Orientador"]
    df_display = df.reset_index(drop=False)[cols_to_show + ['index']]

    # Exibe o DataFrame sem índice
    st.dataframe(df_display.drop(columns=['index']), hide_index=True)

    # Selecione o registro via selectbox para modal
    registro_map = {row['index']: f"{row['Autor']} - {row['Título']}" for _, row in df_display.iterrows()}
    selected_idx = st.selectbox(
        "Selecione o registro para detalhes:",
        options=list(registro_map.keys()),
        format_func=lambda x: registro_map[x]
    )

    # Botão para abrir modal com detalhes
    if st.button("Exibir detalhes do registro selecionado"):
        detalhes = df.loc[selected_idx]
        # Abre modal popup com informações completas
        with st.modal(f"Detalhes do Registro #{selected_idx}"):
            st.subheader(f"Detalhes do Registro #{selected_idx}")
            for col, val in detalhes.items():
                st.write(f"- **{col}**: {val}")

if __name__ == "__main__":
    main()
