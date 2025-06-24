import streamlit as st
import pandas as pd

# Se desejar usar um grid com seleção baseada em checkbox, instale o componente:
# pip install streamlit-aggrid
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

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
    df_display = df.reset_index(drop=True)[cols_to_show]

    # Exibe o grid com checkbox de seleção
    st.markdown("Selecione um registro na tabela abaixo:")
    gb = GridOptionsBuilder.from_dataframe(df_display)
    gb.configure_selection(selection_mode="single", use_checkbox=True)
    grid_opts = gb.build()
    grid_resp = AgGrid(
        df_display,
        gridOptions=grid_opts,
        enable_enterprise_modules=False,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=True
    )
    selected = grid_resp.get("selected_rows", [])

    # Botão para exibir detalhes em um expander (popup simplificado)
    st.markdown("---")
    if st.button("Exibir detalhes do registro selecionado"):
        if selected:
            # Obtem o índice original via key '_selectedRowNodeInfo'
            sel_idx = selected[0]["_selectedRowNodeInfo"]["nodeRowIndex"]
            detalhes = df.loc[sel_idx]
            with st.expander("Detalhes do Registro", expanded=True):
                for col, val in detalhes.items():
                    st.write(f"- **{col}**: {val}")
        else:
            st.warning("Nenhum registro selecionado. Por favor, marque a checkbox da linha desejada.")

if __name__ == "__main__":
    main()
