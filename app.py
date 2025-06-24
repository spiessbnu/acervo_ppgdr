import streamlit as st
import pandas as pd

# Se você ainda não instalou streamlit-aggrid, adicione ao seu requirements.txt:
# streamlit-aggrid
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# Configuração da página do Streamlit
def setup_page():
    st.set_page_config(
        page_title="Visualização de Dados",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """
    Carrega o CSV e retorna o DataFrame.
    """
    return pd.read_csv(path)

def main():
    setup_page()
    st.title("Aplicativo Inicial em Streamlit")
    st.markdown("**Visualização do DataFrame com seleção de linha**")

    # Carrega dados
    df = load_data("dados_finais_com_resumo_llm.csv")
    df = df.rename(columns={"Tipo_Documento": "Tipo de Documento"})
    cols = ["Tipo de Documento", "Autor", "Título", "Ano", "Assuntos", "Orientador"]
    df_display = df.reset_index(drop=False)[cols + ['index']]

    # Configura AgGrid para seleção
    gb = GridOptionsBuilder.from_dataframe(df_display[cols])
    gb.configure_selection(selection_mode="single", use_checkbox=True)
    grid_opts = gb.build()

    # Exibe grid
    grid_response = AgGrid(
        df_display[cols],
        gridOptions=grid_opts,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        enable_enterprise_modules=False,
        fit_columns_on_grid_load=True
    )
    selected = grid_response.get("selected_rows")

    # Botão para detalhes
    st.markdown("---")
    if st.button("Exibir detalhes do registro selecionado"):
        if selected:
            # O AgGrid retorna um dict dos dados da linha selecionada
            sel_data = selected[0]
            # Recupera índice original
            sel_idx = sel_data.get('index')
            detalhes = df.loc[sel_idx]

            with st.modal(f"Detalhes do Registro #{sel_idx}"):
                st.subheader(f"Registro #{sel_idx}")
                for col, val in detalhes.items():
                    st.write(f"- **{col}**: {val}")
        else:
            st.warning("Nenhum registro selecionado. Marque a checkbox da linha desejada.")

if __name__ == "__main__":
    main()
