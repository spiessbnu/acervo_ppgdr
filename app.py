# --------------------------------------------------------------------------
# BIBLIOTECAS NECESSÁRIAS
# --------------------------------------------------------------------------
import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# --------------------------------------------------------------------------
# FUNÇÃO 1: Configuração da página do Streamlit
# --------------------------------------------------------------------------
def setup_page():
    st.set_page_config(
        page_title="Visualização de Dados do Acervo v1",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# --------------------------------------------------------------------------
# FUNÇÃO 2: Carregamento dos dados com cache
# --------------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """
    Carrega o arquivo CSV e retorna um DataFrame do Pandas.
    """
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"Erro: O arquivo '{path}' não foi encontrado. Verifique o caminho.")
        return pd.DataFrame()

# --------------------------------------------------------------------------
# FUNÇÃO 3: Corpo principal do aplicativo
# --------------------------------------------------------------------------
def main():
    # 1. Configura a página
    setup_page()
    st.title("Visualizador de Acervo Acadêmico")
    st.markdown("Selecione uma linha na tabela abaixo para ver os detalhes.")

    # 2. Carrega os dados
    df = load_data("dados_finais_com_resumo_llm.csv")
    
    if df.empty:
        return

    # 3. Prepara o DataFrame para exibição
    df = df.rename(columns={"Tipo_Documento": "Tipo de Documento"})
    df['index_original'] = df.index
    
    cols_display = ["Tipo de Documento", "Autor", "Título", "Ano", "Assuntos", "Orientador"]
    df_aggrid = df[cols_display + ['index_original']]

    # 4. Configura a grade interativa (AgGrid)
    gb = GridOptionsBuilder.from_dataframe(df_aggrid)
    gb.configure_selection(selection_mode="single", use_checkbox=True)
    gb.configure_column("index_original", hide=True)
    grid_opts = gb.build()

    # 5. Exibe a grade na tela
    grid_response = AgGrid(
        df_aggrid,
        gridOptions=grid_opts,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        enable_enterprise_modules=False,
        fit_columns_on_grid_load=True,
        key='data_grid'
    )

    st.divider() # Linha divisória

    # 6. Lógica para exibir os detalhes da linha selecionada
    selected_rows = grid_response.get("selected_rows")
    
    if not selected_rows.empty:
        selected_data_row = selected_rows.iloc[0]
        original_index = selected_data_row.get('index_original')
        detalhes = df.loc[original_index]

        # --- INÍCIO DO LAYOUT SIMPLIFICADO ---
        # Exibe apenas os 3 campos solicitados.

        st.subheader("Detalhes do Registro")
        st.markdown("---")

        # 1. Assuntos
        st.markdown("#### Assuntos")
        st.write(detalhes.get('Assuntos', 'Nenhum assunto listado.'))
        st.markdown("<br>", unsafe_allow_html=True) # Adiciona espaço extra

        # 2. Resumo
        st.markdown("#### Resumo")
        resumo = detalhes.get('Resumo_LLM', 'Resumo não disponível.')
        st.write(resumo)
        st.markdown("<br>", unsafe_allow_html=True) # Adiciona espaço extra

        # 3. Link para Download
        st.markdown("#### Link para Download")
        link_pdf = detalhes.get('Link_PDF')
        if link_pdf and isinstance(link_pdf, str):
            st.link_button("Baixar PDF", url=link_pdf, use_container_width=True)
        else:
            st.warning("Nenhum link para download disponível.")

        # --- FIM DO LAYOUT SIMPLIFICADO ---
            
    else:
        st.info("Selecione um registro na tabela acima para ver os detalhes.")


# --------------------------------------------------------------------------
# Ponto de entrada do script
# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
