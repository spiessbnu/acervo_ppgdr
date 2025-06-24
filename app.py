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
    
    df['index_original'] = df.index
    
    cols_display = ["Tipo de Documento", "Autor", "Título", "Ano", "Assuntos", "Orientador"]
    
    df_aggrid = df[cols_display + ['index_original']]

    # Configura AgGrid
    gb = GridOptionsBuilder.from_dataframe(df_aggrid)
    gb.configure_selection(selection_mode="single", use_checkbox=True)
    gb.configure_column("index_original", hide=True)
    grid_opts = gb.build()

    # Exibe grid
    grid_response = AgGrid(
        df_aggrid,
        gridOptions=grid_opts,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        enable_enterprise_modules=False,
        fit_columns_on_grid_load=True,
        key='data_grid'
    )

    st.markdown("---")
    
    selected_rows = grid_response.get("selected_rows")
    
    if not selected_rows.empty:
        # Recupera a linha de dados completa do dataframe original
        selected_data_row = selected_rows.iloc[0]
        original_index = selected_data_row.get('index_original')
        detalhes = df.loc[original_index]

        # --- INÍCIO DA NOVA SEÇÃO DE LAYOUT ---

        st.subheader(detalhes.get('Título', 'Título não disponível'))

        # Exibindo metadados principais em colunas para melhor organização
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Autor:** {detalhes.get('Autor', 'N/A')}")
            st.write(f"**Tipo:** {detalhes.get('Tipo de Documento', 'N/A')}")
        with col2:
            st.write(f"**Orientador:** {detalhes.get('Orientador', 'N/A')}")
            st.write(f"**Ano:** {detalhes.get('Ano', 'N/A')}")

        st.divider() # Adiciona uma linha divisória

        # Seção de Assuntos
        st.markdown("##### Assuntos")
        st.write(detalhes.get('Assuntos', 'Nenhum assunto listado.'))
        
        # Seção de Resumo (usando um expander para não ocupar muito espaço)
        with st.expander("**Ver Resumo**", expanded=True):
            resumo = detalhes.get('Resumo_LLM', 'Resumo não disponível.')
            st.write(resumo)

        # Seção de Download (com um botão clicável)
        st.markdown("##### Link para Download")
        link_pdf = detalhes.get('Link_PDF')
        if link_pdf and isinstance(link_pdf, str):
            st.link_button("Baixar PDF", url=link_pdf, use_container_width=True)
        else:
            st.warning("Nenhum link para download disponível.")

        # --- FIM DA NOVA SEÇÃO DE LAYOUT ---
            
    else:
        st.info("Selecione um registro na tabela acima para ver os detalhes.")


if __name__ == "__main__":
    main()
