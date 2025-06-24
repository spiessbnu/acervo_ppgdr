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
    # Usamos .copy() para evitar SettingWithCopyWarning
    df_display = df.reset_index(drop=True)[cols_to_show].copy()

    # Exibe o DataFrame com seleção de linha usando a abordagem moderna
    st.markdown("Selecione um registro diretamente na tabela abaixo:")
    
    # st.dataframe agora suporta seleção e armazena o resultado em st.session_state
    # O evento 'on_select' pode ser "rerun" (padrão) ou um callback.
    # selection_mode pode ser "single-row", "multi-row", "single-column", ou "multi-column".
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row"
    )

    # Detalhes do registro selecionado
    st.markdown("---")
    st.subheader("Detalhes do Registro Selecionado")

    # A seleção fica armazenada em st.session_state. "selection" é a chave padrão.
    if "selection" in st.session_state and len(st.session_state.selection["rows"]) > 0:
        # Pega o índice da linha selecionada no df_display
        sel_idx = st.session_state.selection["rows"][0]
        
        # Usa esse índice para localizar os detalhes no DataFrame original (df)
        detalhes = df.iloc[sel_idx]

        st.markdown("**Informações completas do registro:**")
        for col, val in detalhes.items():
            # Trata valores NaN para melhor exibição
            if pd.isna(val):
                val = "Não informado"
            st.write(f"- **{col}**: {val}")
    else:
        st.info("Nenhum registro selecionado. Clique em uma linha na tabela acima.")

# Não se esqueça de manter o resto do seu código (imports, setup_page, load_data e o if __name__...)
