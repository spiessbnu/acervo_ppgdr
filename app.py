def render_page_consultas(df: pd.DataFrame, embeddings: np.ndarray, matriz_similaridade: np.ndarray, subject_options: list):
    st.title("Consulta ao Acervo de Dissertações e Teses")

    # --- Estado da Sessão ---
    if 'selected_rows_cache' not in st.session_state:
        st.session_state.selected_rows_cache = pd.DataFrame()
    if 'search_term' not in st.session_state: st.session_state.search_term = ""
    if 'semantic_term' not in st.session_state: st.session_state.semantic_term = ""
    if 'subject_filter' not in st.session_state: st.session_state.subject_filter = subject_options[0]
    if 'analysis_cache' not in st.session_state: st.session_state.analysis_cache = {}

    # --- Lógica de Filtros ---
    def clear_all_filters():
        """
        [CORREÇÃO] A função de callback que limpa o estado da sessão.
        A chamada st.rerun() foi removida daqui, pois o clique no botão
        já agenda um rerun automático, tornando a chamada explícita redundante.
        """
        st.session_state.search_term = ""
        st.session_state.semantic_term = ""
        st.session_state.subject_filter = subject_options[0]
        st.session_state.selected_rows_cache = pd.DataFrame()
        if 'semantic_query_input' in st.session_state:
            st.session_state.semantic_query_input = ""
        # st.rerun() -> REMOVIDO

    search_col1, search_col2 = st.columns(2)
    with search_col1:
        st.text_input("Busca simples por palavra-chave", key="search_term", placeholder="Filtre por autor, título, resumo...")
    with search_col2:
        with st.form(key="semantic_form"):
            semantic_input = st.text_input("Busca semântica (com IA)", placeholder="Qual o tema do seu interesse?", key="semantic_query_input")
            semantic_submitted = st.form_submit_button("Buscar com IA 🧠")
            if semantic_submitted and semantic_input:
                st.session_state.semantic_term = semantic_input
                st.session_state.search_term = ""
                st.session_state.subject_filter = subject_options[0]
                st.rerun()

    filter_col1, filter_col2 = st.columns([3, 1])
    with filter_col1:
        st.selectbox("Filtro por Assunto", options=subject_options, key="subject_filter")
    with filter_col2:
        st.button("Limpar Filtros e Seleção 🧹", on_click=clear_all_filters, use_container_width=True, type="primary")

    # --- Aplicando filtros ---
    df_filtered = df.copy()
    if st.session_state.semantic_term:
        with st.spinner("Buscando por significado..."):
            ranked_indices = search_semantic(st.session_state.semantic_term, embeddings)
        if ranked_indices:
            df_filtered = df.loc[ranked_indices]
        else:
            df_filtered = pd.DataFrame(columns=df.columns); st.warning("Nenhum resultado para a busca semântica.")
    elif st.session_state.search_term:
        term = st.session_state.search_term
        df_search = df_filtered.copy()
        df_search['Assuntos_str_search'] = df_search['Assuntos_Processados'].apply(lambda x: ', '.join(map(str, x)))
        cols_to_search = ["Autor", "Título", "Resumo_LLM", "Orientador", "Assuntos_str_search"]
        mask = df_search[[c for c in cols_to_search if c in df_search.columns]].fillna('').astype(str).apply(
            lambda col: col.str.contains(term, case=False, na=False)
        ).any(axis=1)
        df_filtered = df_filtered[mask]
    
    selected_subject = st.session_state.get('subject_filter', subject_options[0])
    if selected_subject != '-- Selecione um Assunto --':
        mask_subject = df_filtered['Assuntos_Processados'].apply(lambda lista: selected_subject in lista)
        df_filtered = df_filtered[mask_subject]

    st.divider()

    # --- Tabela Interativa e Processamento da Seleção ---
    df_para_exibir = df_filtered.copy()
    
    if 'Assuntos_Processados' in df_para_exibir.columns:
        df_para_exibir["Assuntos"] = df_para_exibir["Assuntos_Processados"].apply(
            lambda x: ', '.join(map(str, x)) if isinstance(x, list) else str(x)
        )
    
    cols_display = ["Tipo de Documento", "Autor", "Título", "Ano", "Orientador", "Assuntos"]
    cols_display_existentes = [c for c in cols_display if c in df_para_exibir.columns]
    
    df_aggrid = df_para_exibir[cols_display_existentes + ['index_original']].fillna('')
    df_aggrid[SELECAO_COL] = False
    
    if not st.session_state.selected_rows_cache.empty:
        prev_idx = st.session_state.selected_rows_cache.iloc[0]['index_original']
        if prev_idx in df_aggrid['index_original'].values:
            df_aggrid.loc[df_aggrid['index_original'] == prev_idx, SELECAO_COL] = True

    gb = GridOptionsBuilder.from_dataframe(df_aggrid)
    gb.configure_default_column(resizable=True, wrapText=True, autoHeight=True, suppressMenu=True, sortable=True)
    gb.configure_column("Título", width=500); gb.configure_column("Autor", width=250)
    gb.configure_column("Orientador", width=250); gb.configure_column("Assuntos", width=350)
    gb.configure_column("Tipo de Documento", width=150); gb.configure_column("Ano", width=90)
    gb.configure_column(SELECAO_COL, header_name="Analisar", editable=True, cellRenderer='agCheckboxCellRenderer', width=120)
    gb.configure_column("index_original", hide=True)

    simple_toggle_js = JsCode(f"""
    function(e) {{
      if (e.colDef.field === '{SELECAO_COL}') {{
        e.node.setDataValue('{SELECAO_COL}', !e.value);
      }}
    }}
    """)
    gb.configure_grid_options(onCellClicked=simple_toggle_js, stopEditingWhenCellsLoseFocus=True, suppressRowClickSelection=True)
    grid_opts = gb.build()
    
    grid_response = AgGrid(
        df_aggrid,
        gridOptions=grid_opts,
        data_return_mode=DataReturnMode.AS_INPUT,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        fit_columns_on_grid_load=False,
        enable_enterprise_modules=False,
        allow_unsafe_jscode=True,
        key="main_interactive_grid"
    )

    df_return = pd.DataFrame(grid_response.get("data", []))
    if not df_return.empty and SELECAO_COL in df_return.columns:
        escolhidos = df_return[df_return[SELECAO_COL] == True]
        
        selecao_final = pd.DataFrame()
        if not escolhidos.empty:
            selecao_final = escolhidos.tail(1).copy()

        idx_atual = st.session_state.selected_rows_cache['index_original'].iloc[0] if not st.session_state.selected_rows_cache.empty else None
        idx_novo = selecao_final['index_original'].iloc[0] if not selecao_final.empty else None

        if idx_atual != idx_novo:
            st.session_state.selected_rows_cache = selecao_final
            st.rerun()
    
    st.divider()

    selected_rows_df = st.session_state.selected_rows_cache
    if selected_rows_df.empty:
        st.info("ℹ️ Para ver detalhes e trabalhos similares, marque uma linha na coluna 'Analisar' da tabela acima.")

    tab_detalhes, tab_similares = st.tabs(["Detalhes", "Trabalhos Similares"])

    with tab_detalhes:
        if not selected_rows_df.empty:
            idx_original = selected_rows_df.iloc[0]['index_original']
            detalhes = df.loc[idx_original]
            st.subheader(detalhes.get('Título', ''))
            st.divider()
            st.markdown("#### Assuntos"); st.write(', '.join(detalhes.get('Assuntos_Processados', [])))
            st.markdown("#### Resumo"); st.write(detalhes.get('Resumo_LLM', ''))
            st.markdown("#### Link para Download")
            link_pdf = detalhes.get('Link_PDF')
            if link_pdf and isinstance(link_pdf, str) and 'http' in link_pdf:
                st.link_button("Baixar PDF", url=link_pdf, use_container_width=True)
            else:
                st.warning("Nenhum link para download disponível.")
        else:
            st.info("Nenhum item selecionado.")

    with tab_similares:
        if not matriz_similaridade.any():
            st.warning("Dados de similaridade não disponíveis.")
        elif not selected_rows_df.empty:
            id_selecionado = selected_rows_df.iloc[0]['index_original']
            num_vizinhos = st.slider("Número de vizinhos", 1, 10, 5, 1, key=f"slider_vizinhos_{id_selecionado}")
            fig, node_indices = generate_similarity_graph(df, matriz_similaridade, id_selecionado, num_vizinhos)
            st.plotly_chart(fig, use_container_width=True)
            df_similares = df.loc[list(node_indices)][["Autor", "Título", "Ano"]].reset_index(drop=True)
            st.dataframe(df_similares, use_container_width=True, hide_index=True)
            st.divider()
            if st.button("Gerar análise da rede de trabalhos com IA 🧠", key=f"btn_analise_{id_selecionado}"):
                cache_key = (id_selecionado, num_vizinhos)
                if cache_key in st.session_state.analysis_cache:
                    analysis = st.session_state.analysis_cache[cache_key]; st.toast("Reexibindo análise em cache. ⚡")
                else:
                    summaries_to_analyze = df.loc[list(node_indices)]['Resumo_LLM'].dropna()
                    if not summaries_to_analyze.empty:
                        with st.spinner('A IA está lendo e preparando a análise...'):
                            analysis = get_ai_synthesis("\n\n---\n\n".join(summaries_to_analyze))
                            st.session_state.analysis_cache[cache_key] = analysis
                    else:
                        analysis = "Não há resumos disponíveis para gerar análise."; st.warning(analysis)
                with st.container(border=True):
                    st.subheader("Análise Gerada por IA"); st.markdown(analysis)
        else:
            st.info("Nenhum item selecionado para mostrar similares.")
