# --------------------------------------------------------------------------
# BIBLIOTECAS NECESSÁRIAS
# Certifique-se de que todas estão no seu arquivo requirements.txt ou
# instaladas via pip: pip install streamlit pandas streamlit-aggrid
# --------------------------------------------------------------------------
import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# --------------------------------------------------------------------------
# FUNÇÃO 1: Configuração da página do Streamlit
# Define o título, ícone e layout da página.
# --------------------------------------------------------------------------
def setup_page():
    st.set_page_config(
        page_title="Visualização de Dados do Acervo",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# --------------------------------------------------------------------------
# FUNÇÃO 2: Carregamento dos dados
# Usa o cache do Streamlit para carregar o CSV apenas uma vez,
# melhorando a performance.
# --------------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """
    Carrega o arquivo CSV e retorna um DataFrame do Pandas.
    """
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"Erro: O arquivo '{path}' não foi encontrado. Verifique o caminho e o nome do arquivo.")
        return pd.DataFrame() # Retorna um DataFrame vazio para evitar outros erros

# --------------------------------------------------------------------------
# FUNÇÃO 3: Corpo principal do aplicativo
# Onde toda a lógica de exibição acontece.
# --------------------------------------------------------------------------
def main():
    # 1. Configura a página
    setup_page()
    st.title("Visualizador de Acervo Acadêmico")
    st.markdown("Selecione uma linha na tabela abaixo para ver os detalhes completos do registro.")

    # 2. Carrega os dados
    df = load_data("dados_finais_com_resumo_llm.csv")
    
    # Se o dataframe estiver vazio (devido a um erro de carregamento), interrompe a execução.
    if df.empty:
        return

    # 3. Prepara o DataFrame para exibição
    df = df.rename(columns={"Tipo_Documento": "Tipo de Documento"})
    df['index_original'] = df.index  # Guarda o índice original para referência segura
    
    # Define as colunas que serão mostradas na tabela principal
    cols_display = ["Tipo de Documento", "Autor", "Título", "Ano", "Assuntos", "Orientador"]
    
    # Cria o DataFrame para o AgGrid, garantindo que a coluna de índice esteja presente
    df_aggrid = df[cols_display + ['index_original']]

    # 4. Configura a grade interativa (AgGrid)
    gb = GridOptionsBuilder.from_dataframe(df_aggrid)
    gb.configure_selection(selection_mode="single", use_checkbox=True)
    gb.configure_column("index_original", hide=True)  # Oculta a coluna de índice da visão do usuário
    grid_opts = gb.build()

    # 5. Exibe a grade na tela
    grid_response = AgGrid(
        df_aggrid,
        gridOptions=grid_opts,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        enable_enterprise_modules=False,
        fit_columns_on_grid_load=True,
        key='data_grid'  # Chave única para o componente
    )

    st.markdown("---") # Linha divisória

    # 6. Lógica para exibir os detalhes da linha selecionada
    selected_rows = grid_response.get("selected_rows")
    
    if not selected_rows.empty:
        # Recupera a linha de dados completa do DataFrame original
        selected_data_row = selected_rows.iloc[0]
        original_index = selected_data_row.get('index_original')
        detalhes = df.loc[original_index]

        # --- Início da Seção de Layout dos Detalhes ---

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

        # --- Fim da Seção de Layout dos Detalhes ---
            
    else:
        st.info("Selecione um registro na tabela acima para ver os detalhes.")


# --------------------------------------------------------------------------
# Ponto de entrada do script
# Garante que a função main() seja executada quando o script é chamado.
# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
