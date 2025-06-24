import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

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
    
    # Adiciona o índice original do DataFrame como uma nova coluna para referência segura
    df['index_original'] = df.index
    
    cols_display = ["Tipo de Documento", "Autor", "Título", "Ano", "Assuntos", "Orientador"]
    
    # O DataFrame enviado para o AgGrid deve conter a coluna de índice
    df_aggrid = df[cols_display + ['index_original']]

    # Configura AgGrid para seleção
    gb = GridOptionsBuilder.from_dataframe(df_aggrid)
    gb.configure_selection(selection_mode="single", use_checkbox=True)
    
    # Opcional: Oculta a coluna de índice na visualização da tabela
    gb.configure_column("index_original", hide=True)
    
    grid_opts = gb.build()

    # Exibe grid
    grid_response = AgGrid(
        df_aggrid,
        gridOptions=grid_opts,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        enable_enterprise_modules=False,
        fit_columns_on_grid_load=True,
        key='data_grid' # Adicionar uma chave é uma boa prática
    )

    st.markdown("---")
    
    # Lógica para exibir os detalhes sem precisar de um botão
    selected_rows = grid_response.get("selected_rows")
    
    if selected_rows:
        st.subheader("Detalhes do Registro Selecionado")
        
        # O AgGrid retorna uma lista de dicionários
        selected_data = selected_rows[0]
        
        # Recupera o índice original que foi adicionado ao DataFrame
        original_index = selected_data.get('index_original')

        # Verifica se o índice foi encontrado antes de tentar acessar os dados
        if original_index is not None:
            # Usa o índice para obter todos os detalhes do DataFrame original
            detalhes = df.loc[original_index]
            
            # Exibe os detalhes em um formato limpo
            for col, val in detalhes.items():
                # Não exibe a coluna de índice que criamos
                if col != 'index_original':
                    st.write(f"**{col}:** {val}")
        else:
            st.error("Não foi possível encontrar o índice do registro selecionado.")
            
    else:
        st.info("Selecione um registro na tabela acima para ver os detalhes.")


if __name__ == "__main__":
    main()
