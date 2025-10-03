# pages/2_Sobre.py

import streamlit as st

def render_page_sobre():
    st.set_page_config(layout="wide", page_title="Sobre o Projeto")
    st.title("ℹ️ Sobre o Projeto")
    st.markdown("""
    Esta aplicação foi desenvolvida como uma interface inteligente para explorar o acervo de dissertações e teses do Programa de Pós-Graduação em Desenvolvimento Regional (PPGDR). 
    Ela utiliza técnicas de Processamento de Linguagem Natural (PLN) e Inteligência Artificial (IA) para facilitar a descoberta de conhecimento e a análise de tendências no acervo.
    """)
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.subheader("🔎 Explore e Selecione")
            st.markdown("Na página de **Consulta**, use as buscas ou filtros para encontrar trabalhos. Clique em uma linha na tabela para carregar os detalhes e visualizar documentos similares.")

    with col2:
        with st.container(border=True):
            st.subheader("🧠 Descubra Conexões com a IA")
            st.markdown("Ao selecionar um trabalho, a aba 'Trabalhos Similares' mostra um grafo de documentos próximos e permite gerar uma síntese analítica da rede com IA.")

    with col3:
        with st.container(border=True):
            st.subheader("📊 Visualize o Panorama")
            st.markdown("Acesse o **Dashboard** para explorar gráficos interativos sobre a produção anual, os assuntos mais frequentes e uma visualização 3D dos clusters temáticos.")
            
    st.divider()

    st.markdown("#### Detalhes Técnicos")
    st.info("""
    - **Embeddings de Documentos:** Os resumos foram vetorizados usando o modelo `text-embedding-3-large` da OpenAI.
    - **Busca Semântica:** A "busca inteligente" compara o significado da sua pergunta com o conteúdo dos documentos.
    - **Visualizações:** Gráficos criados com Plotly. Grafo de rede gerado com NetworkX.
    - **Interface:** Desenvolvida com Streamlit.
    """, icon="🤖")

    st.caption("""
        **Autoria do Aplicativo:** Maiko R. Spiess  
        **Fonte dos Dados:** Biblioteca Universitária FURB  
        **Data da Base de Conhecimento:** 08/2025
    """)

if __name__ == "__main__":
    render_page_sobre()
