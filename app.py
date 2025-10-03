# app.py (Novo conteúdo)

import streamlit as st

st.set_page_config(
    page_title="Acervo PPGDR",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #0F5EDD;
            color: white; /* Adicione esta linha */
        }
        /* Para garantir que os links também fiquem brancos */
        [data-testid="stSidebar"] a {
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Bem-vindo ao Acervo de Dissertações e Teses do PPGDR! 📚")
st.markdown("---")
st.subheader("Utilize o menu de navegação à esquerda para explorar o acervo:")
st.markdown("""
- **Consultas:** Para fazer buscas detalhadas, encontrar trabalhos por similaridade e usar a análise de IA.
- **Dashboard:** Para visualizar gráficos e estatísticas sobre o acervo completo.
- **Sobre:** Para mais informações sobre o projeto.
""")

# Adiciona a imagem no rodapé da barra lateral (veja o item c)
with st.sidebar:
    st.image("NET-01.png", use_column_width=True)
