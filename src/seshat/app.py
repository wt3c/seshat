import streamlit as st
from pathlib import Path

from config import settings
from ingestion.pdf_processor import PDFProcessor
from ingestion.chunker import SemanticChunker
from retrieval.vector_store import VectorStoreManager
from chain.rag_chain import RAGChain


# Configuração da página
st.set_page_config(
    page_title="Seshat - Consulta de Documentos",
    page_icon="📚",
    layout="wide",
)

# CSS customizado
st.markdown(
    """
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .source-chip {
        background-color: #f0f2f6;
        padding: 0.25rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_rag_chain():
    """Inicializa a RAG chain (cached)."""
    return RAGChain()


def process_pdfs_tab():
    """Tab de processamento de PDFs."""
    st.header("📄 Processamento de PDFs")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload de Arquivos")
        uploaded_files = st.file_uploader("Arraste PDFs aqui", type=["pdf"], accept_multiple_files=True)

        if uploaded_files:
            for file in uploaded_files:
                save_path = settings.pdf_input_dir / file.name
                save_path.write_bytes(file.read())
                st.success(f"✅ {file.name} salvo!")

    with col2:
        st.subheader("Arquivos Disponíveis")
        pdf_files = list(settings.pdf_input_dir.glob("*.pdf"))

        if pdf_files:
            for pdf in pdf_files:
                st.text(f"📄 {pdf.name}")
        else:
            st.info("Nenhum PDF encontrado.")

    st.divider()

    if st.button("🚀 Processar e Indexar Todos os PDFs", type="primary"):
        with st.spinner("Processando PDFs com Docling..."):
            processor = PDFProcessor()
            docs = processor.process_directory()

        if docs:
            with st.spinner("Criando chunks semânticos..."):
                chunker = SemanticChunker()
                chunks = chunker.process_all(docs)

            with st.spinner("Gerando embeddings e indexando..."):
                vector_store = VectorStoreManager()
                vector_store.add_documents(chunks)

            st.success(f"✅ {len(docs)} documentos processados em {len(chunks)} chunks!")
        else:
            st.warning("Nenhum documento para processar.")


def chat_tab():
    """Tab de chat com os documentos."""
    st.header("💬 Consulta aos Documentos")

    # Inicializa histórico de mensagens na sessão
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "session_id" not in st.session_state:
        import uuid

        st.session_state.session_id = str(uuid.uuid4())

    # Sidebar com controles
    with st.sidebar:
        st.subheader("⚙️ Configurações")

        if st.button("🗑️ Limpar Conversa"):
            st.session_state.messages = []
            chain = get_rag_chain()
            chain.clear_history(st.session_state.session_id)
            st.rerun()

        # Stats do vector store
        try:
            vs = VectorStoreManager()
            stats = vs.get_collection_stats()
            st.metric("Documentos Indexados", stats["count"])
        except Exception:
            st.warning("Vector store não inicializado")

    # Exibe mensagens anteriores
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input do usuário
    if prompt := st.chat_input("Faça uma pergunta sobre seus documentos..."):
        # Adiciona mensagem do usuário
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Gera resposta
        with st.chat_message("assistant"):
            with st.spinner("Consultando documentos..."):
                chain = get_rag_chain()
                response = chain.ask(prompt, st.session_state.session_id)

            st.markdown(response)

            # Mostra fontes consultadas
            with st.expander("📎 Fontes consultadas"):
                docs = chain.get_relevant_docs(prompt)
                for doc in docs:
                    source = doc.metadata.get("source", "Desconhecido")
                    st.caption(f"**{source}**")
                    st.text(doc.page_content[:300] + "...")
                    st.divider()

        # Salva resposta
        st.session_state.messages.append({"role": "assistant", "content": response})


def main():
    st.title("📚 Seshat - Sistema de Consulta Inteligente")
    st.caption("Powered by Claude + LangChain + ChromaDB + Docling")

    tab1, tab2 = st.tabs(["💬 Chat", "📄 Processar PDFs"])

    with tab1:
        chat_tab()

    with tab2:
        process_pdfs_tab()


if __name__ == "__main__":
    main()
