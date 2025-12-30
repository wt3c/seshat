from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from seshat.config import settings
from seshat.retrieval.vector_store import VectorStoreManager


class RAGChain:
    """Chain de RAG com suporte a histórico de conversa."""

    SYSTEM_PROMPT = """Você é Seshat, uma assistente especializada em análise de documentos.
Sua função é responder perguntas baseando-se EXCLUSIVAMENTE no contexto fornecido.

Diretrizes:
- Responda em português brasileiro, de forma clara e objetiva
- Cite as fontes quando relevante (nome do documento)
- Se a informação não estiver no contexto, diga claramente que não encontrou
- Use formatação Markdown para melhor legibilidade
- Para dados numéricos ou tabelas, preserve a formatação original

Contexto dos documentos:
{context}
"""

    def __init__(self):
        self.llm = ChatAnthropic(
            model=settings.claude_model,
            api_key=settings.ANTHROPODS_API_KEY,
            temperature=0.3,
            max_tokens=4096,
        )

        self.vector_store_manager = VectorStoreManager()
        self.retriever = self.vector_store_manager.as_retriever(search_kwargs={"k": 5})

        # Armazena históricos por session_id
        self._message_histories: dict[str, ChatMessageHistory] = {}

        # Constrói a chain
        self._chain = self._build_chain()
        self._chain_with_history = self._build_chain_with_history()

    def _format_docs(self, docs: list[Document]) -> str:
        """Formata documentos recuperados para o contexto."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Desconhecido")
            header = doc.metadata.get("header_1", "")
            content = doc.page_content

            formatted.append(f"[Documento {i} - Fonte: {source}]\n{f'Seção: {header}' if header else ''}\n{content}\n")
        return "\n---\n".join(formatted)

    def _build_chain(self):
        """Constrói a chain básica de RAG."""
        prompt = ChatPromptTemplate.from_messages(
            [("system", self.SYSTEM_PROMPT), MessagesPlaceholder(variable_name="chat_history"), ("human", "{question}")]
        )

        # Chain com retrieval paralelo
        chain = (
            RunnableParallel(
                context=lambda x: self._format_docs(self.retriever.invoke(x["question"])),
                question=lambda x: x["question"],
                chat_history=lambda x: x.get("chat_history", []),
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Obtém ou cria histórico para uma sessão."""
        if session_id not in self._message_histories:
            self._message_histories[session_id] = ChatMessageHistory()
        return self._message_histories[session_id]

    def _build_chain_with_history(self):
        """Adiciona gerenciamento automático de histórico."""
        return RunnableWithMessageHistory(
            self._chain,
            self._get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )

    def ask(self, question: str, session_id: str = "default") -> str:
        """Faz uma pergunta com histórico de conversa."""
        response = self._chain_with_history.invoke(
            {"question": question}, config={"configurable": {"session_id": session_id}}
        )
        return response

    def ask_simple(self, question: str) -> str:
        """Faz uma pergunta sem histórico (stateless)."""
        return self._chain.invoke({"question": question, "chat_history": []})

    def clear_history(self, session_id: str = "default") -> None:
        """Limpa o histórico de uma sessão."""
        if session_id in self._message_histories:
            self._message_histories[session_id].clear()

    def get_relevant_docs(self, query: str, k: int = 4) -> list[Document]:
        """Retorna documentos relevantes sem gerar resposta."""
        return self.vector_store_manager.similarity_search(query, k=k)
