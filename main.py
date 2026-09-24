from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough
)


load_dotenv()


# ---------------------------------------------------------
# Documents
# ---------------------------------------------------------

documents = [
    Document(
        page_content="Jungkook sings Seven and Standing Next to You.",
        metadata={
            "source": "jungkook.txt",
            "artist": "Jungkook"
        }
    ),

    Document(
        page_content="Jimin sings Like Crazy and Promise.",
        metadata={
            "source": "jimin.txt",
            "artist": "Jimin"
        }
    ),

    Document(
        page_content="Taehyung sings Love Me Again and Winter Bear.",
        metadata={
            "source": "taetae.txt",
            "artist": "Taehyung"
        }
    )
]


# ---------------------------------------------------------
# Embeddings
# ---------------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ---------------------------------------------------------
# Vector Store
# ---------------------------------------------------------

vector_store = InMemoryVectorStore(
    embedding=embeddings
)

vector_store.add_documents(documents)


# ---------------------------------------------------------
# Retriever
# ---------------------------------------------------------

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)


# ---------------------------------------------------------
# LLM
# ---------------------------------------------------------

model = ChatGroq(
    model="openai/gpt-oss-20b"
)


# ---------------------------------------------------------
# Format retrieved documents
# ---------------------------------------------------------

def format_documents(documents):
    formatted = []

    for document in documents:
        formatted.append(document.page_content)

    return "\n\n".join(formatted)


# ---------------------------------------------------------
# Prompt
# ---------------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a helpful assistant.

Answer the question using only the provided context.

If the answer cannot be determined from the context,
say "I don't know based on the provided documents."

Context:
{context}
"""
    ),
    (
        "human",
        "{question}"
    )
])


# ---------------------------------------------------------
# Complete RAG Chain
# ---------------------------------------------------------

rag_chain = (
    {
        "context":
            retriever
            | RunnableLambda(format_documents),

        "question":
            RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)


# ---------------------------------------------------------
# Ask question
# ---------------------------------------------------------

question = input("Ask a question: ")

answer = rag_chain.invoke(question)

print("\nAnswer:")
print(answer)