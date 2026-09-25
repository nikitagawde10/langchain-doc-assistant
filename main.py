from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough
)


load_dotenv()


# =========================================================
# 1. LOAD DOCUMENT
# =========================================================

loader = TextLoader(
    "documents/notes.txt",
    encoding="utf-8"
)

documents = loader.load()

print(f"Loaded {len(documents)} document(s)")


# =========================================================
# 2. SPLIT DOCUMENT INTO CHUNKS
# =========================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")


# =========================================================
# 3. CREATE EMBEDDING MODEL
# =========================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# 4. CREATE VECTOR STORE
# =========================================================

vector_store = InMemoryVectorStore(
    embedding=embeddings
)

vector_store.add_documents(chunks)


# =========================================================
# 5. CREATE RETRIEVER
# =========================================================

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# =========================================================
# 6. FORMAT RETRIEVED DOCUMENTS
# =========================================================

def format_documents(documents):
    formatted = []

    for document in documents:
        formatted.append(document.page_content)

    return "\n\n".join(formatted)


# =========================================================
# 7. CREATE PROMPT
# =========================================================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a helpful document assistant.

Answer the user's question using only the provided context.

If the answer cannot be determined from the context,
say "I don't know based on the provided document."

Context:
{context}
"""
    ),
    (
        "human",
        "{question}"
    )
])


# =========================================================
# 8. CREATE LLM
# =========================================================

model = ChatGroq(
    model="openai/gpt-oss-20b"
)


# =========================================================
# 9. CREATE RAG CHAIN
# =========================================================

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


# =========================================================
# 10. QUESTION LOOP
# =========================================================

while True:

    question = input("\nAsk a question (or type 'exit'): ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    retrieved_docs = retriever.invoke(question)

    print("\n--- RETRIEVED CHUNKS ---")

    for doc in retrieved_docs:
        print(doc.page_content)
        print("-----")

    answer = rag_chain.invoke(question)

    print("\nAnswer:")
    print(answer)