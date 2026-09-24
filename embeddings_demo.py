from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore


# 1. Embedding model

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 2. Documents

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


# 3. Create vector store

vector_store = InMemoryVectorStore(
    embedding=embeddings
)


# 4. Add documents

vector_store.add_documents(documents=documents)


# 5. Search

query = "Who has a song about winter??"

# results = vector_store.similarity_search(
#     query,
#     k=2
# )
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)
results = retriever.invoke(
    "Who has a song about winter?"
)
def format_documents(documents):
    formatted = []
    for document in documents:
        formatted.append(
            document.page_content,
        )
    return "\n\n".join(formatted)

results = retriever.invoke(
    "Who has a song about winter?"
)

context = format_documents(results)

print(type(context))
print(context)