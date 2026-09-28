import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

from sentence_transformers import CrossEncoder


# =========================================================
# 1. LOAD PDF
# =========================================================

loader = PyPDFLoader(
    "documents/jikook.pdf"
)

documents = loader.load()

print(
    f"Loaded {len(documents)} PDF pages"
)


# =========================================================
# 2. SPLIT PDF INTO CHUNKS
# =========================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=500
)

chunks = text_splitter.split_documents(
    documents
)

print(
    f"Created {len(chunks)} chunks"
)


# =========================================================
# 3. BM25 TOKENIZER
# =========================================================

def tokenize(text):

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


# =========================================================
# 4. CREATE BM25 RETRIEVER
# =========================================================

bm25_retriever = BM25Retriever.from_documents(
    chunks,
    preprocess_func=tokenize
)

bm25_retriever.k = 10


# =========================================================
# 5. EMBEDDINGS
# =========================================================

print(
    "Loading embedding model..."
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print(
    "Embedding model loaded!"
)


# =========================================================
# 6. VECTOR STORE
# =========================================================

vector_store = InMemoryVectorStore(
    embedding=embeddings
)

print(
    "Creating embeddings..."
)

vector_store.add_documents(
    chunks
)

print(
    "Embeddings created!"
)


# =========================================================
# 7. CROSS-ENCODER
# =========================================================

print(
    "Loading reranker..."
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)

print(
    "Reranker loaded!"
)


# =========================================================
# 8. RECIPROCAL RANK FUSION
# =========================================================

def reciprocal_rank_fusion(
    result_lists,
    k=60
):

    scores = {}
    documents_by_id = {}

    for results in result_lists:

        for rank, document in enumerate(
            results,
            start=1
        ):

            document_id = (
                document.metadata.get("source"),
                document.metadata.get("page_label"),
                document.page_content
            )

            documents_by_id[
                document_id
            ] = document

            if document_id not in scores:

                scores[
                    document_id
                ] = 0

            scores[
                document_id
            ] += (
                1 / (k + rank)
            )

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    return [
        documents_by_id[
            document_id
        ]
        for document_id
        in ranked_ids
    ]


# =========================================================
# 9. CROSS-ENCODER RERANKING
# =========================================================

def rerank_documents(
    question,
    candidate_documents,
    top_k=5
):

    if not candidate_documents:

        return []

    pairs = []

    for document in candidate_documents:

        pairs.append([
            question,
            document.page_content
        ])

    scores = reranker.predict(
        pairs
    )

    scored_documents = list(
        zip(
            candidate_documents,
            scores
        )
    )

    scored_documents.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return scored_documents[
        :top_k
    ]


# =========================================================
# 10. HYBRID RETRIEVAL
# =========================================================

def hybrid_retrieve(
    question,
    top_k=5
):

    # -----------------------------------------------------
    # BM25
    # -----------------------------------------------------

    bm25_results = (
        bm25_retriever.invoke(
            question
        )
    )

    # -----------------------------------------------------
    # VECTOR SEARCH
    # -----------------------------------------------------

    vector_results_with_scores = (
        vector_store
        .similarity_search_with_score(
            question,
            k=10
        )
    )

    vector_results = [
        document
        for document, score
        in vector_results_with_scores
    ]

    # -----------------------------------------------------
    # RRF
    # -----------------------------------------------------

    hybrid_results = (
        reciprocal_rank_fusion([
            bm25_results,
            vector_results
        ])
    )

    # Send the best 10 RRF results
    # into the more expensive reranker.

    candidates = (
        hybrid_results[:10]
    )

    # -----------------------------------------------------
    # CROSS-ENCODER
    # -----------------------------------------------------

    reranked = (
        rerank_documents(
            question,
            candidates,
            top_k=top_k
        )
    )

    return reranked
