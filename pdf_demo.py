from dotenv import load_dotenv
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from sentence_transformers import CrossEncoder


load_dotenv()


# =========================================================
# 1. LOAD PDF
# =========================================================

loader = PyPDFLoader(
    "documents/jikook.pdf"
)

documents = loader.load()

print(f"Loaded {len(documents)} PDF pages")


# =========================================================
# 2. SPLIT PDF INTO CHUNKS
# =========================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=500
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")


# =========================================================
# 3. BM25 TOKENIZER
# =========================================================

def tokenize(text):
    """
    Normalizes text for BM25.

    Example:

    "Who is Hoseok?"
            ↓
    ["who", "is", "hoseok"]
    """

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

# Retrieve broadly.
# The reranker will later choose the best results.
bm25_retriever.k = 10


# =========================================================
# 5. CREATE EMBEDDING MODEL
# =========================================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded!")


# =========================================================
# 6. CREATE VECTOR STORE
# =========================================================

vector_store = InMemoryVectorStore(
    embedding=embeddings
)

print("Creating embeddings...")

vector_store.add_documents(chunks)

print("Embeddings created!")


# =========================================================
# 7. CREATE CROSS-ENCODER RERANKER
# =========================================================
#
# Embedding model:
#
# question ----> vector
#                   \
#                    similarity
#                   /
# document ----> vector
#
#
# Cross encoder:
#
# [question + document]
#          ↓
#       model
#          ↓
# relevance score
#
# =========================================================

print("Loading reranker...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)

print("Reranker loaded!")


# =========================================================
# 8. FORMAT DOCUMENTS FOR THE LLM
# =========================================================

def format_documents(documents):

    formatted = []

    for document in documents:

        page = document.metadata.get(
            "page_label",
            "Unknown"
        )

        formatted.append(
            f"[Page {page}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(formatted)


# =========================================================
# 9. RECIPROCAL RANK FUSION
# =========================================================

def reciprocal_rank_fusion(
    result_lists,
    k=60
):
    """
    Combine rankings from multiple retrievers.

    RRF formula:

        score += 1 / (k + rank)

    If the same chunk is highly ranked by both BM25
    and vector search, it receives contributions from
    both systems.
    """

    scores = {}

    documents_by_id = {}


    # Go through each retrieval system
    for results in result_lists:

        # Go through that retriever's ranking
        for rank, document in enumerate(
            results,
            start=1
        ):

            # Create a unique identity for this chunk.
            #
            # This lets us recognize when BM25 and
            # vector search returned the SAME chunk.

            document_id = (
                document.metadata.get("source"),
                document.metadata.get("page_label"),
                document.page_content
            )


            # Store actual Document object
            documents_by_id[document_id] = document


            # Initialize score if this is the first
            # time we've encountered this chunk.

            if document_id not in scores:
                scores[document_id] = 0


            # Add RRF contribution

            scores[document_id] += (
                1 / (k + rank)
            )


    # Sort IDs from highest RRF score to lowest

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True
    )


    # Convert IDs back into Documents

    ranked_documents = [
        documents_by_id[document_id]
        for document_id in ranked_ids
    ]


    return ranked_documents


# =========================================================
# 10. CROSS-ENCODER RERANKING
# =========================================================

def rerank_documents(
    question,
    documents,
    top_k=5
):
    """
    Rerank candidate Documents using a CrossEncoder.

    Each candidate is paired with the question:

        [
            [question, document1],
            [question, document2],
            ...
        ]

    The CrossEncoder reads BOTH pieces of text together
    and predicts how relevant the document is to the
    question.
    """

    if not documents:
        return []


    # Create question/document pairs

    pairs = []

    for document in documents:

        pairs.append([
            question,
            document.page_content
        ])


    # Ask cross-encoder for relevance scores

    scores = reranker.predict(
        pairs
    )


    # Combine each Document with its score:
    #
    # [
    #     (Document, 7.4),
    #     (Document, -2.1),
    #     ...
    # ]

    scored_documents = list(
        zip(
            documents,
            scores
        )
    )


    # Sort according to score.
    #
    # item looks like:
    #
    # (Document, score)
    #
    # item[1] therefore means:
    #
    # score

    scored_documents.sort(
        key=lambda item: item[1],
        reverse=True
    )


    # Keep only the best results

    return scored_documents[:top_k]


# =========================================================
# 11. CREATE PROMPT
# =========================================================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a document question-answering assistant.

Answer the user's question using only the provided context.

You may make an inference only when the context strongly
supports it. Clearly indicate when something is an inference
rather than an explicitly stated fact.

Do not infer occupations, relationships, locations, ages,
or other factual attributes unless the context supports them.

If the answer cannot be determined from the context, say:

"I don't know based on the provided document."

Do not invent information.

When possible, mention the page number associated with
the information.

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
# 12. CREATE LLM
# =========================================================

model = ChatGroq(
    model="openai/gpt-oss-20b"
)


# =========================================================
# 13. CREATE ANSWER CHAIN
# =========================================================

answer_chain = (
    prompt
    | model
    | StrOutputParser()
)


# =========================================================
# 14. QUESTION LOOP
# =========================================================

while True:

    question = input(
        "\nAsk a question (or type 'exit'): "
    )


    # =====================================================
    # EXIT
    # =====================================================

    if question.lower() == "exit":

        print("Goodbye!")

        break


    # =====================================================
    # 15. BM25 RETRIEVAL
    # =====================================================

    bm25_results = bm25_retriever.invoke(
        question
    )


    print("\n================================")
    print("BM25 RESULTS")
    print("================================")


    for index, document in enumerate(
        bm25_results
    ):

        page = document.metadata.get(
            "page_label",
            "Unknown"
        )

        print(f"\nRESULT {index + 1}")

        print(
            f"Page: {page}"
        )

        print(
            document.page_content[:500]
        )

        print(
            "--------------------------------"
        )


    # =====================================================
    # 16. VECTOR RETRIEVAL
    # =====================================================

    vector_results_with_scores = (
        vector_store.similarity_search_with_score(
            question,
            k=10
        )
    )


    print("\n================================")
    print("VECTOR SEARCH RESULTS")
    print("================================")


    for index, (
        document,
        score
    ) in enumerate(
        vector_results_with_scores
    ):

        page = document.metadata.get(
            "page_label",
            "Unknown"
        )

        print(f"\nRESULT {index + 1}")

        print(
            f"Page: {page}"
        )

        print(
            f"Similarity: {score:.4f}"
        )

        print(
            document.page_content[:500]
        )

        print(
            "--------------------------------"
        )


    # =====================================================
    # 17. REMOVE VECTOR SCORES FOR RRF
    # =====================================================
    #
    # We currently have:
    #
    # [
    #     (Document, similarity_score),
    #     ...
    # ]
    #
    # RRF uses ranking position, so we only need:
    #
    # [
    #     Document,
    #     Document,
    #     ...
    # ]
    #
    # =====================================================

    vector_results = [
        document
        for document, score
        in vector_results_with_scores
    ]


    # =====================================================
    # 18. HYBRID RETRIEVAL USING RRF
    # =====================================================

    hybrid_results = reciprocal_rank_fusion(
        [
            bm25_results,
            vector_results
        ]
    )


    # =====================================================
    # 19. KEEP TOP 10 RRF CANDIDATES
    # =====================================================
    #
    # IMPORTANT:
    #
    # We are NOT choosing the final context here.
    #
    # RRF gives us candidates.
    #
    # The reranker makes the final relevance decision.
    #
    # =====================================================

    rrf_candidates = hybrid_results[:10]


    print("\n================================")
    print("RRF CANDIDATES")
    print("================================")


    for index, document in enumerate(
        rrf_candidates
    ):

        page = document.metadata.get(
            "page_label",
            "Unknown"
        )

        print(f"\nRESULT {index + 1}")

        print(
            f"Page: {page}"
        )

        print(
            document.page_content[:500]
        )

        print(
            "--------------------------------"
        )


    # =====================================================
    # 20. CROSS-ENCODER RERANKING
    # =====================================================

    reranked_results = rerank_documents(
        question=question,
        documents=rrf_candidates,
        top_k=5
    )


    # =====================================================
    # 21. DISPLAY RERANKED RESULTS
    # =====================================================

    print("\n================================")
    print("RERANKED RESULTS")
    print("================================")


    for index, (
        document,
        score
    ) in enumerate(
        reranked_results
    ):

        page = document.metadata.get(
            "page_label",
            "Unknown"
        )

        print(f"\nRESULT {index + 1}")

        print(
            f"Page: {page}"
        )

        print(
            f"Reranker score: {float(score):.4f}"
        )

        print(
            document.page_content[:500]
        )

        print(
            "--------------------------------"
        )


    # =====================================================
    # 22. EXTRACT FINAL DOCUMENTS
    # =====================================================
    #
    # reranked_results looks like:
    #
    # [
    #     (Document, score),
    #     (Document, score),
    #     ...
    # ]
    #
    # Groq only needs the Documents.
    #
    # =====================================================

    retrieved_documents = [
        document
        for document, score
        in reranked_results
    ]


    # =====================================================
    # 23. FORMAT FINAL CONTEXT
    # =====================================================

    context = format_documents(
        retrieved_documents
    )


    # =====================================================
    # 24. SEND QUESTION + CONTEXT TO GROQ
    # =====================================================

    answer = answer_chain.invoke({
        "context": context,
        "question": question
    })


    # =====================================================
    # 25. DISPLAY ANSWER
    # =====================================================

    print("\n================================")
    print("ANSWER")
    print("================================")

    print(answer)


    # =====================================================
    # 26. DISPLAY FINAL SOURCE PAGES
    # =====================================================

    pages = []


    for document in retrieved_documents:

        page = document.metadata.get(
            "page_label"
        )

        if page and page not in pages:
            pages.append(page)


    print(
        "\nFinal reranked pages:",
        ", ".join(pages)
    )