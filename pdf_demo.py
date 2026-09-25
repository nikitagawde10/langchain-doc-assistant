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
    Normalizes text before BM25 indexes/searches it.

    Example:

    "Who is Hoseok?"
          ↓
    ["who", "is", "hoseok"]

    This avoids problems such as:

    "Hoseok?" != "Hoseok"
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

bm25_retriever.k = 5


# =========================================================
# 5. CREATE EMBEDDING MODEL
# =========================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


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
# 7. FORMAT DOCUMENTS FOR THE LLM
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
# 8. RECIPROCAL RANK FUSION
# =========================================================

def reciprocal_rank_fusion(
    result_lists,
    k=60
):
    """
    Combines rankings from multiple retrievers.

    Documents that appear near the top of multiple
    result lists receive a higher final score.

    RRF score:

        1 / (k + rank)

    The scores from each retriever are added together.
    """

    scores = {}

    documents_by_id = {}

    # Go through each retriever's result list
    for results in result_lists:

        # rank starts at 1:
        #
        # result #1 → rank 1
        # result #2 → rank 2
        # etc.
        for rank, document in enumerate(
            results,
            start=1
        ):

            # ---------------------------------------------
            # Create an identity for this chunk
            # ---------------------------------------------
            #
            # BM25 and vector search might return the
            # exact same chunk.
            #
            # We need to recognize that they are the
            # same result so their RRF scores are combined.
            # ---------------------------------------------

            document_id = (
                document.metadata.get("source"),
                document.metadata.get("page_label"),
                document.page_content
            )

            # Store the actual Document object
            documents_by_id[document_id] = document

            # Initialize score
            if document_id not in scores:
                scores[document_id] = 0

            # Add this retriever's contribution
            scores[document_id] += (
                1 / (k + rank)
            )


    # =====================================================
    # Sort chunk IDs by RRF score
    # =====================================================

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True
    )


    # =====================================================
    # Convert IDs back into Document objects
    # =====================================================

    ranked_documents = [
        documents_by_id[document_id]
        for document_id in ranked_ids
    ]


    return ranked_documents


# =========================================================
# 9. CREATE PROMPT
# =========================================================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a document question-answering assistant.
Answer using only the provided context.

You may make an inference only when multiple details in the
context strongly support it. Clearly label such statements
as inferences.

Do not infer occupations, relationships, locations, ages,
or other factual attributes unless the context supports them.

If the answer cannot be determined, say:
"I don't know based on the provided document."

Cite the page label for factual claims when possible.



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
# 10. CREATE LLM
# =========================================================

model = ChatGroq(
    model="openai/gpt-oss-20b"
)


# =========================================================
# 11. CREATE ANSWER CHAIN
# =========================================================

answer_chain = (
    prompt
    | model
    | StrOutputParser()
)


# =========================================================
# 12. QUESTION LOOP
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
    # 13. BM25 SEARCH
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
    # 14. VECTOR SEARCH
    # =====================================================

    vector_results_with_scores = (
        vector_store.similarity_search_with_score(
            question,
            k=5
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
    # 15. EXTRACT DOCUMENTS FROM VECTOR RESULTS
    # =====================================================
    #
    # similarity_search_with_score() returned:
    #
    # [
    #     (Document, score),
    #     (Document, score),
    #     ...
    # ]
    #
    # RRF only needs the Documents.
    # =====================================================

    vector_results = [
        document
        for document, score
        in vector_results_with_scores
    ]


    # =====================================================
    # 16. HYBRID RETRIEVAL USING RRF
    # =====================================================

    hybrid_results = reciprocal_rank_fusion(
        [
            bm25_results,
            vector_results
        ]
    )


    # =====================================================
    # 17. KEEP TOP 5 HYBRID RESULTS
    # =====================================================

    retrieved_documents = hybrid_results[:5]


    # =====================================================
    # 18. DISPLAY HYBRID RESULTS
    # =====================================================

    print("\n================================")
    print("HYBRID RESULTS")
    print("================================")


    for index, document in enumerate(
        retrieved_documents
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
    # 19. FORMAT HYBRID RESULTS INTO CONTEXT
    # =====================================================

    context = format_documents(
        retrieved_documents
    )


    # =====================================================
    # 20. SEND CONTEXT + QUESTION TO LLM
    # =====================================================

    answer = answer_chain.invoke({
        "context": context,
        "question": question
    })


    # =====================================================
    # 21. DISPLAY ANSWER
    # =====================================================

    print("\n================================")
    print("ANSWER")
    print("================================")

    print(answer)


    # =====================================================
    # 22. DISPLAY RETRIEVED PAGES
    # =====================================================

    pages = []


    for document in retrieved_documents:

        page = document.metadata.get(
            "page_label"
        )

        if page and page not in pages:
            pages.append(page)


    print(
        "\nHybrid-retrieved pages:",
        ", ".join(pages)
    )