from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retrieval import (
    documents,
    hybrid_retrieve
)


# =========================================================
# 1. ENVIRONMENT
# =========================================================

load_dotenv()


# =========================================================
# 2. LLM
# =========================================================

model = ChatGroq(
    model="openai/gpt-oss-20b"
)


# =========================================================
# 3. QUERY ROUTER SCHEMA
# =========================================================

class QueryClassification(
    BaseModel
):

    query_type: Literal[
        "semantic",
        "exact",
        "aggregation"
    ] = Field(
        description=(
            "The retrieval strategy required "
            "for the question."
        )
    )


# =========================================================
# 4. ROUTER MODEL
# =========================================================

router_model = (
    model.with_structured_output(
        QueryClassification
    )
)


# =========================================================
# 5. ROUTER PROMPT
# =========================================================

router_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You classify document questions by the SEARCH STRATEGY
needed to answer them.

Choose exactly one category.


SEMANTIC:

Use when the answer can likely be found in one or a small
number of relevant passages using meaning and context.

Examples:

"Who is Hoseok?"
-> semantic

"Why is Jimin angry?"
-> semantic

"Which ice creams do Jimin and Taehyung eat?"
-> semantic


EXACT:

Use when the user explicitly wants to locate a specific
word, phrase, quotation, or exact textual occurrence.

Examples:

"Find the phrase 'mise en place'."
-> exact

"Where does 'parfait' appear?"
-> exact


AGGREGATION:

Use when answering requires an exhaustive search across
the document.

Examples:

"How many times is Hoseok mentioned?"
-> aggregation

"How many times does Jungkook say parfait?"
-> aggregation

"List every time Jungkook compliments Jimin."
-> aggregation


IMPORTANT:

Do NOT classify something as aggregation merely because
multiple people or facts are mentioned.

Ask:

"Must I inspect the entire document to guarantee that
the answer is complete?"

If yes:
-> aggregation

Otherwise, if exact matching is unnecessary:
-> semantic

Return only the structured classification.
"""
    ),

    (
        "human",
        "{question}"
    )
])


router_chain = (
    router_prompt
    | router_model
)


# =========================================================
# 6. CLASSIFY QUERY
# =========================================================

def classify_query(
    question
):

    result = router_chain.invoke({
        "question": question
    })

    if isinstance(
        result,
        QueryClassification
    ):

        return result

    return QueryClassification.model_validate(
        result
    )


# =========================================================
# 7. FORMAT DOCUMENTS
# =========================================================

def format_documents(
    retrieved_documents
):

    formatted = []

    for document in retrieved_documents:

        page = document.metadata.get(
            "page_label",
            "Unknown"
        )

        formatted.append(
            f"[Page {page}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(
        formatted
    )


# =========================================================
# 8. SEMANTIC ANSWER PROMPT
# =========================================================

answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a document question-answering assistant.

Answer the user's question using only the provided context.

You may make an inference only when the context strongly
supports it.

Clearly identify inference rather than presenting it as an
explicitly stated fact.

If the answer cannot be determined from the context, say:

"I don't know based on the provided document."

Do not invent information.

When possible, mention the page associated with the
information.

Context:
{context}
"""
    ),

    (
        "human",
        "{question}"
    )
])


answer_chain = (
    answer_prompt
    | model
    | StrOutputParser()
)


# =========================================================
# 9. SEMANTIC SEARCH
# =========================================================

def semantic_search(
    question
):

    reranked_results = (
        hybrid_retrieve(
            question,
            top_k=5
        )
    )

    retrieved_documents = [
        document
        for document, score
        in reranked_results
    ]

    print(
        "\n================================"
    )

    print(
        "SEMANTIC RETRIEVAL RESULTS"
    )

    print(
        "================================"
    )

    for index, (
        document,
        score
    ) in enumerate(
        reranked_results,
        start=1
    ):

        page = document.metadata.get(
            "page_label",
            "Unknown"
        )

        print(
            f"\nRESULT {index}"
        )

        print(
            f"Page: {page}"
        )

        print(
            f"Reranker score: "
            f"{float(score):.4f}"
        )

        print(
            document.page_content[:500]
        )

        print(
            "--------------------------------"
        )

    context = format_documents(
        retrieved_documents
    )

    answer = answer_chain.invoke({
        "context": context,
        "question": question
    })

    pages = []

    for document in retrieved_documents:

        page = document.metadata.get(
            "page_label"
        )

        if (
            page
            and page not in pages
        ):

            pages.append(
                page
            )

    return {
        "answer": answer,
        "pages": pages
    }


# =========================================================
# 10. EXACT SEARCH SCHEMA
# =========================================================

class ExactSearchQuery(
    BaseModel
):

    search_text: str = Field(
        description=(
            "The exact word or phrase to search "
            "for in the document."
        )
    )


# =========================================================
# 11. EXACT EXTRACTION MODEL
# =========================================================

exact_model = (
    model.with_structured_output(
        ExactSearchQuery
    )
)


# =========================================================
# 12. EXACT EXTRACTION PROMPT
# =========================================================

exact_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Extract the exact word or phrase the user wants to locate.

Do not answer the question.

Examples:

Find the phrase "parfait"
-> parfait

Where does mise en place appear?
-> mise en place

Show me the sentence containing "golden maknae"
-> golden maknae
"""
    ),

    (
        "human",
        "{question}"
    )
])


exact_chain = (
    exact_prompt
    | exact_model
)


# =========================================================
# 13. EXTRACT EXACT SEARCH TEXT
# =========================================================

def extract_search_text(
    question
):

    result = exact_chain.invoke({
        "question": question
    })

    if isinstance(
        result,
        ExactSearchQuery
    ):

        extraction = result

    else:

        extraction = (
            ExactSearchQuery.model_validate(
                result
            )
        )

    return extraction.search_text.strip()


# =========================================================
# 14. EXACT TEXT SEARCH
# =========================================================

def find_exact_occurrences(
    search_text
):

    matches = []

    needle = search_text.lower()

    # Important:
    #
    # Search original PDF pages, NOT overlapping chunks.

    for document in documents:

        original_text = (
            document.page_content
        )

        searchable_text = (
            original_text.lower()
        )

        start = 0

        while True:

            position = (
                searchable_text.find(
                    needle,
                    start
                )
            )

            if position == -1:

                break

            context_start = max(
                0,
                position - 150
            )

            context_end = min(
                len(original_text),
                position
                + len(search_text)
                + 150
            )

            snippet = original_text[
                context_start:
                context_end
            ]

            matches.append({
                "page": (
                    document.metadata.get(
                        "page_label",
                        "Unknown"
                    )
                ),

                "position": position,

                "snippet": snippet
            })

            start = (
                position
                + len(search_text)
            )

    return matches


# =========================================================
# 15. EXACT SEARCH ROUTE
# =========================================================

def exact_search(
    question
):

    search_text = (
        extract_search_text(
            question
        )
    )

    matches = (
        find_exact_occurrences(
            search_text
        )
    )

    return {
        "search_text": search_text,
        "matches": matches,
        "count": len(matches)
    }


# =========================================================
# 16. AGGREGATION TYPE
# =========================================================

class AggregationClassification(
    BaseModel
):

    aggregation_type: Literal[
        "exact_count",
        "semantic"
    ] = Field(
        description=(
            "Whether the aggregation can be solved "
            "using deterministic exact text counting "
            "or requires semantic understanding."
        )
    )

    search_text: str | None = Field(
        default=None,
        description=(
            "Exact text to count when aggregation_type "
            "is exact_count."
        )
    )


# =========================================================
# 17. AGGREGATION CLASSIFIER
# =========================================================

aggregation_model = (
    model.with_structured_output(
        AggregationClassification
    )
)


aggregation_prompt = (
    ChatPromptTemplate.from_messages([
        (
            "system",
            """
Classify an aggregation question.

Choose:

exact_count:
Use when the question asks how many times a specific
word, name, or exact phrase occurs in the document.

Also extract the text that should be counted.

Example:

"How many times is Hoseok mentioned?"

aggregation_type:
exact_count

search_text:
Hoseok


semantic:
Use when answering requires understanding events,
meaning, actions, relationships, descriptions, or
concepts rather than simply counting a literal string.

Example:

"List every time Jungkook compliments Jimin."

aggregation_type:
semantic

search_text:
null


IMPORTANT:

"How many times does Jungkook say parfait?"

is NOT a simple exact count.

Counting the word "parfait" would include occurrences
spoken by other characters or used in narration.

Determining whether Jungkook actually said it requires
semantic understanding.

Therefore:

"How many times does Jungkook say parfait?"
-> semantic
"""
        ),

        (
            "human",
            "{question}"
        )
    ])
)


aggregation_chain = (
    aggregation_prompt
    | aggregation_model
)


# =========================================================
# 18. CLASSIFY AGGREGATION
# =========================================================

def classify_aggregation(
    question
):

    result = aggregation_chain.invoke({
        "question": question
    })

    if isinstance(
        result,
        AggregationClassification
    ):

        return result

    return (
        AggregationClassification.model_validate(
            result
        )
    )


# =========================================================
# 19. AGGREGATION SEARCH
# =========================================================

def aggregation_search(
    question
):

    classification = (
        classify_aggregation(
            question
        )
    )

    # -----------------------------------------------------
    # EXACT COUNT
    # -----------------------------------------------------

    if (
        classification.aggregation_type
        == "exact_count"
    ):

        search_text = (
            classification.search_text
        )

        if not search_text:

            return {
                "type": "error",
                "message": (
                    "The aggregation classifier did not "
                    "provide text to count."
                )
            }

        matches = (
            find_exact_occurrences(
                search_text
            )
        )

        return {
            "type": "exact_count",
            "search_text": search_text,
            "count": len(matches),
            "matches": matches
        }

    # -----------------------------------------------------
    # SEMANTIC AGGREGATION
    # -----------------------------------------------------

    return {
        "type": "semantic",
        "message": (
            "Semantic aggregation route selected. "
            "We will implement this next."
        )
    }
