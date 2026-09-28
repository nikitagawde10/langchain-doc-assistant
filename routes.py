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

class QueryClassification(BaseModel):

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

router_model = model.with_structured_output(
    QueryClassification
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

def classify_query(question):

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

def semantic_search(question):

    reranked_results = hybrid_retrieve(
        question,
        top_k=5
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

class ExactSearchQuery(BaseModel):

    search_text: str = Field(
        description=(
            "The exact word or phrase to search "
            "for in the document."
        )
    )


# =========================================================
# 11. EXACT EXTRACTION MODEL
# =========================================================

exact_model = model.with_structured_output(
    ExactSearchQuery
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

def extract_search_text(question):

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

    # Search original PDF pages rather than chunks.
    #
    # This prevents overlapping chunks from creating
    # duplicate occurrences.

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
                position - 400
            )

            context_end = min(
                len(original_text),
                position
                + len(search_text)
                + 400
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

def exact_search(question):

    search_text = extract_search_text(
        question
    )

    matches = find_exact_occurrences(
        search_text
    )

    return {
        "search_text": search_text,
        "matches": matches,
        "count": len(matches)
    }


# =========================================================
# 16. AGGREGATION CLASSIFICATION
# =========================================================

class AggregationClassification(BaseModel):

    aggregation_type: Literal[
        "exact_count",
        "semantic"
    ] = Field(
        description=(
            "Whether the aggregation can be solved "
            "with exact text counting or requires "
            "semantic verification."
        )
    )

    search_text: str | None = Field(
        default=None,
        description=(
            "Exact text to count when aggregation_type "
            "is exact_count."
        )
    )


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

Choose exactly one:


EXACT_COUNT:

Use when answering only requires counting occurrences
of a literal word, name, or phrase.

Example:

"How many times is Hoseok mentioned?"

-> exact_count
-> search_text = "Hoseok"


SEMANTIC:

Use when occurrences must be examined to determine whether
they actually satisfy some condition.

Examples:

"How many times does Jungkook say parfait?"

Counting "parfait" alone is insufficient because another
character could say it or it could appear in narration.

-> semantic


"List every time Jungkook compliments Jimin."

There may be no literal word "compliment".

-> semantic


IMPORTANT:

If determining WHO performed an action requires reading
context, classify it as semantic.
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
# 17. CLASSIFY AGGREGATION
# =========================================================

def classify_aggregation(question):

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
# 18. SEMANTIC AGGREGATION PLAN
# =========================================================
#
# For our first version, we support semantic aggregation
# questions that contain a literal candidate term.
#
# Example:
#
# "How many times does Jungkook say parfait?"
#
# Candidate term:
# "parfait"
#
# We can find EVERY "parfait" deterministically and then
# ask the LLM whether each occurrence satisfies the
# semantic condition.
#
# =========================================================

class SemanticAggregationPlan(BaseModel):

    candidate_search_text: str | None = Field(
        default=None,
        description=(
            "A literal word or phrase whose occurrences "
            "can be searched exhaustively to generate "
            "candidates. Null if no suitable literal "
            "candidate term exists."
        )
    )


semantic_plan_model = (
    model.with_structured_output(
        SemanticAggregationPlan
    )
)


semantic_plan_prompt = (
    ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are planning candidate generation for a semantic
aggregation question.

Determine whether the user's question contains a literal
word or phrase that can be searched across the entire
document to generate ALL possible candidates.

The candidate search text is NOT necessarily the complete
answer condition.

Example:

Question:
"How many times does Jungkook say parfait?"

candidate_search_text:
parfait

Why:
Every valid occurrence must contain the literal word
"parfait". We can therefore search every occurrence of
"parfait" first and later verify whether Jungkook said it.


Question:
"How many times does Jungkook say 'I love you'?"

candidate_search_text:
I love you


Question:
"List every time Jungkook compliments Jimin."

candidate_search_text:
null

Why:
A compliment can be expressed in many different ways.
There is no single literal phrase guaranteed to appear in
every valid event.


Question:
"List every time Jimin says sorry."

candidate_search_text:
sorry


IMPORTANT:

Only return candidate_search_text when EVERY valid answer
would necessarily contain that literal text.

Otherwise return null.
"""
        ),
        (
            "human",
            "{question}"
        )
    ])
)


semantic_plan_chain = (
    semantic_plan_prompt
    | semantic_plan_model
)


# =========================================================
# 19. CREATE SEMANTIC AGGREGATION PLAN
# =========================================================

def create_semantic_plan(
    question
):

    result = (
        semantic_plan_chain.invoke({
            "question": question
        })
    )

    if isinstance(
        result,
        SemanticAggregationPlan
    ):
        return result

    return (
        SemanticAggregationPlan.model_validate(
            result
        )
    )


# =========================================================
# 20. VERIFICATION SCHEMA
# =========================================================

class VerificationResult(BaseModel):

    matches: bool = Field(
        description=(
            "True only if the candidate passage "
            "actually satisfies the user's condition."
        )
    )

    evidence: str | None = Field(
        default=None,
        description=(
            "A short explanation of the evidence "
            "supporting the decision."
        )
    )


# =========================================================
# 21. VERIFICATION MODEL
# =========================================================

verification_model = (
    model.with_structured_output(
        VerificationResult
    )
)


# =========================================================
# 22. VERIFICATION PROMPT
# =========================================================

verification_prompt = (
    ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are verifying a candidate passage from a document.

Determine whether the passage actually satisfies the
user's question.

Be strict.

Return matches=true ONLY when the provided passage gives
enough evidence to conclude that this occurrence satisfies
the requested condition.

Do not assume speaker identity merely because a character
is mentioned nearby.

For dialogue questions, determine who actually speaks the
relevant words from dialogue attribution and surrounding
context.

If the passage is ambiguous, return matches=false.

The candidate was found because it contains a literal
search term. The presence of that term alone does NOT
prove that the candidate is valid.
"""
        ),
        (
            "human",
            """
Question:

{question}


Candidate passage:

{passage}
"""
        )
    ])
)


verification_chain = (
    verification_prompt
    | verification_model
)


# =========================================================
# 23. VERIFY ONE CANDIDATE
# =========================================================

def verify_candidate(
    question,
    candidate
):

    result = (
        verification_chain.invoke({
            "question": question,
            "passage": candidate["snippet"]
        })
    )

    if isinstance(
        result,
        VerificationResult
    ):
        return result

    return (
        VerificationResult.model_validate(
            result
        )
    )


# =========================================================
# 24. LITERAL SEMANTIC AGGREGATION
# =========================================================

def literal_semantic_aggregation(
    question,
    search_text
):

    # -----------------------------------------------------
    # A. CANDIDATE GENERATION
    # -----------------------------------------------------
    #
    # This is exhaustive.
    #
    # If search_text = "parfait", every occurrence of
    # "parfait" in the original document becomes a
    # candidate.
    # -----------------------------------------------------

    candidates = find_exact_occurrences(
        search_text
    )

    print(
        "\n================================"
    )

    print(
        "SEMANTIC AGGREGATION CANDIDATES"
    )

    print(
        "================================"
    )

    print(
        f"Candidate search text: "
        f"{repr(search_text)}"
    )

    print(
        f"Candidates found: "
        f"{len(candidates)}"
    )

    # -----------------------------------------------------
    # B. VERIFY EACH CANDIDATE
    # -----------------------------------------------------

    verified_matches = []

    for index, candidate in enumerate(
        candidates,
        start=1
    ):

        print(
            f"\nVerifying candidate "
            f"{index}/{len(candidates)}..."
        )

        verification = (
            verify_candidate(
                question,
                candidate
            )
        )

        print(
            "Page:",
            candidate["page"]
        )

        print(
            "Matches:",
            verification.matches
        )

        print(
            "Evidence:",
            verification.evidence
        )

        if verification.matches:

            verified_matches.append({
                **candidate,

                "evidence": (
                    verification.evidence
                )
            })

    # -----------------------------------------------------
    # C. RETURN VERIFIED RESULTS
    # -----------------------------------------------------

    return {
        "type": "semantic_literal",
        "search_text": search_text,
        "candidate_count": len(candidates),
        "count": len(verified_matches),
        "matches": verified_matches
    }


# =========================================================
# 25. AGGREGATION SEARCH
# =========================================================

def aggregation_search(
    question
):

    classification = (
        classify_aggregation(
            question
        )
    )

    # =====================================================
    # EXACT COUNT
    # =====================================================

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
                    "The aggregation classifier "
                    "did not provide text to count."
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

    # =====================================================
    # SEMANTIC AGGREGATION
    # =====================================================

    plan = create_semantic_plan(
        question
    )

    candidate_search_text = (
        plan.candidate_search_text
    )

    # -----------------------------------------------------
    # We have a deterministic candidate generator
    # -----------------------------------------------------

    if candidate_search_text:

        return (
            literal_semantic_aggregation(
                question,
                candidate_search_text
            )
        )

    # -----------------------------------------------------
    # No exhaustive literal candidate generator exists
    # -----------------------------------------------------

    return {
        "type": "semantic_broad",
        "message": (
            "This question requires broad semantic "
            "aggregation because there is no single "
            "literal search term guaranteed to occur "
            "in every valid answer. We will implement "
            "broad semantic candidate generation next."
        )
    }
