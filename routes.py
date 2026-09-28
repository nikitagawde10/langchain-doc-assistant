from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import (
    ChatPromptTemplate
)
from langchain_core.output_parsers import (
    StrOutputParser
)
import time
import random

from groq import RateLimitError
from retrieval import (
    documents,
    hybrid_retrieve,
    broad_hybrid_retrieve
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
# 4. ROUTER
# =========================================================
router_model = (
    model.with_structured_output(
        QueryClassification
    )
)
router_prompt = (
    ChatPromptTemplate.from_messages([
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
Use when answering requires searching broadly across
the document.
Examples:
"How many times is Hoseok mentioned?"
-> aggregation
"How many times does Jungkook say parfait?"
-> aggregation
"List every time Jungkook compliments Jimin."
-> aggregation
Ask yourself:
"Must I inspect broadly across the document to answer
this completely?"
If yes:
-> aggregation
Otherwise, if exact matching is unnecessary:
-> semantic
"""
        ),
        (
            "human",
            "{question}"
        )
    ])
)
router_chain = (
    router_prompt
    | router_model
)


def classify_query(
    question
):
    result = (
        router_chain.invoke({
            "question": question
        })
    )
    if isinstance(
        result,
        QueryClassification
    ):
        return result
    return (
        QueryClassification.model_validate(
            result
        )
    )
# =========================================================
# 5. FORMAT DOCUMENTS
# =========================================================


def format_documents(
    retrieved_documents
):
    formatted = []
    for document in retrieved_documents:
        page = (
            document.metadata.get(
                "page_label",
                "Unknown"
            )
        )
        formatted.append(
            f"[Page {page}]\n"
            f"{document.page_content}"
        )
    return "\n\n".join(
        formatted
    )


# =========================================================
# 6. NORMAL SEMANTIC ANSWERING
# =========================================================
answer_prompt = (
    ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a document question-answering assistant.
Answer the user's question using only the provided
context.
You may make an inference only when the context strongly
supports it.
Clearly identify inference rather than presenting it as
an explicitly stated fact.
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
)
answer_chain = (
    answer_prompt
    | model
    | StrOutputParser()
)


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
        page = (
            document.metadata.get(
                "page_label",
                "Unknown"
            )
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
    context = (
        format_documents(
            retrieved_documents
        )
    )
    answer = (
        answer_chain.invoke({
            "context": context,
            "question": question
        })
    )
    pages = []
    for document in retrieved_documents:
        page = (
            document.metadata.get(
                "page_label"
            )
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
# 7. EXACT SEARCH
# =========================================================


class ExactSearchQuery(BaseModel):
    search_text: str = Field(
        description=(
            "The exact word or phrase to search "
            "for in the document."
        )
    )


exact_model = (
    model.with_structured_output(
        ExactSearchQuery
    )
)
exact_prompt = (
    ChatPromptTemplate.from_messages([
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
)
exact_chain = (
    exact_prompt
    | exact_model
)


def extract_search_text(
    question
):
    result = (
        exact_chain.invoke({
            "question": question
        })
    )
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
    return (
        extraction.search_text.strip()
    )


def find_exact_occurrences(
    search_text
):
    matches = []
    needle = (
        search_text.lower()
    )
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
# 8. AGGREGATION CLASSIFIER
# =========================================================


class AggregationClassification(BaseModel):
    aggregation_type: Literal[
        "exact_count",
        "semantic"
    ] = Field(
        description=(
            "Whether the aggregation requires "
            "literal counting or semantic verification."
        )
    )
    search_text: str = Field(
        default="",
        description=(
            "The literal text to count when "
            "aggregation_type is exact_count. "
            "Use an empty string for semantic aggregation."
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
EXACT_COUNT:
Use when answering only requires counting occurrences
of a literal word, name, or phrase.
Example:
"How many times is Hoseok mentioned?"
-> exact_count
-> search_text = "Hoseok"
SEMANTIC:
Use when occurrences must be examined to determine
whether they satisfy some condition.
Examples:
"How many times does Jungkook say parfait?"
-> semantic
"List every time Jungkook compliments Jimin."
-> semantic
If determining WHO performed an action or WHAT an event
means requires reading context, classify it as semantic.
IMPORTANT:
Never use null or None for search_text.
For exact_count:
search_text must contain the literal text to count.
For semantic:
search_text must be an empty string.
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


def classify_aggregation(
    question
):
    result = (
        aggregation_chain.invoke({
            "question": question
        })
    )
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
# 9. SEMANTIC AGGREGATION PLAN
# =========================================================


class SemanticAggregationPlan(BaseModel):
    strategy: Literal[
        "literal",
        "broad"
    ] = Field(
        description=(
            "Use 'literal' when every valid answer "
            "must contain a specific searchable word "
            "or phrase. Use 'broad' when no single "
            "literal search term can guarantee finding "
            "every valid answer."
        )
    )
    candidate_search_text: str = Field(
        default="",
        description=(
            "The literal word or phrase to search for "
            "when strategy is 'literal'. "
            "Use an empty string when strategy is 'broad'."
        )
    )
    strategy: Literal[
        "literal",
        "broad"
    ]
    candidate_search_text: str = ""


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
Determine whether this semantic aggregation question has
a literal word or phrase that can generate ALL possible
candidates.
Examples:
Question:
"How many times does Jungkook say parfait?"
candidate_search_text:
parfait
Question:
"How many times does Jimin say sorry?"
candidate_search_text:
sorry
Question:
"List every time Jungkook compliments Jimin."
candidate_search_text:
null
Only provide candidate_search_text when EVERY valid
answer must necessarily contain that literal text.
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
# 10. VERIFICATION
# =========================================================


class VerificationResult(BaseModel):
    matches: bool
    evidence: str | None = None


verification_model = (
    model.with_structured_output(
        VerificationResult
    )
)
verification_prompt = (
    ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are verifying whether a passage satisfies a user's
document question.
Be strict.
Return matches=true ONLY if the passage contains enough
evidence to establish that the requested event or
condition actually occurs.
For dialogue questions, identify who actually speaks.
For relationship or action questions, verify the people
involved.
Do not mark a passage true merely because the relevant
characters are mentioned.
If the passage is ambiguous, return matches=false.
Evidence should briefly explain the relevant event.
"""
        ),
        (
            "human",
            """
Question:
{question}
Passage:
{passage}
"""
        )
    ])
)
verification_chain = (
    verification_prompt
    | verification_model
)


def verify_passage(
    question,
    passage,
    max_retries=5
):

    for attempt in range(max_retries):

        try:

            result = (
                verification_chain.invoke({
                    "question": question,
                    "passage": passage
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

        except RateLimitError as error:

            # Exponential backoff:
            # ~1 sec, 2 sec, 4 sec, 8 sec, 16 sec
            wait_time = (
                (2 ** attempt)
                + random.uniform(0, 0.5)
            )

            print(
                "\nGroq rate limit reached."
            )

            print(
                f"Waiting {wait_time:.2f} seconds "
                f"before retry "
                f"{attempt + 1}/{max_retries}..."
            )

            time.sleep(
                wait_time
            )

    print(
        "\nVerification failed after "
        f"{max_retries} retries."
    )

    return VerificationResult(
        matches=False,
        evidence=(
            "Verification could not be completed "
            "because the model repeatedly hit "
            "the API rate limit."
        )
    )
# =========================================================
# 11. LITERAL SEMANTIC AGGREGATION
# =========================================================


def literal_semantic_aggregation(
    question,
    search_text
):
    candidates = (
        find_exact_occurrences(
            search_text
        )
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
        "Candidate search text:",
        repr(search_text)
    )
    print(
        "Candidates found:",
        len(candidates)
    )
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
            verify_passage(
                question,
                candidate["snippet"]
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
    return {
        "type": "semantic_literal",
        "search_text": search_text,
        "candidate_count": len(candidates),
        "count": len(verified_matches),
        "matches": verified_matches
    }
# =========================================================
# 12. QUERY EXPANSION
# =========================================================


class QueryExpansion(BaseModel):
    queries: list[str] = Field(
        description=(
            "Different semantic retrieval queries "
            "representing the user's request."
        )
    )


query_expansion_model = (
    model.with_structured_output(
        QueryExpansion
    )
)
query_expansion_prompt = (
    ChatPromptTemplate.from_messages([
        (
            "system",
            """
Generate several different search queries for retrieving
passages relevant to the user's request.
The goal is HIGH RECALL.
Generate 5 concise retrieval queries.
Express the same underlying event in different ways.
Do not answer the question.
Do not invent specific events from the document.
Example:
User:
"List every time Jungkook compliments Jimin."
Possible retrieval queries:
- Jungkook compliments Jimin
- Jungkook praises Jimin
- Jungkook approves of Jimin
- Jungkook says something positive to Jimin
- Jungkook praises Jimin's cooking
The queries should overlap conceptually but use different
wording so semantic retrieval has multiple chances to find
relevant passages.
"""
        ),
        (
            "human",
            "{question}"
        )
    ])
)
query_expansion_chain = (
    query_expansion_prompt
    | query_expansion_model
)


def expand_query(
    question
):
    result = (
        query_expansion_chain.invoke({
            "question": question
        })
    )
    if isinstance(
        result,
        QueryExpansion
    ):
        expansion = result
    else:
        expansion = (
            QueryExpansion.model_validate(
                result
            )
        )
    # Include the original question as well.
    queries = [
        question,
        *expansion.queries
    ]
    # Remove duplicate generated queries.
    unique_queries = []
    seen = set()
    for query in queries:
        normalized = (
            query.strip().lower()
        )
        if normalized in seen:
            continue
        seen.add(
            normalized
        )
        unique_queries.append(
            query.strip()
        )
    return unique_queries
# =========================================================
# 13. BROAD SEMANTIC AGGREGATION
# =========================================================


def broad_semantic_aggregation(
    question
):
    # -----------------------------------------------------
    # A. QUERY EXPANSION
    # -----------------------------------------------------
    queries = (
        expand_query(
            question
        )
    )
    print(
        "\n================================"
    )
    print(
        "QUERY EXPANSION"
    )
    print(
        "================================"
    )
    for index, query in enumerate(
        queries,
        start=1
    ):
        print(
            f"{index}. {query}"
        )
    # -----------------------------------------------------
    # B. HIGH-RECALL CANDIDATE GENERATION
    # -----------------------------------------------------
    candidates = (
        broad_hybrid_retrieve(
            queries,
            candidates_per_query=20
        )
    )
    print(
        "\n================================"
    )
    print(
        "BROAD CANDIDATES"
    )
    print(
        "================================"
    )
    print(
        "Unique candidates:",
        len(candidates)
    )
    # -----------------------------------------------------
    # C. VERIFY CANDIDATES
    # -----------------------------------------------------
    verified_matches = []
    for index, document in enumerate(
        candidates,
        start=1
    ):
        page = (
            document.metadata.get(
                "page_label",
                "Unknown"
            )
        )
        print(
            f"\nVerifying candidate "
            f"{index}/{len(candidates)} "
            f"(page {page})..."
        )
        verification = (
            verify_passage(
                question,
                document.page_content
            )
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
                "page": page,
                "content": (
                    document.page_content
                ),
                "evidence": (
                    verification.evidence
                )
            })
    # -----------------------------------------------------
    # D. BASIC EVENT DEDUPLICATION
    # -----------------------------------------------------
    #
    # Because our chunks overlap by 500 characters,
    # the same event may appear in multiple chunks.
    #
    # For now, collapse identical evidence on the
    # same page.
    #
    # We'll improve event-level deduplication later.
    # -----------------------------------------------------
    deduplicated_matches = []
    seen = set()
    for match in verified_matches:
        evidence = (
            match["evidence"] or ""
        )
        key = (
            str(match["page"]),
            evidence.strip().lower()
        )
        if key in seen:
            continue
        seen.add(
            key
        )
        deduplicated_matches.append(
            match
        )
    return {
        "type": "semantic_broad",
        "queries": queries,
        "candidate_count": len(candidates),
        "verified_before_dedup": (
            len(verified_matches)
        ),
        "count": (
            len(deduplicated_matches)
        ),
        "matches": (
            deduplicated_matches
        )
    }
# =========================================================
# 14. AGGREGATION SEARCH
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
    # =====================================================
    # SEMANTIC
    # =====================================================
    plan = (
        create_semantic_plan(
            question
        )
    )
    print(
        "\n================================"
    )
    print(
        "SEMANTIC AGGREGATION PLAN"
    )
    print(
        "================================"
    )
    print(
        "Strategy:",
        plan.strategy
    )
    print(
        "Candidate search text:",
        repr(
            plan.candidate_search_text
        )
    )
    # =====================================================
    # LITERAL STRATEGY
    # =====================================================
    if plan.strategy == "literal":
        candidate_search_text = (
            plan.candidate_search_text.strip()
        )
        if not candidate_search_text:
            return {
                "type": "error",
                "message": (
                    "The semantic planner selected "
                    "the literal strategy but did not "
                    "provide candidate search text."
                )
            }
        return (
            literal_semantic_aggregation(
                question,
                candidate_search_text
            )
        )
    # =====================================================
    # BROAD STRATEGY
    # =====================================================
    return (
        broad_semantic_aggregation(
            question
        )
    )
