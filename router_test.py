from dotenv import load_dotenv

from typing import Literal
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


# =========================================================
# 1. DEFINE STRUCTURED OUTPUT
# =========================================================

class QueryClassification(BaseModel):

    query_type: Literal[
        "semantic",
        "exact",
        "aggregation"
    ] = Field(
        description="The retrieval strategy required for the question."
    )


# =========================================================
# 2. CREATE MODEL
# =========================================================

model = ChatGroq(
    model="openai/gpt-oss-20b"
)


# =========================================================
# 3. REQUIRE STRUCTURED OUTPUT
# =========================================================

router_model = model.with_structured_output(
    QueryClassification
)


# =========================================================
# 4. ROUTER PROMPT
# =========================================================

router_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
    Classify the user's question about a PDF by the retrieval
    strategy needed to answer it. Choose exactly one category based
    on the question's intent, not on any assumed subject, genre, or
    domain of the PDF.

    SEMANTIC:
    Use when the user asks for information, explanation, or a
    description that can likely be answered from one or a small
    number of relevant passages by understanding their meaning.
    This includes questions about entities, concepts, events,
    relationships, causes, and facts, regardless of the PDF's topic.

    EXACT:
    Use when the user explicitly asks to find or locate a specific
    literal word, phrase, quotation, or text occurrence. The main
    task is matching the requested text, not interpreting its meaning.

    AGGREGATION:
    Use when a complete answer requires searching across the
    document to count, total, compare frequencies, or collect every
    or all instances that meet a condition. A request for an exact
    text's count is aggregation, because it asks for a total; a
    request to locate that text is exact.

    Do not choose aggregation merely because the question mentions
    multiple items or asks for more than one fact. Choose it when
    the user asks for a total, an exhaustive list, or otherwise
    requires broad document-wide coverage. If the question does not
    require literal text matching or exhaustive coverage, choose
    semantic.

    Return only the required structured classification.
"""
    ),
    (
        "human",
        "{question}"
    )
])
# =========================================================
# 5. CREATE ROUTER CHAIN
# =========================================================

router_chain = (
    router_prompt
    | router_model
)


# =========================================================
# 6. TEST
# =========================================================

while True:

    question = input(
        "\nAsk a question (or type 'exit'): "
    )

    if question.lower() == "exit":
        break

    result = router_chain.invoke({
        "question": question
    })
    classification = (
        result
        if isinstance(result, QueryClassification)
        else QueryClassification.model_validate(result)
    )

    print("\nRESULT:")
    print(result)

    print("\nTYPE:")
    print(type(result))

    print("\nQUERY TYPE:")
    print(classification.query_type)
