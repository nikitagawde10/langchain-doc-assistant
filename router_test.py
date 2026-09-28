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
You classify document questions by the SEARCH STRATEGY needed
to answer them.

Choose exactly one category:

SEMANTIC:
Use when the answer can likely be found in one or a small number
of relevant passages using meaning and context.

This includes questions about:
- people
- relationships
- events
- explanations
- descriptions
- actions
- facts about one or several named entities

A question is still semantic when it asks about multiple people
or asks for multiple facts, as long as it does NOT require
exhaustively searching the entire document.

Examples:
"Who is Hoseok?" -> semantic
"Why is Jimin angry?" -> semantic
"What ice cream does Jimin eat?" -> semantic
"Which ice creams do Jimin and Taehyung eat?" -> semantic


EXACT:
Use when the user explicitly wants to find or locate a specific
word, phrase, quotation, or exact textual occurrence.

Examples:
"Find the phrase 'mise en place'." -> exact
"Where does the phrase 'parfait' appear?" -> exact
"Show me the sentence containing 'golden maknae'." -> exact


AGGREGATION:
Use ONLY when answering requires an EXHAUSTIVE search across
the document.

Typical aggregation questions ask for:
- how many times something occurs
- every occurrence
- all instances
- a complete list across the document
- totals or frequencies

Examples:
"How many times does Jungkook say 'parfait'?" -> aggregation
"List every time Jungkook compliments Jimin." -> aggregation
"How many times is Hoseok mentioned?" -> aggregation

IMPORTANT:
Do NOT classify a question as aggregation merely because it
mentions multiple people, objects, or facts.

Ask yourself:

"Do I need to search the ENTIRE document to guarantee that
the answer is complete?"

If YES -> aggregation.
If NO and exact text matching is not required -> semantic.

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


    print("\nRESULT:")
    print(result)

    print("\nTYPE:")
    print(type(result))

    print("\nQUERY TYPE:")
    print(result.query_type)