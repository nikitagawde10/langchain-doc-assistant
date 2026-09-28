from routes import (
    classify_query,
    semantic_search,
    exact_search,
    aggregation_search
)


# =========================================================
# MAIN APPLICATION
# =========================================================

while True:

    question = input(
        "\nAsk a question "
        "(or type 'exit'): "
    )

    # =====================================================
    # EXIT
    # =====================================================

    if question.lower() == "exit":

        print(
            "Goodbye!"
        )

        break

    # =====================================================
    # 1. ROUTE QUESTION
    # =====================================================

    classification = (
        classify_query(
            question
        )
    )

    query_type = (
        classification.query_type
    )

    print(
        "\n================================"
    )

    print(
        "QUERY ROUTER"
    )

    print(
        "================================"
    )

    print(
        f"Query type: {query_type}"
    )

    # =====================================================
    # 2. SEMANTIC
    # =====================================================

    if query_type == "semantic":

        result = (
            semantic_search(
                question
            )
        )

        print(
            "\n================================"
        )

        print(
            "ANSWER"
        )

        print(
            "================================"
        )

        print(
            result["answer"]
        )

        print(
            "\nRetrieved pages:",
            ", ".join(
                result["pages"]
            )
        )

    # =====================================================
    # 3. EXACT
    # =====================================================

    elif query_type == "exact":

        result = (
            exact_search(
                question
            )
        )

        print(
            "\n================================"
        )

        print(
            "EXACT SEARCH RESULTS"
        )

        print(
            "================================"
        )

        print(
            "\nSearch text:",
            repr(
                result["search_text"]
            )
        )

        print(
            "Total occurrences:",
            result["count"]
        )

        if not result["matches"]:

            print(
                "\nNo exact matches found."
            )

        else:

            for index, match in enumerate(
                result["matches"],
                start=1
            ):

                print(
                    f"\nMATCH {index}"
                )

                print(
                    "Page:",
                    match["page"]
                )

                print(
                    match["snippet"]
                )

                print(
                    "--------------------------------"
                )

    # =====================================================
    # 4. AGGREGATION
    # =====================================================

    elif query_type == "aggregation":

        result = (
            aggregation_search(
                question
            )
        )

        print(
            "\n================================"
        )

        print(
            "AGGREGATION RESULTS"
        )

        print(
            "================================"
        )

        # -------------------------------------------------
        # EXACT COUNT
        # -------------------------------------------------

        if (
            result["type"]
            == "exact_count"
        ):

            print(
                "\nAggregation type: "
                "EXACT COUNT"
            )

            print(
                "Search text:",
                repr(
                    result[
                        "search_text"
                    ]
                )
            )

            print(
                "Total occurrences:",
                result["count"]
            )

            # ---------------------------------------------
            # Show where occurrences happened
            # ---------------------------------------------

            pages = []

            for match in result[
                "matches"
            ]:

                page = match["page"]

                if page not in pages:

                    pages.append(
                        page
                    )

            print(
                "Pages:",
                ", ".join(pages)
            )

        # -------------------------------------------------
        # SEMANTIC AGGREGATION
        # -------------------------------------------------

        elif (
            result["type"]
            == "semantic"
        ):

            print(
                "\nAggregation type: "
                "SEMANTIC"
            )

            print(
                result["message"]
            )

        # -------------------------------------------------
        # ERROR
        # -------------------------------------------------

        else:

            print(
                result["message"]
            )
