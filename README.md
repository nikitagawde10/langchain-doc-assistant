                         PDF
                          │
                          ▼
                     PyPDFLoader
                          │
                          ▼
                    502 Documents
                          │
                          ▼
                     Text Splitter
                          │
                          ▼
                        Chunks
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
         BM25 Index              Embeddings
                                      │
                                      ▼
                                Vector Store


                    USER QUESTION
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
             BM25                  Vector
          lexical search         semantic search
              │                       │
            top 10                  top 10
              │                       │
              └───────────┬───────────┘
                          ▼
                         RRF
                  combines rankings
                          │
                          ▼
                  top 10 candidates
                          │
                          ▼
                    CrossEncoder
                          │
              question + each chunk
                          │
                          ▼
                  relevance scores
                          │
                          ▼
                    best 5 chunks
                          │
                          ▼
                  format_documents()
                          │
                          ▼
                        Prompt
                          │
                          ▼
                         Groq
                          │
                          ▼
                        Answer
