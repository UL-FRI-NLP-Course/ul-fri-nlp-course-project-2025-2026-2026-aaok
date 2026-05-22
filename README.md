# Natural language processing course: AI Assistant for employment rights, contracts, and termination in Slovenia
In this work, we developed and evaluated a retrieval-augmented generation (RAG) conversational assistant for Slovenian labour law, built on the COLESLAW 1.0 legal corpus. We compared different retrieval strategies and found that a hybrid approach combining chunk-level semantic retrieval with BM25 keyword search, query year injection, and time-based filtering provides the best balance between recall and precision in selecting relevant legal sources. The system is designed to ground responses in authoritative legislation rather than relying solely on parametric model knowledge. In the generation stage, both evaluated language models performed well when given relevant context, with GaMS3-12B achieving the strongest overall performance and the most stable Slovenian language outputs. However, the evaluation also highlights remaining challenges in handling complex legal questions, including occasional omissions and hallucinated interpretations, underscoring the need for careful validation in high-stakes legal applications. Overall, the results demonstrate that retrieval-augmented language models provide a strong foundation for building practical AI tools in specialized legal domains such as Slovenian labour law.

## Repository structure
- `code/` - Project source code
- `code/source/` - External sources used in the project and how to download them 
- `report/` - Project reports

## Set-up and running the chatbot evaluation
See `code/README.md` for instructions and relevant information.
