from datetime import datetime

from database.vector_store import VectorStore
from services.synthesizer import Synthesizer
from timescale_vector import client

# Initialize VectorStore
vec = VectorStore()


def main(query: str):
    # --------------------------------------------------------------
    # DeepAR question
    # --------------------------------------------------------------

    relevant_question = query

    # --------------------------------------------------------------
    # Semantic search
    # --------------------------------------------------------------

    semantic_results = vec.semantic_search(relevant_question, limit=3)
    print(semantic_results)
    # --------------------------------------------------------------
    # Keyword search
    # --------------------------------------------------------------

    keyword_results = vec.keyword_search(relevant_question, limit=3)
    print(keyword_results)

    # --------------------------------------------------------------
    # Hybrid search
    # --------------------------------------------------------------

    hybrid_results = vec.hybrid_search(relevant_question, keyword_k=3)
    print(hybrid_results)

    # --------------------------------------------------------------
    # Reranking
    # --------------------------------------------------------------

    reranked_results = vec.hybrid_search(
        relevant_question, keyword_k=5, semantic_k=5, rerank=True
    )
    print(reranked_results)

    # --------------------------------------------------------------
    # Search
    # --------------------------------------------------------------

    response = Synthesizer.generate_response(
        question=relevant_question, context=reranked_results
    )

    return {
        "answer": response.answer,
        "thought_process": response.thought_process,
        "enough_context": response.enough_context,
    }


if __name__ == "__main__":
    main()
