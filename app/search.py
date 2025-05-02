from datetime import datetime

from database.vector_store import VectorStore
from services.synthesizer import Synthesizer
from timescale_vector import client

# Initialize VectorStore
vec = VectorStore()

# --------------------------------------------------------------
# DeepAR question
# --------------------------------------------------------------

relevant_question = "What is DeepAR?"

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

print(f"\n{response.answer}")
print("\nThought process:")
for thought in response.thought_process:
    print(f"- {thought}")
print(f"\nContext: {response.enough_context}")
