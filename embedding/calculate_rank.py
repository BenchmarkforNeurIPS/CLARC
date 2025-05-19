import pickle
import numpy as np
from tqdm import tqdm

with open("./TACO_retrieval_query_embeddings.pickle", 'rb') as f:
    query_emb = pickle.load(f)

with open("./TACO_retrieval_corpus_embeddings.pickle", 'rb') as f:
    cand_emb = pickle.load(f)

# with open("./TACO_retrieval_query_embeddings.pickle", 'rb') as f:
#     cand_emb = pickle.load(f)

# Define the rerank_by_cosine_similarity_matrix function (as defined earlier)
def rerank_by_cosine_similarity_matrix(q_emb, c_emb):
    q_keys, q_matrix = zip(*q_emb.items())
    c_keys, c_matrix = zip(*c_emb.items())

    q_matrix = np.vstack(q_matrix)
    c_matrix = np.vstack(c_matrix)

    q_norms = np.linalg.norm(q_matrix, axis=1, keepdims=True)
    c_norms = np.linalg.norm(c_matrix, axis=1, keepdims=True)

    q_matrix_normalized = q_matrix / q_norms
    c_matrix_normalized = c_matrix / c_norms

    similarities = np.dot(q_matrix_normalized, c_matrix_normalized.T)
    print("Cosine Similarity Finished")

    reranked = {}
    for i, q_key in tqdm(enumerate(q_keys)):
        similarity_scores = similarities[i]
        sorted_indices = np.argsort(similarity_scores)[::-1]
        sorted_c_keys = [c_keys[j] for j in sorted_indices]
        reranked[q_key] = sorted_c_keys

    return reranked

reranked_by_query = rerank_by_cosine_similarity_matrix(query_emb, cand_emb)

new_reranked_by_query = {}
for k in tqdm(reranked_by_query):
    new_reranked_by_query = reranked_by_query[:20000]

with open("./rank_by_codet5p_emb.pickle", 'wb') as f:
    pickle.dump(reranked_by_query, f)