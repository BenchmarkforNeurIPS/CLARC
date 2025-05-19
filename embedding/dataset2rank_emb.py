import argparse
import pickle
from emb_util import emb_with_voyage_code_3, emb_with_codet5_plus, emb_with_oasis, emb_with_nomic_code_compat, embed_dictionary_with_openai
from tqdm import tqdm
import numpy as np
import json
import pickle
import time
import re
from typing import Dict, List, Tuple, Optional, Set
from rank_bm25 import BM25Okapi

bm25_index_filepath = "./original_BM25_index_data.pickle" 
print(f"Loading BM25 index from {bm25_index_filepath}...")


def simple_tokenizer(text: str) -> List[str]:
    """Basic tokenizer: lowercase and split by non-alphanumeric characters."""
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

def bm25_rank(
    query: str,
    index_filepath: str,
    keys_to_rank: List[str]
) -> List[Tuple[str, float]]:
    """
    Loads a pre-computed BM25 index and ranks a specific list of document keys
    based on their relevance to a given query.

    Args:
        query: The search query string.
        index_filepath: Path to the pickled BM25 index data file.
        keys_to_rank: A list of document IDs (keys) that should be considered
                      for ranking. Only these keys will be included in the output.

    Returns:
        A list of (document_id, bm25_score) tuples, containing only keys from
        keys_to_rank, sorted by score descending.
        Returns an empty list if loading fails, query is empty, keys_to_rank
        is empty, or an error occurs.
    """
    # Set current time context for logging/potential use
    current_time = time.strftime("%Y-%m-%d %H:%M:%S %Z") # Example format

    start_time = time.time()

    if not keys_to_rank:
        print("Warning: The list of keys to rank is empty. Returning empty list.")
        return []

    # Use a set for efficient lookup of keys to rank
    keys_to_rank_set = set(keys_to_rank)

    # --- Loading Index ---
    # print("Loading BM25 index data...")
    try:
        with open(index_filepath, 'rb') as f_in:
            index_data = pickle.load(f_in)
        bm25_model = index_data['bm25_model']
        # Get the full list of IDs that were originally indexed
        all_indexed_doc_ids = index_data['doc_ids']
        # print(f"Index loaded successfully. Contains {len(all_indexed_doc_ids)} indexed documents.")
    except FileNotFoundError:
        print(f"Error: Index file not found at {index_filepath}")
        return []
    except IOError as e:
        print(f"Error loading index file: {e}")
        return []
    except (pickle.UnpicklingError, KeyError, AttributeError, EOFError) as e:
        print(f"Error unpickling or validating BM25 data: {e}. File might be corrupted/incompatible.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during index loading: {e}")
        return []

    # --- Query Processing ---
    if not query:
        print("Warning: Query is empty. Returning empty list.")
        return []

    try:
        tokenized_query = simple_tokenizer(query)
        if not tokenized_query:
             print("Warning: Query resulted in empty tokens. Returning empty list.")
             return []
    except Exception as e:
        print(f"Error during query tokenization: {e}")
        return []

    # --- Scoring (Calculate scores for ALL indexed documents first) ---
    try:
        # Calculate scores for the query against the entire indexed corpus
        all_scores = bm25_model.get_scores(tokenized_query)
        if len(all_scores) != len(all_indexed_doc_ids):
             print(f"Error: Number of scores ({len(all_scores)}) doesn't match indexed doc IDs ({len(all_indexed_doc_ids)}).")
             return []
    except Exception as e:
        print(f"Error calculating BM25 scores: {e}")
        return []

    # --- Filtering and Mapping Scores ---
    results_to_sort = []
    found_count = 0
    # Create a map for efficient lookup if needed, or iterate directly
    # Direct iteration with check is fine here:
    for i, doc_id in enumerate(all_indexed_doc_ids):
        if doc_id in keys_to_rank_set:
            results_to_sort.append((doc_id, all_scores[i]))
            found_count += 1

    # Optional: Check if all requested keys were found in the index
    if found_count != len(keys_to_rank_set):
        print(f"Warning: Found scores for {found_count} out of {len(keys_to_rank_set)} requested keys.")
        missing_keys = keys_to_rank_set - set(all_indexed_doc_ids)
        if missing_keys:
             print(f"  -> {len(missing_keys)} requested keys were not found in the loaded index: {list(missing_keys)[:10]}...") # Show a few missing keys

    if not results_to_sort:
         print("No scores found for any of the requested keys.")
         return []

    # --- Sorting Filtered Results ---
    results_to_sort.sort(key=lambda item: item[1], reverse=True)

    end_time = time.time()
    return results_to_sort

def sort_by_cosine_similarity(query_emb: np.ndarray, corpus_emb: dict) -> list:
    """
    Sorts the keys of corpus_emb based on cosine similarity with query_emb.
    
    Parameters:
        query_emb (np.ndarray): The query embedding vector.
        corpus_emb (dict): A dictionary with string keys and numpy array values representing embeddings.

    Returns:
        list: A list of keys sorted based on their cosine similarity with query_emb in descending order.
    """
    def cosine_similarity(vec1, vec2):
        # Compute cosine similarity
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    # Compute cosine similarities
    similarities = {
        key: cosine_similarity(query_emb, value)
        for key, value in corpus_emb.items()
    }
    
    # Sort keys based on cosine similarity in descending order
    sorted_keys = sorted(similarities, key=similarities.get, reverse=True)
    
    return sorted_keys

def bm_25_rank_dataset(query, corpus, emb_store_path, rank_store_path, group3_flag='None'):
    full_rank = {}
    key_to_rerank = list(corpus.keys())
    if group3_flag == 'None':
        key_to_rank = []
        for i in key_to_rerank:
            key_to_rank.append(i)
    else:
        if group3_flag == 'long':
            key_to_rank = []
            for i in key_to_rerank:
                if 'group_3' in i:
                    key_to_rank.append(i + "_long")
                else:
                    key_to_rank.append(i)
        if group3_flag == 'short':
            key_to_rank = []
            for i in key_to_rerank:
                if 'group_3' in i:
                    key_to_rank.append(i + "_short")
                else:
                    key_to_rank.append(i)

    
    
    for qid, query_text in tqdm(query.items()):
        curr_bm25_rank = bm25_rank(query_text, bm25_index_filepath, key_to_rank)
        converted_rank = [i[0].replace("_long", "").replace("_short", "") for i in curr_bm25_rank]
        full_rank[qid] = converted_rank

    with open(rank_store_path, 'wb') as f:
        pickle.dump(full_rank, f)
    print(f"Ranking results saved to {rank_store_path}")
    exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset_path", type = str, required = True, help = "Name of the test dataset")
    parser.add_argument("--emb_store_path", type = str, required = True, help = "Path to store the embedding results")
    parser.add_argument("--rank_store_path", type = str, required=True, help="Path to store the rank results")
    parser.add_argument("--model", type = str, required = True, help = "the model to use for embedding")
    parser.add_argument("--max_tokens", type = int, required = True, help = "Maximum number of tokens to use for each text")
    parser.add_argument("--batch_size", type = int, required = True, help = "Batch size for embedding")
    parser.add_argument("--loading_path", type = str, required = False, default="None", help = "Path to load the model")

    args = parser.parse_args()
    max_tokens = args.max_tokens
    batch_size = args.batch_size
    model_name = args.model
    if args.loading_path == "None":
        loading_path = None
    else:
        loading_path = args.loading_path

    # with open(args.dataset_path, 'rb') as f:
    #     dataset = pickle.load(f)
    with open(args.dataset_path, 'r') as f:
        dataset = json.load(f)

    assert 'query' in dataset and 'corpus' in dataset, "The dataset must contain both 'query' and 'corpus' keys"
    assert 'qrel' in dataset, "The dataset must contain 'qrel' key"

    queries = dataset['query']
    corpus = dataset['corpus']
    qrel = dataset['qrel']


    # if model_name not in ['voyage_ai', 'openai_text_embedding', 'openai_text_embedding', 'vic', 'code_t5p', 'graph_codebert']:
    #     raise ValueError("The model name must be one of 'voyage_ai', 'openai_text_embedding', 'openai_text_embedding', 'vic', 'code_t5p'")
    
    emb_func = None

    if model_name == 'voyage_ai':
        emb_func = emb_with_voyage_code_3
    elif model_name == 'code_t5p':
        emb_func = emb_with_codet5_plus
    elif model_name == 'oasis':
        emb_func = emb_with_oasis
    elif model_name == 'nomic':
        emb_func = emb_with_nomic_code_compat
    elif model_name == 'openai':
        emb_func = embed_dictionary_with_openai
    elif model_name == 'bm25':
        group3_flag = 'None'
        if "group3" in args.dataset_path:
            if "helper_as_part_of_groundtruth" in args.dataset_path:
                group3_flag = 'long'
            elif "helper_as_other_candidates":
                group3_flag = 'short'
        bm_25_rank_dataset(queries, corpus, args.emb_store_path, args.rank_store_path, group3_flag)
    else:
        raise ValueError("Not supported yet")
    

    
    query_emb = emb_func(queries, max_tokens=max_tokens, batch_size=batch_size, mode='query', loading_path=loading_path)
    corpus_emb = emb_func(corpus, max_tokens=max_tokens, batch_size=batch_size, mode='corpus', loading_path=loading_path)

    embedding_results = {'query_emb': query_emb, 'corpus_emb': corpus_emb}
    with open(args.emb_store_path, 'wb') as f:
        pickle.dump(embedding_results, f)

    ranking_results = {}

    for qid in tqdm(qrel):
        relevant_corpus_emb = {}
        for cid in qrel[qid]:
            relevant_corpus_emb[cid] = corpus_emb[cid]
        
        sorted_cids = sort_by_cosine_similarity(query_emb[qid], relevant_corpus_emb)
        ranking_results[qid] = sorted_cids

    with open(args.rank_store_path, 'wb') as f:
        pickle.dump(ranking_results, f)

    print(f"Ranking results saved to {args.rank_store_path}")


