import argparse
import sys
import numpy as np
import pickle
import os
import math
from datetime import datetime
import pytz
import csv
import math
import pdb

def get_west_coast_time():
    # Define the timezone for US West Coast (Pacific Time)
    pacific_time = pytz.timezone('America/Los_Angeles')

    # Get the current time in Pacific Time
    now = datetime.now(pacific_time)

    # Format the time as mm_dd_hour_minute
    formatted_time = now.strftime('%m_%d_%H_%M')
    
    return formatted_time


def add_line_to_csv(file_path, new_data):
    """
    Adds a new line of data to an existing CSV file.
    
    :param file_path: The path to the CSV file.
    :param new_data: A dictionary where keys are column names, and values are the corresponding row data.
    """
    try:
        file_exists = os.path.exists(file_path)

        if file_exists:
            # Read the existing header
            with open(file_path, mode='r', newline='') as file:
                reader = csv.reader(file)
                existing_columns = next(reader)  # Read the header
        else:
            # If file doesn't exist, create with new data columns
            existing_columns = []

        # Add any new columns from the new data dictionary
        new_columns = list(new_data.keys())
        all_columns = list(existing_columns)  # Keep the original columns order
        for col in new_columns:
            if col not in all_columns:
                all_columns.append(col)

        # If the file does not exist, create it with the appropriate header
        if not file_exists:
            with open(file_path, mode='w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=all_columns)
                writer.writeheader()
            print(f'CSV file {file_path} created with column names: {all_columns}')

        # Open the file in append mode and add the new data
        with open(file_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=all_columns)

            # Fill missing columns with empty strings
            row = {col: new_data.get(col, '') for col in all_columns}
            writer.writerow(row)
        
        print("New line added successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Precision at k: It measures the proportion of recommended items in the top-k set that are relevant
def precision_at_k(relevance_list, ranking_result, k=20):
    """
    Parameters:
        relevance_list (list): List of relevant document ids.
        ranking_result (list): List that shows the ranking of the documents.
        k (int): The 'k' in 'P@k'. Default is 20.

    Returns:
        float: The precision at k of the ranking result.
    """
    ranking_result = ranking_result[:k]
    relevance_set = set(relevance_list)
    ranking_set = set(ranking_result)
    inter_size = len(relevance_set.intersection(ranking_set))
    return inter_size / k


# Recall at k: It measures the proportion of relevant items found in the top-k recommendations
def recall_at_k(relevance_list, ranking_result, k=20):
    """
    Parameters:
        relevance_list (list): List of relevant document ids.
        ranking_result (list): List that shows the ranking of the documents.
        k (int): The 'k' in 'R@k'. Default is 20.

    Returns:
        float: The recall at k of the ranking result.
    """
    ranking_result = ranking_result[:k]
    relevance_set = set(relevance_list)
    ranking_set = set(ranking_result)

    inter_size = len(relevance_set.intersection(ranking_set))

    # return k/len(relevance_list)
    return inter_size / len(relevance_list)


def r_precision(relevance_list, ranking_result):
    """
    Parameters:
        relevance_list (list): List of relevant document ids.
        ranking_result (list): List that shows the ranking of the documents.

    Returns:
        float: R-precision of the ranking result.
    """
    relevance_len = len(relevance_list)
    ranking_result = ranking_result[:relevance_len]
    relevance_set = set(relevance_list)
    ranking_set = set(ranking_result)

    inter_size = len(relevance_set.intersection(ranking_set))
    return inter_size / relevance_len


# Normalized Discounted Cumulative Gain (NDCG): A measure of ranking quality. It uses the graded relevance of a query result set and discounts the relevance of documents lower down in the result list.
def ndcg_normal(ground_truth_ranking, ground_truth_score, ranking_result, p):
    """
    Parameters:
        ground_truth_ranking (list): List that shows the correct ranking of the documents.
        ground_truth_score (list): List that shows the correct score of the documents.
        ranking_result (list): List that shows the ranking of the documents.
        p (float): A float between 0 and 1

    Returns:
        float: NDCG of the ranking result given the ground truth ranking and score.
    """
    if p == -1:
        consider_length = len(ground_truth_ranking)
    else:
        consider_length = p
    dcg = 0
    for ind in range(consider_length):
        curr_id = ranking_result[ind]
        pos_in_ground_truth = ground_truth_ranking.index(curr_id)
        rel_i = ground_truth_score[pos_in_ground_truth]
        dcg += rel_i / math.log2(ind + 2)
    idcg = 0
    for ind in range(consider_length):
        rel_i = ground_truth_score[ind]
        idcg += rel_i / math.log2(ind + 2)
    return dcg / idcg


# Alternate version of NDCG: Similar to NDCG, but this version uses a exponential gain to emphasise the importance of relevance.
def ndcg_exp(ground_truth_ranking, ground_truth_score, ranking_result, p):
    """
    Parameters:
        ground_truth_ranking (list): List that shows the correct ranking of the documents.
        ground_truth_score (list): List that shows the correct score of the documents.
        ranking_result (list): List that shows the ranking of the documents.
        p (float): A float between 0 and 1

    Returns:
        float: NDCG (exponential) of the ranking result given the ground truth ranking and score.
    """
    consider_length = 10
    dcg = 0
    for ind in range(consider_length):
        curr_id = ranking_result[ind]
        pos_in_ground_truth = ground_truth_ranking.index(curr_id)
        rel_i = ground_truth_score[pos_in_ground_truth]
        # print(rel_i)
        dcg += (2 ** rel_i) / math.log2(ind + 2)
        # dcg += rel_i/math.log2(ind + 2)
    idcg = 0
    for ind in range(consider_length):
        rel_i = ground_truth_score[ind]
        idcg += (2 ** rel_i) / math.log2(ind + 2)
        # idcg += rel_i / math.log2(ind + 2)
    return dcg / idcg


# Reciprocal Rank: The reciprocal of the rank of the first relevant document
def reciprocal_rank(ground_truth_rank, ground_truth_score, ranking_result, k):
    """
    Parameters:
        ground_truth_rank (list): List that shows the correct ranking of the documents.
        ground_truth_score (list): List that shows the correct score of the documents.
        ranking_result (list): List that shows the ranking of the documents.
        k (int): An integer between 1 and the length of the ranking result

    Returns:
        float: Reciprocal rank of the ranking result.
    """

    visible_ranking_result = ranking_result[:k]
    max_score = ground_truth_score[0]
    for ind in range(len(ground_truth_rank)):
        if ground_truth_score[ind] < max_score:
            break
    relevance_docs = ground_truth_rank[:ind]
    for i in range(len(visible_ranking_result)):
        if visible_ranking_result[i] in relevance_docs:
            return 1 / (i + 1)
    return 0


def average_precision(relevance_list, ranking_result):
    """
    Parameters:
        relevance_list (list): List of relevant document ids.
        ranking_result (list): List that shows the ranking of the documents.

    Returns:
        float: R-precision of the ranking result.
    """
    relevant_docs = set(relevance_list)
    num_relevant_docs = len(relevant_docs)
    if num_relevant_docs == 0:
        return 0.0
    cum_sum_precisions = 0.0
    num_hits = 0
    for i, doc_id in enumerate(ranking_result):
        if doc_id in relevant_docs:
            num_hits += 1
            precision_at_i = num_hits / (i + 1.0)
            cum_sum_precisions += precision_at_i
    ret = cum_sum_precisions / num_relevant_docs
    return ret


def evaluate_performance(model_rank, q_rels, relevance_threshold=1):
    average_p_list = []
    for doc_id in q_rels:
        curr_relevant_list = []
        for cand_id in q_rels[doc_id]:
            if q_rels[doc_id][cand_id] >= relevance_threshold:
                curr_relevant_list.append(cand_id)
        if len(curr_relevant_list) == 0:
            continue
        if doc_id not in model_rank:
            print("Continued...Please FIX in util.py")
            continue
        curr_model_rank = model_rank[doc_id]

        average_p_list.append(average_precision(curr_relevant_list, curr_model_rank))

    _map = average_p_list

    

    ndcg_list = []
    for doc_id in q_rels:
        
        curr_candidate_and_score = q_rels[doc_id]
        sorted_candidates = dict(sorted(curr_candidate_and_score.items(), key=lambda item: item[1], reverse=True))
        curr_ground_truth_ranking = list(sorted_candidates.keys())
        curr_ground_truth_score = list(sorted_candidates.values())
        if doc_id not in model_rank:
            print("Continued...Please FIX in util.py")
            continue
        curr_model_rank = model_rank[doc_id]

        # pdb.set_trace()

        ndcg_list.append(ndcg_normal(curr_ground_truth_ranking, curr_ground_truth_score, curr_model_rank, -1))
        
    recall_list = []
    for doc_id in q_rels:
        curr_relevant_list = []
        for cand_id in q_rels[doc_id]:
            if q_rels[doc_id][cand_id] >= relevance_threshold:
                curr_relevant_list.append(cand_id)
        if len(curr_relevant_list) == 0:
            continue
        if doc_id not in model_rank:
            print("Continued...Please FIX in util.py")
            continue
        curr_model_rank = model_rank[doc_id]

        recall_list.append(recall_at_k(curr_relevant_list, curr_model_rank, k=5))

    r_10 = []
    for doc_id in q_rels:
        curr_relevant_list = []
        for cand_id in q_rels[doc_id]:
            if q_rels[doc_id][cand_id] >= relevance_threshold:
                curr_relevant_list.append(cand_id)
        if len(curr_relevant_list) == 0:
            continue
        if doc_id not in model_rank:
            print("Continued...Please FIX in util.py")
            continue
        curr_model_rank = model_rank[doc_id]

        r_10.append(recall_at_k(curr_relevant_list, curr_model_rank, k=10))
        
    r_20 = []
    for doc_id in q_rels:
        curr_relevant_list = []
        for cand_id in q_rels[doc_id]:
            if q_rels[doc_id][cand_id] >= relevance_threshold:
                curr_relevant_list.append(cand_id)
        if len(curr_relevant_list) == 0:
            continue
        if doc_id not in model_rank:
            print("Continued...Please FIX in util.py")
            continue
        curr_model_rank = model_rank[doc_id]

        r_20.append(recall_at_k(curr_relevant_list, curr_model_rank, k=20))

    r_100 = []
    for doc_id in q_rels:
        curr_relevant_list = []
        for cand_id in q_rels[doc_id]:
            if q_rels[doc_id][cand_id] >= relevance_threshold:
                curr_relevant_list.append(cand_id)
        if len(curr_relevant_list) == 0:
            continue
        if doc_id not in model_rank:
            print("Continued...Please FIX in util.py")
            continue
        curr_model_rank = model_rank[doc_id]

        r_100.append(recall_at_k(curr_relevant_list, curr_model_rank, k=100))

    mrr = []
    for doc_id in q_rels:
        curr_candidate_and_score = q_rels[doc_id]
        sorted_candidates = dict(sorted(curr_candidate_and_score.items(), key=lambda item: item[1], reverse=True))
        curr_ground_truth_ranking = list(sorted_candidates.keys())
        curr_ground_truth_score = list(sorted_candidates.values())
        if doc_id not in model_rank:
            print("Continued...Please FIX in util.py")
            continue
        curr_model_rank = model_rank[doc_id]
    
        mrr.append(reciprocal_rank(curr_ground_truth_ranking, curr_ground_truth_score, curr_model_rank, 10))

    p_1 = []
    for doc_id in q_rels:
        curr_precision_list = []
        for cand_id in q_rels[doc_id]:
            if q_rels[doc_id][cand_id] >= relevance_threshold:
                curr_precision_list.append(cand_id)
        if len(curr_precision_list) == 0:
            continue
        if doc_id not in model_rank:
            print("Continued...Please FIX in util.py")
            continue
        curr_model_rank = model_rank[doc_id]

        p_1.append(precision_at_k(curr_precision_list, curr_model_rank, k=1))


    return recall_list, r_10, r_20, r_100, ndcg_list, mrr, _map, p_1


def bootstrap_stats(result_list, num_of_trial=1000, random_seed=42):
    np.random.seed(random_seed)
    bs_result = []
    for ind in range(num_of_trial):
        curr_sample_ind = list(np.random.choice(range(len(result_list)), len(result_list), replace=True))
        curr_sample = [result_list[i] for i in curr_sample_ind]
        bs_result.append(np.mean(curr_sample))
    return np.mean(bs_result), np.std(bs_result)

def main(corpus, query, qrel, dataset_name, rank_path = None):

    if not os.path.exists(rank_path):
       raise RuntimeError("The rank file does NOT exist!")
    else:
    
        with open(rank_path, 'rb') as f:
            model_rank = pickle.load(f)

        # remove the query itself from the rank list for arguana
        for key in model_rank:
            if key in model_rank[key]:
                model_rank[key].remove(key)
            
    relevance_threshold = 1
            
    recall_list, r_10, r_20, r_100, ndcg_list, mrr, _map, p_1 = evaluate_performance(model_rank, qrel, relevance_threshold = relevance_threshold)

    return recall_list, r_10, r_20, r_100, ndcg_list, mrr, _map, p_1

