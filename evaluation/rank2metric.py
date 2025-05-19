import argparse
import sys
import numpy as np
import pickle
import os
from rank_util import main, bootstrap_stats, get_west_coast_time, add_line_to_csv
import pdb
import json

def test_run(dataset_path, boot, rank_path):
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    corpus = dataset['corpus']
    query = dataset['query']
    qrel = dataset['qrel']

    recall_list, r_10, r_20, r_100, ndcg, mrr, _map, p_1 = main(corpus, query, qrel, dataset_name, rank_path = rank_path)

    if boot.lower() == 'true':

        print("The bootstrap default number is 1000...")
    
        r_at_5_mean, r_at_5_std = bootstrap_stats(recall_list, num_of_trial=1000, random_seed=42)
        print(f"R@5: {round(r_at_5_mean*100, 3)} +/- {round(r_at_5_std*100, 3)}")

        r_at_10_mean, r_at_10_std = bootstrap_stats(r_10, num_of_trial=1000, random_seed=42)
        print(f"R@10: {round(r_at_10_mean*100, 3)} +/- {round(r_at_10_std*100, 3)}")

        r_at_20_mean, r_at_20_std = bootstrap_stats(r_20, num_of_trial=1000, random_seed=42)
        print(f"R@20: {round(r_at_20_mean*100, 3)} +/- {round(r_at_20_std*100, 3)}")\

        r_at_100_mean, r_at_100_std = bootstrap_stats(r_100, num_of_trial=1000, random_seed=42)
        print(f"R@100: {round(r_at_100_mean*100, 3)} +/- {round(r_at_100_std*100, 3)}")

        ndcg_mean, ndcg_std = bootstrap_stats(ndcg, num_of_trial=1000, random_seed=42)
        print(f"NDCG: {round(ndcg_mean*100, 3)} +/- {round(ndcg_std*100, 3)}")

        mrr_mean, mrr_std = bootstrap_stats(mrr, num_of_trial=1000, random_seed=42)
        print(f"mrr: {round(mrr_mean*100, 3)} +/- {round(mrr_std*100, 3)}")

        map_mean, map_std = bootstrap_stats(_map, num_of_trial=1000, random_seed=42)
        print(f"map: {round(map_mean*100, 3)} +/- {round(map_std*100, 3)}")

        p1_mean, p1_std = bootstrap_stats(p_1, num_of_trial=1000, random_seed=42)
        print(f"P@1: {round(p1_mean*100, 3)} +/- {round(p1_std*100, 3)}")

    else:
        print(f"R@5: {round(np.mean(recall_list)*100, 3)}")
        print(f"R@10: {round(np.mean(r_10)*100, 3)}")
        print(f"R@20: {round(np.mean(r_20)*100, 3)}")
        print(f"R@100: {round(np.mean(r_100)*100, 3)}")
        print(f"NDCG@10: {round(np.mean(ndcg)*100, 3)}")
        print(f"MRR@10: {round(np.mean(mrr)*100, 3)}")
        print(f"MAP: {round(np.mean(_map)*100, 3)}")
        print(f"P@1: {round(np.mean(p_1)*100, 3)}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset_path", "--dataset_path", type = str, required = True, help = "Name of the test dataset")
    parser.add_argument("--dataset_name", "--dataset_name", type = str, required = True, help = "Name of the test dataset")
    parser.add_argument("--experiment_name", "--experiment_name", type = str, required = True, help = "Name of the experiment")
    parser.add_argument("--boot", "--boot", type = str, required = True, help = "Bootstrap option")
    parser.add_argument("--rank_path", "--rank_path", type = str, required = True, help = "Path to the rank")
    parser.add_argument("--metric_by_key_result_path", "--metric_by_key_result_path", type = str, required = True, help = "Path to store the metric by key result")

    args = parser.parse_args()
    dataset_path = args.dataset_path
    dataset_name = args.dataset_name
    rank_path = args.rank_path
    experiment_name = args.experiment_name
    boot = args.boot

    # with open(dataset_path, 'rb') as f:
    #     dataset = pickle.load(f)
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    corpus = dataset['corpus']
    query = dataset['query']
    qrel = dataset['qrel']

    recall_list, r_10, r_20, r_100, ndcg, mrr, _map, p_1 = main(corpus, query, qrel, dataset_name, rank_path = rank_path)

    if boot.lower() == 'true':

        # rank_name = rank_path.split("/")[-1].split(".")[0]

        # with open(f"./metric_results/{dataset_name}_{rank_name}_r_5.pickle", 'wb') as f:
        #     pickle.dump(recall_list, f)

        # with open(f"./metric_results/{dataset_name}_{rank_name}_ndcg_10.pickle", 'wb') as f:
        #     pickle.dump(ndcg, f)
        

        print("The bootstrap default number is 1000...")
    
        r_at_5_mean, r_at_5_std = bootstrap_stats(recall_list, num_of_trial=1000, random_seed=42)
        print(f"R@5: {round(r_at_5_mean*100, 3)} +/- {round(r_at_5_std*100, 3)}")

        r_at_10_mean, r_at_10_std = bootstrap_stats(r_10, num_of_trial=1000, random_seed=42)
        print(f"R@10: {round(r_at_10_mean*100, 3)} +/- {round(r_at_10_std*100, 3)}")

        r_at_20_mean, r_at_20_std = bootstrap_stats(r_20, num_of_trial=1000, random_seed=42)
        print(f"R@20: {round(r_at_20_mean*100, 3)} +/- {round(r_at_20_std*100, 3)}")\

        r_at_100_mean, r_at_100_std = bootstrap_stats(r_100, num_of_trial=1000, random_seed=42)
        print(f"R@100: {round(r_at_100_mean*100, 3)} +/- {round(r_at_100_std*100, 3)}")

        ndcg_mean, ndcg_std = bootstrap_stats(ndcg, num_of_trial=1000, random_seed=42)
        print(f"NDCG: {round(ndcg_mean*100, 3)} +/- {round(ndcg_std*100, 3)}")

        mrr_mean, mrr_std = bootstrap_stats(mrr, num_of_trial=1000, random_seed=42)
        print(f"mrr: {round(mrr_mean*100, 3)} +/- {round(mrr_std*100, 3)}")

        map_mean, map_std = bootstrap_stats(_map, num_of_trial=1000, random_seed=42)
        print(f"map: {round(map_mean*100, 3)} +/- {round(map_std*100, 3)}")

        p1_mean, p1_std = bootstrap_stats(p_1, num_of_trial=1000, random_seed=42)
        print(f"P@1: {round(p1_mean*100, 3)} +/- {round(p1_std*100, 3)}")
        
        curr_time = get_west_coast_time()
        new_row = {'Exp_Name': f"{dataset_name}_{rank_path}", 'Time': curr_time, 'NDCG_mean': ndcg_mean, 'NDCG_STD': ndcg_std, 'R@5_mean': r_at_5_mean, 'R@5_std': r_at_5_std}
        
        add_line_to_csv("./full_log_table_new.csv", new_row)

    else:
        print(f"R@5: {round(np.mean(recall_list)*100, 3)}")
        print(f"R@10: {round(np.mean(r_10)*100, 3)}")
        print(f"R@20: {round(np.mean(r_20)*100, 3)}")
        print(f"R@100: {round(np.mean(r_100)*100, 3)}")
        print(f"NDCG@10: {round(np.mean(ndcg)*100, 3)}")
        print(f"MRR@10: {round(np.mean(mrr)*100, 3)}")
        print(f"MAP: {round(np.mean(_map)*100, 3)}")
        print(f"P@1: {round(np.mean(p_1)*100, 3)}")

        # pdb.set_trace()

        curr_time = get_west_coast_time()

        new_row = {'Exp_Name': experiment_name, 'Time': curr_time, 'NDCG_mean': np.mean(ndcg)*100, 'R@5_mean': np.mean(recall_list)*100, 'R@10_mean': np.mean(r_10)*100, 'R@20_mean': np.mean(r_20)*100, 'R@100_mean': np.mean(r_100)*100, 'MRR_mean': np.mean(mrr)*100, 'MAP_mean': np.mean(_map)*100, 'P@1_mean': np.mean(p_1)*100}

        metric_by_key_result_path = args.metric_by_key_result_path
        metric_by_key_result = {}
        for idx, qid in enumerate(qrel):
            metric_by_key_result[qid] = {'NDCG': ndcg[idx], 'R@5': recall_list[idx], 'R@10': r_10[idx], 'R@20': r_20[idx], 'R@100': r_100[idx], 'MRR': mrr[idx], 'MAP': _map[idx], 'P@1': p_1[idx]}

        with open(metric_by_key_result_path, 'wb') as f:
            pickle.dump(metric_by_key_result, f)
        add_line_to_csv("./full_log_table_new.csv", new_row)
        