# CLARC: C/C++ Benchmark for Robust Code Search

**This repository provides the codebase for the research paper titled "CLARC: C/C++ Benchmark for Robust Code Search". It includes the CLARC dataset and scripts for evaluating the models discussed in the paper.**


The capability to retrieve code snippets effectively through natural language descriptions is vital for enhancing code reuse and boosting productivity in software engineering. Current evaluation datasets for this task predominantly focus on Python, the most common programming language for research, leaving a gap in benchmarks for industry-focused languages like C or C++. Furthermore, existing datasets often lack structured categorization and can rely heavily on superficial features, such as the direct overlap between code identifiers and query keywords. To address these limitations, we introduce CLARC (C/C++ LAnguage Retrieval with Anonymized Code), a novel benchmark for C/C++ code search that is fully compilable, configurable, and extensible. CLARC consists of code snippets and corresponding queries organized into three distinct groups based on their dependencies. We also provide various settings for CLARC that test model robustness by removing superficial code features through anonymization. In addition, an automated pipeline is designed to expand the benchmark while preserving data quality.

## Embedding Generation and Reranking

The script `embedding/dataset2rank_emb.py` is used to calculate embeddings for queries and code snippets. These embeddings are then utilized to rerank candidate pools within the dataset.

Execute the script using the following command structure:

```shell
python3 dataset2rank_emb.py --dataset_path <dataset_path> \
                           --emb_store_path <emb_output_path> \
                           --rank_store_path <rank_output_path> \
                           --model <model_name> \
                           --max_tokens <context_window_length> \
                           --batch_size <batch_size>
```

**Arguments:**

* `--dataset_path <dataset_path>`: Specifies the path to the input dataset files (e.g., files located in the `cleaned/` directory).
* `--emb_store_path <emb_output_path>`: Defines the path where the generated embeddings will be stored (as a pickle file).
* `--rank_store_path <rank_output_path>`: Defines the path where the reranking results will be stored (as a pickle file).
* `--model <model_name>`: Specifies the embedding model. Supported models include: `voyage_ai`, `code_t5p`, `oasis`, `nomic`, `openai`, and `bm25`.
* `--max_tokens <context_window_length>`: Defines the context window length for the embedding model. Typically `512`. For `asm` and `wasm` inputs, `8192` is commonly used.
* `--batch_size <batch_size>`: Sets the batch size for the embedding process.

**Example:**

```shell
python3 dataset2rank_emb.py --dataset_path ../cleaned/reconstructed_group1_original_cleaned.json \
                            --emb_store_path ../embedding_results/group1/original__voyage_ai.pickle \
                            --rank_store_path ../ranks/group1/original__voyage_ai.pickle \
                            --model voyage_ai \
                            --max_tokens 512 \
                            --batch_size 64
```

**Important:** Please ensure that the output directories (e.g., `../embedding_results/group1/` and `../ranks/group1/` in the example) are created before running this command.

## Evaluation

After generating embeddings and reranking the candidate pools, use the following command to calculate retrieval metrics based on the reranking results:

```shell
python3 rank2metric.py --dataset_path <dataset_path> \
                      --dataset_name <dataset_name> \
                      --experiment_name <exp_name> \
                      --boot <boot_option> \
                      --rank_path <rank_path> \
                      --metric_by_key_result_path <detailed_results_path>
```

**Arguments:**

* `--dataset_path <dataset_path>`: Path to the dataset file (this should correspond to the one used in the embedding step, e.g., `../cleaned/reconstructed_group1_original_cleaned.json`).
* `--dataset_name <dataset_name>`: A string to identify the dataset being evaluated (e.g., "Group1_Original").
* `--experiment_name <exp_name>`: A string to name this specific evaluation run (e.g., "Group1_Original_Voyage").
* `--boot <boot_option>`: Boolean flag (`True` or `False`) to indicate if bootstrapping is required for the evaluation. This is typically set to `False`.
* `--rank_path <rank_path>`: Path to the reranking results file (pickle file) generated in the previous step.
* `--metric_by_key_result_path <detailed_results_path>`: Path to store a file (pickle file) containing detailed evaluation results for each query.

**Example:**

```shell
python3 rank2metric.py --dataset_path ../cleaned/reconstructed_group1_original_cleaned.json \
                       --dataset_name Group1_Original \
                       --experiment_name Group1_Original_Voyage \
                       --boot False \
                       --rank_path ../ranks/group1/original__voyage_ai.pickle \
                       --metric_by_key_result_path ./detailed_results/group1/original__voyage_ai.pickle
```

**Important:** Ensure that the necessary output directory (e.g., `./detailed_results/group1/` in the example) is created before executing this command.
