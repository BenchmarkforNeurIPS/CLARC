import voyageai
import numpy
from tqdm import tqdm
import numpy as np
from transformers import BatchEncoding, AutoTokenizer, MPNetTokenizerFast, AutoModel, T5EncoderModel
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  RobertaConfig, RobertaModel, RobertaTokenizer)
import torch
import torch.utils.checkpoint
from torch import nn
from typing import Optional
import torch.nn.functional as F
import pdb
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import sys
from typing import Dict, List, Union, Optional
from sentence_transformers import SentenceTransformer
from openai import OpenAI, APIError # Import APIError for specific handling


def split_dict_into_batches(original_dict, batch_size):
    """
    Splits a dictionary into smaller sub-dictionaries with a specified batch size.

    Args:
        original_dict (dict): The input dictionary to split.
        batch_size (int): The maximum number of items each sub-dictionary can contain.

    Returns:
        list[dict]: A list of sub-dictionaries, each with length <= batch_size.
    """
    if not isinstance(original_dict, dict):
        raise TypeError("The input must be a dictionary.")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    
    keys = list(original_dict.keys())
    sub_dicts = []
    
    for i in range(0, len(keys), batch_size):
        batch_keys = keys[i:i + batch_size]
        sub_dict = {key: original_dict[key] for key in batch_keys}
        sub_dicts.append(sub_dict)
    
    return sub_dicts

def emb_with_voyage_code_3(text_dict, max_tokens=2048, batch_size=64, mode='query', loading_path=None):
    print("Voyage AI embedding with code-3 model use the same encoder for both query and corpus")
    vo = voyageai.Client(api_key="replace_with_your_api_key")
    
    batch_dicts = split_dict_into_batches(text_dict, batch_size)
    results = {}
    total_tokens = 0
    text2embedding = {}
    for batch_dict in tqdm(batch_dicts):
        curr_text_list = list(batch_dict.values())
        try:
            curr_output = vo.embed(curr_text_list, model='voyage-code-3', output_dimension=1024)
            curr_output_embedding = curr_output.embeddings
            total_tokens += curr_output.total_tokens
        except Exception as e:
            print("Exceed Token Limit, attempt to embed one by one")
            curr_output_embedding = []
            for text in curr_text_list:
                single_output = vo.embed([text], model='voyage-code-3', output_dimension=1024)
                curr_output_embedding.append(single_output.embeddings[0])
                total_tokens += single_output.total_tokens
        for idx, text in enumerate(curr_text_list):
            text2embedding[text] = np.array(curr_output_embedding[idx])

    for k, v in text_dict.items():
        results[k] = text2embedding[v]
    
    print('='*50)
    print()
    cost = total_tokens/1000000 * 0.18
    print(f"Total cost: ${cost:.2f}")
    print()
    print('='*50)

    return results
   

class TextDataset(Dataset):
    def __init__(self, text_list):
        self.text_list = text_list

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, idx):
        return self.text_list[idx]


def emb_with_codet5_plus(text_dict, max_tokens=2048, batch_size=64, mode='query', loading_path=None):
    device = torch.device('cuda:0')
    print("CodeT5+ use the same encoder for both query and corpus")
    if loading_path is None:
        print('='*50)
        print("Loading the model from Hugging Face")
        print('='*50)
        model_name = "Salesforce/codet5p-110m-embedding"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    else:
        print('='*50)
        print(f"Loading the model from {loading_path}")
        print('='*50)
        sys.path.append("../train/")
        from training_with_pytorch_lightning import ModelModule
        pl_model = ModelModule.load_from_checkpoint(loading_path)

        model = pl_model.model.to(device)
        tokenizer = pl_model.tokenizer

        # tokenizer = AutoTokenizer.from_pretrained(loading_path, trust_remote_code=True)
        # model = AutoModel.from_pretrained(loading_path, trust_remote_code=True).to(device)
        
    model.eval()

    dataset = TextDataset(list(text_dict.values()))
    dataloader = DataLoader(dataset, batch_size=batch_size)

    embeds = []

    for batch in tqdm(dataloader, total=len(dataloader)):
        inputs = tokenizer(batch, padding='max_length', truncation=True, max_length=max_tokens, return_tensors="pt").to(device)
        outputs = model(inputs.input_ids, attention_mask=inputs.attention_mask).cpu().detach()
        embeds.append(outputs)

    text_embeddings = torch.cat(embeds, dim=0)
    results = {}
    for i, k in enumerate(text_dict.keys()):
        results[k] = text_embeddings[i].numpy()

    return results

def emb_with_oasis( # Renamed slightly again for clarity
    text_dict: Dict[str, str],
    max_tokens: int = 2048, # Reinstated from previous signature
    batch_size: int = 64,
    mode: str = 'query', # Reinstated from previous signature
    loading_path: Optional[str] = None # Reinstated - IMPORTANT: Meaning changes here!
) -> Dict[str, np.ndarray]:
    """
    Generates embeddings for a dictionary of texts using the OASIS model
    via the sentence-transformers library, maintaining signature compatibility
    with the previous transformers-based functions.

    This method uses the default pooling strategy configured for the model
    in sentence-transformers (usually mean pooling).

    Args:
        text_dict: A dictionary where keys are identifiers and values are the text strings
                   (e.g., code snippets or queries).
        max_tokens: The maximum number of tokens to use. This will attempt to set
                    the max_seq_length of the loaded SentenceTransformer model.
        batch_size: The number of texts to process in parallel during encoding.
        mode: Included for signature compatibility. Typically ignored by sentence-transformers
              for symmetric embedding models like OASIS ('query' vs 'corpus').
        loading_path: If provided, this path is used to load the SentenceTransformer model.
                      **IMPORTANT**: This must be a Hugging Face model identifier OR a path
                      to a directory containing a model saved using the
                      `SentenceTransformer.save()` method, NOT a PyTorch Lightning checkpoint path.
                      If None, uses the default 'Kwaipilot/OASIS-code-embedding-1.5B'.

    Returns:
        A dictionary mapping the original keys from text_dict to their
        corresponding **unnormalized** embeddings as NumPy arrays (identical output format).
    """
    # --- Device Setup ---
    if torch.cuda.is_available():
        device = 'cuda' # sentence-transformers typically uses string identifiers
        print("Using GPU")
    else:
        device = 'cpu'
        print("Using CPU")

    # --- Determine Model Identifier ---
    model_identifier = loading_path if loading_path is not None else "Kwaipilot/OASIS-code-embedding-1.5B"

    # --- Model Loading ---
    print('='*50)
    print(f"Loading model '{model_identifier}' using SentenceTransformer library")
    if loading_path:
        print("NOTE: Using provided 'loading_path'. Ensure it's a SentenceTransformer-compatible path/ID.")
    print('='*50)
    try:
        model = SentenceTransformer(model_name_or_path=model_identifier, device=device)

        # --- Apply max_tokens ---
        # Attempt to set the max sequence length AFTER loading the model
        original_max_tokens = model.max_seq_length
        if max_tokens != original_max_tokens:
             try:
                 model.max_seq_length = max_tokens
                 print(f"Set model max_seq_length to: {model.max_seq_length} (original was {original_max_tokens})")
             except Exception as e:
                 print(f"Warning: Could not set max_seq_length on model. Using default {original_max_tokens}. Error: {e}")
        else:
            print(f"Model max_seq_length already matches requested max_tokens: {original_max_tokens}")


    except Exception as e:
        print(f"Error loading SentenceTransformer model: {e}")
        print("Please ensure the path/identifier is correct and points to a SentenceTransformer compatible model.")
        return {}

    # --- Data Preparation ---
    keys = list(text_dict.keys())
    texts_to_embed = list(text_dict.values())

    # --- Embedding Generation ---
    # Note: 'mode' parameter is not used by model.encode here.
    print(f"Generating embeddings with batch size {batch_size} using SentenceTransformer...")
    print(f"(Parameter 'mode={mode}' is present for compatibility but generally unused by ST encode)")

    try:
        embeddings_np = model.encode(
            texts_to_embed,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
    except Exception as e:
        print(f"Error during SentenceTransformer encoding: {e}")
        return {}

    # --- Post-processing ---
    if embeddings_np is None or embeddings_np.shape[0] != len(keys):
        print(f"Warning: Embedding generation failed or produced unexpected number of results. Expected {len(keys)}, Got {embeddings_np.shape[0] if embeddings_np is not None else 'None'}")
        return {}

    # Create the result dictionary (identical format)
    results = {key: emb/np.linalg.norm(emb) for key, emb in zip(keys, embeddings_np)}

    print("Embedding generation complete.")
    return results

def emb_with_nomic_code_compat( # Renamed function
    text_dict: Dict[str, str],
    max_tokens: int = 2048, # Default matches nomic-embed-code max length
    batch_size: int = 64,
    mode: str = 'query', # Keep for signature compatibility
    loading_path: Optional[str] = None # Keep - IMPORTANT: ST-compatible path/ID
) -> Dict[str, np.ndarray]:
    """
    Generates embeddings for a dictionary of texts using the
    **nomic-ai/nomic-embed-code** model via the sentence-transformers library,
    maintaining signature compatibility with the previous functions.

    This method uses the default pooling strategy configured for the model
    in sentence-transformers (usually mean pooling).

    Args:
        text_dict: A dictionary where keys are identifiers and values are the text strings
                   (e.g., code snippets or queries).
        max_tokens: The maximum number of tokens to use. This will attempt to set
                    the max_seq_length of the loaded SentenceTransformer model.
                    Default 2048 matches nomic-embed-code's context length.
        batch_size: The number of texts to process in parallel during encoding.
        mode: Included for signature compatibility. Typically ignored by sentence-transformers
              for symmetric embedding models like nomic-embed-code ('query' vs 'corpus').
        loading_path: If provided, this path is used to load the SentenceTransformer model.
                      **IMPORTANT**: This must be a Hugging Face model identifier OR a path
                      to a directory containing a model saved using the
                      `SentenceTransformer.save()` method, NOT a PyTorch Lightning checkpoint path.
                      If None, uses the default 'nomic-ai/nomic-embed-code'.

    Returns:
        A dictionary mapping the original keys from text_dict to their
        corresponding **unnormalized** embeddings as NumPy arrays (identical output format).
    """
    # --- Device Setup ---
    if torch.cuda.is_available():
        device = 'cuda'
        print("Using GPU")
    else:
        device = 'cpu'
        print("Using CPU")

    # --- Determine Model Identifier ---
    default_model_name = "nomic-ai/nomic-embed-code" # <<< Use Nomic model here
    model_identifier = loading_path if loading_path is not None else default_model_name

    # --- Model Loading ---
    print('='*50)
    print(f"Loading model '{model_identifier}' using SentenceTransformer library")
    if loading_path:
        print("NOTE: Using provided 'loading_path'. Ensure it's a SentenceTransformer-compatible path/ID.")
    print('='*50)
    try:
        # Add trust_remote_code=True as a precaution for custom model code
        model = SentenceTransformer(
            model_name_or_path=model_identifier,
            device=device,
            trust_remote_code=True
        )

        # --- Apply max_tokens ---
        # Attempt to set the max sequence length AFTER loading the model
        original_max_tokens = model.max_seq_length
        # Check if the requested max_tokens exceeds the model's known limit (if available and different)
        # Nomic Code's limit is 2048. Let's respect that if user tries to go higher,
        # unless they are loading a custom fine-tuned version via loading_path.
        effective_max_tokens = max_tokens
        if loading_path is None and default_model_name == "nomic-ai/nomic-embed-code":
            if max_tokens > 2048:
                print(f"Warning: Requested max_tokens ({max_tokens}) exceeds nomic-embed-code's limit (2048). Clamping to 2048.")
                effective_max_tokens = 2048
            elif max_tokens <= 0:
                 print(f"Warning: Invalid max_tokens ({max_tokens}). Using model default: {original_max_tokens}")
                 effective_max_tokens = original_max_tokens

        if effective_max_tokens != original_max_tokens:
             try:
                 model.max_seq_length = effective_max_tokens
                 print(f"Set model max_seq_length to: {model.max_seq_length} (original was {original_max_tokens})")
             except Exception as e:
                 print(f"Warning: Could not set max_seq_length on model. Using default {original_max_tokens}. Error: {e}")
        else:
             # Only print if it wasn't adjusted or already matched
             if effective_max_tokens == max_tokens:
                 print(f"Model max_seq_length matches requested max_tokens: {effective_max_tokens}")


    except Exception as e:
        print(f"Error loading SentenceTransformer model ('{model_identifier}'): {e}")
        print("Please ensure the path/identifier is correct, the model is compatible with SentenceTransformer,")
        print("and necessary dependencies (like sentence-transformers) are installed.")
        return {}

    # --- Data Preparation ---
    keys = list(text_dict.keys())
    texts_to_embed = list(text_dict.values())

    # --- Embedding Generation ---
    # Note: 'mode' parameter is not used by model.encode here.
    print(f"Generating embeddings with batch size {batch_size} using SentenceTransformer (Nomic Embed Code)...")
    print(f"(Parameter 'mode={mode}' is present for compatibility but generally unused by ST encode)")

    try:
        embeddings_np = model.encode(
            texts_to_embed,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=False, # Ensure output is unnormalized like previous functions
        )
    except Exception as e:
        print(f"Error during SentenceTransformer encoding: {e}")
        return {}

    # --- Post-processing ---
    if embeddings_np is None or not isinstance(embeddings_np, np.ndarray) or embeddings_np.shape[0] != len(keys):
        print(f"Warning: Embedding generation failed or produced unexpected number/type of results. Expected {len(keys)} numpy arrays, Got type {type(embeddings_np)} with shape {embeddings_np.shape if isinstance(embeddings_np, np.ndarray) else 'N/A'}")
        return {}

    # Create the result dictionary (identical format)
    results = {key: emb/np.linalg.norm(emb) for key, emb in zip(keys, embeddings_np)}

    print("Embedding generation complete.")
    return results


OPENAI_EMBEDDING_PRICES = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}


def embed_dictionary_with_openai(
    text_dict: Dict[str, str],
    max_tokens: int = 2048,         # Included in signature, but NOT directly used by OpenAI logic
    batch_size: int = 64,           # USED by the function
    mode: str = 'query',            # Included in signature, but NOT used by OpenAI logic
    loading_path: Optional[str] = None # Included in signature, but NOT used by OpenAI logic
) -> Dict[str, np.ndarray]:
    """
    Generates embeddings using OpenAI for text values in a dictionary,
    but adheres to a specific fixed function signature.

    NOTE: This function uses a hardcoded OpenAI model and requires the API key
          to be set via the OPENAI_API_KEY environment variable.
          The 'max_tokens', 'mode', and 'loading_path' arguments are part of the
          required signature but are ignored by the internal OpenAI logic.

    Args:
        text_dict: A dictionary where keys are identifiers and values are text strings.
        max_tokens (int): Ignored by OpenAI logic. Included for signature compatibility.
                          OpenAI models have their own internal token limits.
        batch_size (int): The number of texts to process in each API call batch. (Used)
        mode (str): Ignored by OpenAI logic. Included for signature compatibility.
        loading_path (str, optional): Ignored by OpenAI logic. Included for signature compatibility.

    Returns:
        A dictionary with the same keys as text_dict, but values are numpy arrays
        representing the embeddings.

    Raises:
        ValueError: If the OPENAI_API_KEY environment variable is not set,
                    or if batch_size is invalid.
        TypeError: If text_dict is not a dictionary.
        ImportError: If the 'openai' library is not installed.
    """
    OPENAI_MODEL_TO_USE = "text-embedding-3-large" # HARDCODED: Choose the desired model
    OPENAI_DIMENSIONS = None # Set desired dimensions (e.g., 1024) or None for default
                             # Only applicable for models like text-embedding-3-*

    
    print(f"--- Starting OpenAI Embedding ---")
    print(f"    Using hardcoded OpenAI Model: {OPENAI_MODEL_TO_USE}")
    if OPENAI_DIMENSIONS:
        print(f"    Using hardcoded Dimensions: {OPENAI_DIMENSIONS}")
    print(f"    NOTE: 'max_tokens', 'mode', 'loading_path' args are ignored.")
    print(f"    Using 'batch_size': {batch_size}")


    # 1. Get API Key from Environment
    api_key = "replace_with_your_api_key"
    if api_key is None:
        raise ValueError("OpenAI API key not provided. Set the OPENAI_API_KEY environment variable.")

    # 2. Initialize OpenAI Client
    try:
        client = OpenAI(api_key=api_key)
    except ImportError:
        raise ImportError("OpenAI library not installed. Please install using: pip install openai")

    # 3. Prepare batches using split_dict_into_batches
    try:
        # Use the batch_size argument passed to the function
        batch_dicts = split_dict_into_batches(text_dict, batch_size)
    except (TypeError, ValueError) as e:
        raise e from None

    results = {}
    total_tokens = 0

    print(f"Total texts: {len(text_dict)}, Batch size: {batch_size}, Batches: {len(batch_dicts)}")

    # 4. Process each batch dictionary
    for i, batch_dict in enumerate(tqdm(batch_dicts, desc="Embedding Batches")):
        if not batch_dict:
            continue

        current_keys = list(batch_dict.keys())
        text_batch = list(batch_dict.values())
        batch_embeddings = {}

        # Prepare API call parameters using hardcoded model/dimensions
        api_params = {
            "input": text_batch,
            "model": OPENAI_MODEL_TO_USE, # Use hardcoded model
        }
        if OPENAI_DIMENSIONS is not None and OPENAI_MODEL_TO_USE.startswith("text-embedding-3"):
             api_params["dimensions"] = OPENAI_DIMENSIONS # Use hardcoded dimensions

        try:
            # 5. Make the batch API call
            response = client.embeddings.create(**api_params)
            batch_embeddings_list = [item.embedding for item in response.data]
            total_tokens += response.usage.total_tokens

            # Map embeddings back to keys
            if len(current_keys) != len(batch_embeddings_list):
                 print(f"\nError: Mismatch between keys ({len(current_keys)}) and embeddings ({len(batch_embeddings_list)}) in batch {i+1}. Skipping batch.")
                 for key in current_keys: batch_embeddings[key] = None
            else:
                for k_idx, key in enumerate(current_keys):
                     batch_embeddings[key] = np.array(batch_embeddings_list[k_idx])

        except APIError as e:
            print(f"\nWarning: Batch {i+1} failed: {e}. Attempting one by one.")
            # 6. Fallback
            for single_key, text in batch_dict.items():
                try:
                    single_api_params = { "input": [text], "model": OPENAI_MODEL_TO_USE }
                    if OPENAI_DIMENSIONS is not None and OPENAI_MODEL_TO_USE.startswith("text-embedding-3"):
                        single_api_params["dimensions"] = OPENAI_DIMENSIONS

                    single_response = client.embeddings.create(**single_api_params)
                    batch_embeddings[single_key] = np.array(single_response.data[0].embedding)
                    total_tokens += single_response.usage.total_tokens
                except Exception as single_e:
                    print(f"\nError: Failed to embed key '{single_key}' individually: {single_e}")
                    batch_embeddings[single_key] = None

        except Exception as e:
             print(f"\nError: Unexpected error processing batch {i+1}: {e}")
             for key in current_keys: batch_embeddings[key] = None

        results.update(batch_embeddings)

    print("\n--- Embedding Complete ---")

    # 7. Calculate and Print Cost
    cost = 0
    # Use the hardcoded model name for price lookup
    price_per_million = OPENAI_EMBEDDING_PRICES.get(OPENAI_MODEL_TO_USE)
    if price_per_million is not None:
        cost = (total_tokens / 1_000_000) * price_per_million
        print(f"Total tokens processed: {total_tokens}")
        print(f"Estimated cost for {OPENAI_MODEL_TO_USE}: ${cost:.6f}")
    else:
        print(f"Total tokens processed: {total_tokens}")
        print(f"Warning: Cost calculation unavailable for model '{OPENAI_MODEL_TO_USE}'.")

    print('='*50)

    final_results = {k: v for k, v in results.items() if v is not None}
    failed_count = len(results) - len(final_results)
    if failed_count > 0:
        print(f"Warning: {failed_count} items could not be embedded.")

    return final_results

