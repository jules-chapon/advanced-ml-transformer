"""Functions for preprocessing."""

from datasets import load_dataset
import pandas as pd
import regex as re
import time
import unicodedata

from typing import Any, Dict, List, Tuple

from src.configs import constants, names


def load_data_from_hf(type: str = "samples") -> pd.DataFrame:
    """
    Load data from Hugging Face datasets.

    Args:
        type (str): Type of data to load. Can be "samples", "train", "valid", or "test".

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    start_time = time.time()
    if type == "samples":
        filename = constants.HF_SAMPLES_FILENAME
        df = load_dataset(filename)["train"].to_pandas()
    elif type == "train":
        filename = constants.HF_TRAIN_FILENAME
        df = (
            load_dataset(filename)["train"]
            .shuffle()
            .select(range(constants.NB_DATA_TRAIN))
            .to_pandas()
        )
    elif type == "valid":
        filename = constants.HF_VALID_FILENAME
        df = load_dataset(filename)["validation"].to_pandas()
    elif type == "test":
        filename = constants.HF_TEST_FILENAME
        df = load_dataset(filename)["test"].to_pandas()
    else:
        raise ValueError("Invalid type")
    print(f"Data loading done in {time.time() - start_time:.2f} seconds")
    return df


def load_data_from_local(type: str = "samples") -> pd.DataFrame:
    """
    Load data from local CSV files.

    Args:
        type (str): Type of data to load. Can be "samples", "train", "valid", or "test".

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    start_time = time.time()
    if type == "samples":
        df = pd.read_csv(constants.DATA_SAMPLES_FILENAME, index_col=False)
    elif type == "train":
        df = pd.read_csv(constants.DATA_TRAIN_FILENAME, index_col=False)
    elif type == "valid":
        df = pd.read_csv(constants.DATA_VALID_FILENAME, index_col=False)
    elif type == "test":
        df = pd.read_csv(constants.DATA_TEST_FILENAME, index_col=False)
    else:
        raise ValueError("Invalid type")
    print(f"Data loading done in {time.time() - start_time:.2f} seconds")
    return df


def load_data(local: bool = True, type: str = "samples") -> pd.DataFrame:
    """
    Load data from either local CSV files or Hugging Face datasets.

    Args:
        local (bool): Whether to load data from local CSV files.
        type (str): Type of data to load. Can be "samples", "train", "valid", or "test".

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    if local:
        print("Loading data from local")
        return load_data_from_local(type=type)
    else:
        print("Loading data from Hugging Face")
        return load_data_from_hf(type=type)


def clean_text(text: str | None) -> str:
    """
    Clean text by removing special characters, URLs, mentions, and converting to lowercase.

    Args:
        text (str | None): Input text.

    Returns:
        str: Cleaned text.
    """
    if text is None:
        return constants.PAD_TOKEN
    normalized_text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in normalized_text if not unicodedata.combining(char))
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"#\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = text.lower()
    return text


def from_text_to_tokens(sentence: str) -> List[str]:
    """
    Convert text to tokens based on the specified tokenization method.

    Args:
        sentence (str): Input text.

    Returns:
        List[str]: List of tokens.
    """
    if isinstance(sentence, str):
        tokens = sentence.split()
    else:
        tokens = [constants.PAD_TOKEN]
    return tokens


def from_tokens_to_text(tokens: List[str]) -> str:
    """
    Convert tokens to text by removing special tokens.

    Args:
        tokens (List[str]): List of tokens.

    Returns:
        str: Text without special tokens.
    """
    sentence = " ".join(
        token
        for token in tokens
        if token
        not in [
            constants.BOS_TOKEN,
            constants.EOS_TOKEN,
            constants.PAD_TOKEN,
        ]
    )
    return sentence


def create_vocabs(
    df: pd.DataFrame, params: Dict[str, Any] | None = None
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Create source and target vocabularies by tokenizing the text and counting the occurrences of each token.

    Args:
        df (pd.DataFrame): DataFrame containing the source and target sentences.
        params (Dict[str, Any] | None): Parameters for tokenization.

    Returns:
        Tuple[Dict[str, int], Dict[str, int]]: Source vocabulary and target vocabulary.
    """
    src_vocab = constants.DEFAULT_VOCAB.copy()  # Initialize source vocabulary
    tgt_vocab = constants.DEFAULT_VOCAB.copy()  # Initialize target vocabulary
    for src_sent, tgt_sent in zip(
        df[params[names.SRC_LANGUAGE]], df[params[names.TGT_LANGUAGE]]
    ):
        src_sent_clean = clean_text(text=src_sent)
        tgt_sent_clean = clean_text(text=tgt_sent)
        src_tokens = from_text_to_tokens(sentence=src_sent_clean)
        tgt_tokens = from_text_to_tokens(sentence=tgt_sent_clean)
        n = len(src_tokens)
        m = len(tgt_tokens)
        for i in range(max(n, m)):
            if i >= n:
                src_token = constants.PAD_TOKEN
            else:
                src_token = src_tokens[i]
            if i >= m:
                tgt_token = constants.PAD_TOKEN
            else:
                tgt_token = tgt_tokens[i]
            if src_token not in src_vocab:
                src_vocab[src_token] = len(src_vocab)
            if tgt_token not in tgt_vocab:
                tgt_vocab[tgt_token] = len(tgt_vocab)
    return src_vocab, tgt_vocab


def tokenize_sentence(sentence: str, vocab: Dict[str, int]) -> List[int]:
    """
    Tokenize a sentence by converting each token to its corresponding token ID.

    Args:
        sentence (str): Input sentence.
        vocab (Dict[str, int]): Vocabulary containing token-to-ID mappings.

    Returns:
        List[int]: List of token IDs.
    """
    sentence_clean = clean_text(text=sentence)
    tokens = from_text_to_tokens(sentence=sentence_clean)
    token_ids = []
    for token in tokens:
        if token not in vocab:
            token_ids.append(constants.PAD_TOKEN_ID)
        else:
            token_ids.append(vocab[token])
    return token_ids


def tokenize_dataframe(
    df: pd.DataFrame,
    src_vocab: Dict[str, int],
    tgt_vocab: Dict[str, int],
    params: Dict[str, Any] | None = None,
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Tokenize a DataFrame by converting each sentence to a list of token IDs.

    Args:
        df (pd.DataFrame): DataFrame containing the source and target sentences.
        src_vocab (Dict[str, int]): Source vocabulary containing token-to-ID mappings.
        tgt_vocab (Dict[str, int]): Target vocabulary containing token-to-ID mappings.
        params (Dict[str, Any] | None): Parameters for tokenization.

    Returns:
        Tuple[List[List[int]], List[List[int]]]: List of source token IDs and list of target token IDs.
    """
    src_all_token_ids = []
    tgt_all_token_ids = []
    for src_sent, tgt_sent in zip(
        df[params[names.SRC_LANGUAGE]], df[params[names.TGT_LANGUAGE]]
    ):
        src_token_ids = tokenize_sentence(sentence=src_sent, vocab=src_vocab)
        src_all_token_ids.append(src_token_ids)
        tgt_token_ids = tokenize_sentence(sentence=tgt_sent, vocab=tgt_vocab)
        tgt_all_token_ids.append(tgt_token_ids)
    return src_all_token_ids, tgt_all_token_ids
