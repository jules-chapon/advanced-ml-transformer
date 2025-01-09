"""Functions for preprocessing"""

from datasets import load_dataset
import pandas as pd
import regex as re
import time
import unicodedata

from typing import Any

from src.configs import constants, names


def load_data_from_hf(small: bool = True) -> pd.DataFrame:
    start_time = time.time()
    if small:
        folder = constants.HF_SMALL_FILENAME
    else:
        folder = constants.HF_LARGE_FILENAME
    df = load_dataset(folder)["train"].to_pandas()
    print(f"Data loading done in {time.time() - start_time:.2f} seconds")
    return df


def load_data_from_local(small: bool = True) -> pd.DataFrame:
    start_time = time.time()
    if small:
        df = pd.read_csv(constants.DATA_SMALL_FILENAME, index_col=False)
    else:
        df = pd.read_csv(constants.DATA_LARGE_FILENAME, index_col=False)
    print(f"Data loading done in {time.time() - start_time:.2f} seconds")
    return df


def load_data(local: bool = True, small: bool = True) -> pd.DataFrame:
    if local:
        print("Loading data from local")
        return load_data_from_local(small=small)
    else:
        print("Loading data from Hugging Face")
        return load_data_from_hf(small=small)


def get_train_valid_test_sets(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(df) > constants.MAX_LEN_DF:
        df = df.sample(
            n=constants.MAX_LEN_DF, random_state=constants.RANDOM_SEED
        ).reset_index(drop=True)
    else:
        df = df.sample(frac=1, random_state=constants.RANDOM_SEED).reset_index(
            drop=True
        )
    nb_sentences = len(df)
    nb_train = int(constants.TRAIN_RATIO * nb_sentences)
    nb_val = int(constants.VALID_RATIO * nb_sentences)
    df_train = df[:nb_train]
    df_valid = df[nb_train : nb_train + nb_val]
    df_test = df[nb_train + nb_val :]
    print("Train valid and test sets are ready")
    return df_train, df_valid, df_test


def clean_text(text: str) -> str:
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


def from_text_to_tokens(
    sentence: str, params: dict[str, Any] | None = None
) -> list[str]:
    if isinstance(sentence, str):
        if params[names.TOKENIZATION] == names.BASIC:
            tokens = sentence.split()
        elif params[names.TOKENIZATION] == names.ADVANCED:
            tokens = sentence.split()
    else:
        tokens = [" "]
    return tokens


def from_tokens_to_text(tokens: list[str], params: dict[str, Any] | None = None) -> str:
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


def tokenize_sentence_src(
    sentence: str, vocab: dict[str, int], params: dict[str, Any] | None = None
) -> list[int]:
    sentence = clean_text(text=sentence)
    tokens = from_text_to_tokens(sentence=sentence, params=params)
    if len(tokens) >= params[names.MAX_SEQUENCE_LENGTH]:
        tokens = tokens[: params[names.MAX_SEQUENCE_LENGTH]]
    else:
        tokens += [constants.PAD_TOKEN] * (
            params[names.MAX_SEQUENCE_LENGTH] - len(tokens)
        )
    token_ids = []
    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
        token_ids.append(vocab[token])
    return token_ids


def tokenize_sentence_inference(
    sentence: str, vocab: dict[str, int], params: dict[str, Any] | None = None
) -> list[int]:
    sentence = clean_text(text=sentence)
    tokens = from_text_to_tokens(sentence=sentence, params=params)
    if len(tokens) >= params[names.MAX_SEQUENCE_LENGTH]:
        tokens = tokens[: params[names.MAX_SEQUENCE_LENGTH]]
    else:
        tokens += [constants.PAD_TOKEN] * (
            params[names.MAX_SEQUENCE_LENGTH] - len(tokens)
        )
    token_ids = []
    for token in tokens:
        if token not in vocab:
            token_ids.append(vocab[constants.PAD_TOKEN])
        else:
            token_ids.append(vocab[token])
    return token_ids


def tokenize_sentence_tgt(
    sentence: str, vocab: dict[str, int], params: dict[str, Any] | None = None
) -> list[int]:
    sentence = clean_text(text=sentence)
    tokens = [constants.BOS_TOKEN] + from_text_to_tokens(
        sentence=sentence, params=params
    )
    if len(tokens) >= params[names.MAX_SEQUENCE_LENGTH] + 1:
        tokens = tokens[: params[names.MAX_SEQUENCE_LENGTH] + 1]
    else:
        tokens += [constants.EOS_TOKEN] + [constants.PAD_TOKEN] * (
            params[names.MAX_SEQUENCE_LENGTH] - len(tokens)
        )
    input_token_ids = []
    for i in range(len(tokens)):
        token = tokens[i]
        if token not in vocab:
            vocab[token] = len(vocab)
        input_token_ids.append(vocab[token])
    return input_token_ids[:-1], input_token_ids[1:]


def tokenize_dataframe(
    df: pd.DataFrame,
    src_vocab: dict[str, int],
    tgt_vocab: dict[str, int],
    src_input_tokens: list[list[int]],
    tgt_input_tokens: list[list[int]],
    tgt_output_tokens: list[int],
    params: dict[str, Any] | None = None,
) -> tuple[dict[str, int], dict[str, int], list[list[int]], list[list[int]]]:
    for src_sent, tgt_sent in zip(
        df[params[names.SRC_LANGUAGE]], df[params[names.TGT_LANGUAGE]]
    ):
        src_input = tokenize_sentence_src(
            sentence=src_sent, vocab=src_vocab, params=params
        )
        tgt_input, tgt_output = tokenize_sentence_tgt(
            sentence=tgt_sent, vocab=tgt_vocab, params=params
        )
        src_input_tokens.append(src_input)
        tgt_input_tokens.append(tgt_input)
        tgt_output_tokens.append(tgt_output)
    return src_vocab, tgt_vocab, src_input_tokens, tgt_input_tokens, tgt_output_tokens


def tokenize_datasets(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> None:
    src_vocab = constants.DEFAULT_VOCAB.copy()  # Initialize source vocabulary
    tgt_vocab = constants.DEFAULT_VOCAB.copy()  # Initialize target vocabulary
    (
        src_vocab,
        tgt_vocab,
        src_input_tokens_train,
        tgt_input_tokens_train,
        tgt_output_tokens_train,
    ) = tokenize_dataframe(
        df=df_train,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        src_input_tokens=[],
        tgt_input_tokens=[],
        tgt_output_tokens=[],
        params=params,
    )
    (
        src_vocab,
        tgt_vocab,
        src_input_tokens_valid,
        tgt_input_tokens_valid,
        tgt_output_tokens_valid,
    ) = tokenize_dataframe(
        df=df_valid,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        src_input_tokens=[],
        tgt_input_tokens=[],
        tgt_output_tokens=[],
        params=params,
    )
    (
        src_vocab,
        tgt_vocab,
        src_input_tokens_test,
        tgt_input_tokens_test,
        tgt_output_tokens_test,
    ) = tokenize_dataframe(
        df=df_test,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        src_input_tokens=[],
        tgt_input_tokens=[],
        tgt_output_tokens=[],
        params=params,
    )
    print("Datasets have been tokenized")
    return (
        src_vocab,
        tgt_vocab,
        src_input_tokens_train,
        tgt_input_tokens_train,
        tgt_output_tokens_train,
        src_input_tokens_valid,
        tgt_input_tokens_valid,
        tgt_output_tokens_valid,
        src_input_tokens_test,
        tgt_input_tokens_test,
        tgt_output_tokens_test,
    )
