"""Functions for preprocessing"""

from datasets import load_dataset
import pandas as pd
import regex as re
import time
import unicodedata

from typing import Any

from src.configs import constants, names


def load_data_from_hf(type: str = "samples") -> pd.DataFrame:
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
    if local:
        print("Loading data from local")
        return load_data_from_local(type=type)
    else:
        print("Loading data from Hugging Face")
        return load_data_from_hf(type=type)


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


def create_vocabs(
    df: pd.DataFrame, params: dict[str, Any] | None = None
) -> tuple[dict[str, int], dict[str, int]]:
    src_vocab = constants.DEFAULT_VOCAB.copy()  # Initialize source vocabulary
    tgt_vocab = constants.DEFAULT_VOCAB.copy()  # Initialize target vocabulary
    for src_sent, tgt_sent in zip(
        df[params[names.SRC_LANGUAGE]], df[params[names.TGT_LANGUAGE]]
    ):
        src_sent_clean = clean_text(text=src_sent)
        tgt_sent_clean = clean_text(text=tgt_sent)
        src_tokens = from_text_to_tokens(sentence=src_sent_clean, params=params)
        tgt_tokens = from_text_to_tokens(sentence=tgt_sent_clean, params=params)
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


def tokenize_sentence(
    sentence: str, vocab: dict[str, int], params: dict[str, Any] | None = None
) -> list[int]:
    sentence_clean = clean_text(text=sentence)
    tokens = from_text_to_tokens(sentence=sentence_clean, params=params)
    token_ids = []
    for token in tokens:
        if token not in vocab:
            token_ids.append(constants.PAD_TOKEN_ID)
        else:
            token_ids.append(vocab[token])
    return token_ids


def tokenize_dataframe(
    df: pd.DataFrame,
    src_vocab: dict[str, int],
    tgt_vocab: dict[str, int],
    params: dict[str, Any] | None = None,
) -> tuple[list[int], list[int]]:
    src_all_token_ids = []
    tgt_all_token_ids = []
    for src_sent, tgt_sent in zip(
        df[params[names.SRC_LANGUAGE]], df[params[names.TGT_LANGUAGE]]
    ):
        src_token_ids = tokenize_sentence(
            sentence=src_sent, vocab=src_vocab, params=params
        )
        src_all_token_ids.append(src_token_ids)
        tgt_token_ids = tokenize_sentence(
            sentence=tgt_sent, vocab=tgt_vocab, params=params
        )
        tgt_all_token_ids.append(tgt_token_ids)
    return src_all_token_ids, tgt_all_token_ids


# def tokenize_sentence_src(
#     sentence: str, vocab: dict[str, int], params: dict[str, Any] | None = None
# ) -> list[int]:
#     sentence = clean_text(text=sentence)
#     tokens = from_text_to_tokens(sentence=sentence, params=params)
#     if len(tokens) >= params[names.MAX_SEQUENCE_LENGTH]:
#         tokens = tokens[: params[names.MAX_SEQUENCE_LENGTH]]
#     else:
#         tokens += [constants.PAD_TOKEN] * (
#             params[names.MAX_SEQUENCE_LENGTH] - len(tokens)
#         )
#     token_ids = []
#     for token in tokens:
#         if token not in vocab:
#             vocab[token] = len(vocab)
#         token_ids.append(vocab[token])
#     return token_ids


# def tokenize_sentence_inference(
#     sentence: str, vocab: dict[str, int], params: dict[str, Any] | None = None
# ) -> list[int]:
#     sentence = clean_text(text=sentence)
#     tokens = from_text_to_tokens(sentence=sentence, params=params)
#     if len(tokens) >= params[names.MAX_SEQUENCE_LENGTH]:
#         tokens = tokens[: params[names.MAX_SEQUENCE_LENGTH]]
#     else:
#         tokens += [constants.PAD_TOKEN] * (
#             params[names.MAX_SEQUENCE_LENGTH] - len(tokens)
#         )
#     token_ids = []
#     for token in tokens:
#         if token not in vocab:
#             token_ids.append(vocab[constants.PAD_TOKEN])
#         else:
#             token_ids.append(vocab[token])
#     return token_ids


# def tokenize_sentence_tgt(
#     sentence: str, vocab: dict[str, int], params: dict[str, Any] | None = None
# ) -> list[int]:
#     sentence = clean_text(text=sentence)
#     tokens = [constants.BOS_TOKEN] + from_text_to_tokens(
#         sentence=sentence, params=params
#     )
#     if len(tokens) >= params[names.MAX_SEQUENCE_LENGTH] + 1:
#         tokens = tokens[: params[names.MAX_SEQUENCE_LENGTH] + 1]
#     else:
#         tokens += [constants.EOS_TOKEN] + [constants.PAD_TOKEN] * (
#             params[names.MAX_SEQUENCE_LENGTH] - len(tokens)
#         )
#     input_token_ids = []
#     for i in range(len(tokens)):
#         token = tokens[i]
#         if token not in vocab:
#             vocab[token] = len(vocab)
#         input_token_ids.append(vocab[token])
#     return input_token_ids[:-1], input_token_ids[1:]


# def tokenize_dataframe(
#     df: pd.DataFrame,
#     src_vocab: dict[str, int],
#     tgt_vocab: dict[str, int],
#     src_input_tokens: list[list[int]],
#     tgt_input_tokens: list[list[int]],
#     tgt_output_tokens: list[int],
#     params: dict[str, Any] | None = None,
# ) -> tuple[dict[str, int], dict[str, int], list[list[int]], list[list[int]]]:
#     for src_sent, tgt_sent in zip(
#         df[params[names.SRC_LANGUAGE]], df[params[names.TGT_LANGUAGE]]
#     ):
#         src_input = tokenize_sentence_src(
#             sentence=src_sent, vocab=src_vocab, params=params
#         )
#         tgt_input, tgt_output = tokenize_sentence_tgt(
#             sentence=tgt_sent, vocab=tgt_vocab, params=params
#         )
#         src_input_tokens.append(src_input)
#         tgt_input_tokens.append(tgt_input)
#         tgt_output_tokens.append(tgt_output)
#     return src_vocab, tgt_vocab, src_input_tokens, tgt_input_tokens, tgt_output_tokens


# def tokenize_datasets(
#     df_train: pd.DataFrame,
#     df_valid: pd.DataFrame,
#     df_test: pd.DataFrame,
#     params: dict[str, Any] | None = None,
# ) -> None:
#     src_vocab = constants.DEFAULT_VOCAB.copy()  # Initialize source vocabulary
#     tgt_vocab = constants.DEFAULT_VOCAB.copy()  # Initialize target vocabulary
#     (
#         src_vocab,
#         tgt_vocab,
#         src_input_tokens_train,
#         tgt_input_tokens_train,
#         tgt_output_tokens_train,
#     ) = tokenize_dataframe(
#         df=df_train,
#         src_vocab=src_vocab,
#         tgt_vocab=tgt_vocab,
#         src_input_tokens=[],
#         tgt_input_tokens=[],
#         tgt_output_tokens=[],
#         params=params,
#     )
#     (
#         src_vocab,
#         tgt_vocab,
#         src_input_tokens_valid,
#         tgt_input_tokens_valid,
#         tgt_output_tokens_valid,
#     ) = tokenize_dataframe(
#         df=df_valid,
#         src_vocab=src_vocab,
#         tgt_vocab=tgt_vocab,
#         src_input_tokens=[],
#         tgt_input_tokens=[],
#         tgt_output_tokens=[],
#         params=params,
#     )
#     (
#         src_vocab,
#         tgt_vocab,
#         src_input_tokens_test,
#         tgt_input_tokens_test,
#         tgt_output_tokens_test,
#     ) = tokenize_dataframe(
#         df=df_test,
#         src_vocab=src_vocab,
#         tgt_vocab=tgt_vocab,
#         src_input_tokens=[],
#         tgt_input_tokens=[],
#         tgt_output_tokens=[],
#         params=params,
#     )
#     print("Datasets have been tokenized")
#     return (
#         src_vocab,
#         tgt_vocab,
#         src_input_tokens_train,
#         tgt_input_tokens_train,
#         tgt_output_tokens_train,
#         src_input_tokens_valid,
#         tgt_input_tokens_valid,
#         tgt_output_tokens_valid,
#         src_input_tokens_test,
#         tgt_input_tokens_test,
#         tgt_output_tokens_test,
#     )
