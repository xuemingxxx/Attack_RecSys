import logging
import os
import random
from typing import Dict, List

import datasets
import torch

from recAtk.run_args import DataArguments
from recAtk.utils import dataset_map_multi_worker, get_num_proc
import ipdb

def retain_dataset_columns(
    d: datasets.Dataset, allowed_columns: List[str]
) -> datasets.Dataset:
    column_names_to_remove = [c for c in d.features if c not in allowed_columns]
    return d.remove_columns(column_names_to_remove)


def load_nq_dpr_corpus() -> datasets.Dataset:
    return datasets.load_dataset("jxm/nq_corpus_dpr")


def load_msmarco_corpus() -> datasets.Dataset:
    # has columns ["title", "text"]. only one split ("train")
    dataset_dict = datasets.load_dataset("Tevatron/msmarco-passage-corpus")
    return dataset_dict["train"]


def create_omi_ex(ex: Dict[str, str]) -> Dict[str, str]:
    ex["text"] = ex["user"]
    return ex


def create_ompi_ex(ex: Dict[str, str]) -> Dict[str, str]:
    ex["user"] = ex["user"].strip()
    ex["system"] = ex["system"].strip()
    ex["text"] = ex["system"] + "\n\n" + ex["user"]
    ex["prefix"] = ex["system"] + "\n\n"
    ex["suffix"] = ex["user"]
    return ex

def get_world_size() -> int:
    try:
        return torch.distributed.get_world_size()
    except (RuntimeError, ValueError):
        return 1


def load_one_million_instructions() -> datasets.Dataset:
    # has only "train" split, and "system" (system prompt)
    # and "user" (user input) columns
    dataset_dict = datasets.load_dataset("wentingzhao/one-million-instructions")
    dataset_dict = dataset_map_multi_worker(dataset_dict, create_ompi_ex)
    return dataset_dict["train"]

def load_invInst_movie() -> datasets.Dataset:
    d = datasets.load_dataset('json', data_files='data/datasets/invInst/invInst_movie.json')["train"]
    return d

def load_invInst_movie_TCD() -> datasets.Dataset:
    d = datasets.load_dataset('json', data_files='data/datasets/invInst/invInst_movie_TCD.json')["train"]
    return d

def load_invInst_book() -> datasets.Dataset:
    d = datasets.load_dataset('json', data_files='data/datasets/invInst/invInst_book.json')["train"]
    return d


def load_anthropic_toxic_prompts() -> datasets.Dataset:
    d = datasets.load_dataset("wentingzhao/anthropic-hh-first-prompt")["train"]
    d = d.rename_column("user", "text")
    return d


def load_luar_reddit() -> datasets.Dataset:
    d = datasets.load_dataset("friendshipkim/reddit_eval_embeddings_luar")
    d = d.rename_column("full_text", "text")
    d = d.rename_column("embedding", "frozen_embeddings")
    return d


def dataset_from_args(data_args: DataArguments) -> datasets.DatasetDict:
    """Loads a dataset from data_args create in `run_args`."""
    if data_args.dataset_name == "one_million_instructions":
        raw_datasets = load_one_million_instructions()
        raw_datasets = raw_datasets.train_test_split(test_size=0.01)
        raw_datasets["validation"] = raw_datasets["test"]

    elif data_args.dataset_name == "invInst_movie_TCD":
        raw_datasets = load_invInst_movie_TCD()
        raw_datasets = raw_datasets.train_test_split(test_size=0.001)
        raw_datasets["validation"] = raw_datasets["test"]

    elif data_args.dataset_name == "invInst_movie":
        raw_datasets = load_invInst_movie()
        raw_datasets = raw_datasets.train_test_split(test_size=0.001)
        raw_datasets["validation"] = raw_datasets["test"]

    elif data_args.dataset_name == "invInst_book":
        raw_datasets = load_invInst_book()
        raw_datasets = raw_datasets.train_test_split(test_size=0.001)
        raw_datasets["validation"] = raw_datasets["test"]

    else:
        raise ValueError(f"unsupported dataset {data_args.dataset_name}")
    return raw_datasets


def load_ag_news_test() -> datasets.Dataset:
    return datasets.load_dataset("ag_news")["test"]


def load_xsum_val(col: str) -> datasets.Dataset:
    d = datasets.load_dataset("xsum")["validation"]
    d = d.rename_column(col, "text")
    return d


def load_wikibio_val() -> datasets.Dataset:
    d = datasets.load_dataset("wiki_bio", trust_remote_code=True)["val"]
    d = d.rename_column("target_text", "text")
    return d


def load_arxiv_val() -> datasets.Dataset:
    d = datasets.load_dataset("ccdv/arxiv-summarization")["validation"]
    d = d.rename_column("abstract", "text")
    return d


def load_python_code_instructions_18k_alpaca() -> datasets.Dataset:
    d = datasets.load_dataset("iamtarun/python_code_instructions_18k_alpaca")["train"]
    d = d.rename_column("instruction", "text")
    return d

# tallrec movie -------------------------
def load_tallrec_ml_1m_profile_7_valid() -> datasets.Dataset:
    d = datasets.load_dataset('json', data_files='data/datasets/test_sets/main/tallrec_ml-1m_profile_7_valid.json')["train"]
    return d

# tallrec book -------------------------
def load_tallrec_book_profile_7_valid() -> datasets.Dataset:
    d = datasets.load_dataset('json', data_files='data/datasets/test_sets/main/tallrec_book_profile_7_valid.json')["train"]
    return d

# collm movie -------------------------
def load_collm_ml_1m_profile_7_valid() -> datasets.Dataset:
    d = datasets.load_dataset('json', data_files='data/datasets/test_sets/main/collm_ml-1m_profile_7_valid.json')["train"]
    return d

# collm book -------------------------
def load_collm_book_profile_7_valid() -> datasets.Dataset:
    d = datasets.load_dataset('json', data_files='data/datasets/test_sets/main/collm_book_profile_7_valid.json')["train"]
    return d

# movie item num test -------------------------
def load_tallrec_movie_3_valid() -> datasets.Dataset:
    d = datasets.load_dataset('json', data_files='data/datasets/test_sets/item_num_test/tallrec_movie_3_valid.json')["train"]
    return d

def load_tallrec_movie_5_valid() -> datasets.Dataset:
    d = datasets.load_dataset('json', data_files='data/datasets/test_sets/item_num_test/tallrec_movie_5_valid.json')["train"]
    return d

def load_tallrec_movie_7_valid() -> datasets.Dataset:
    d = datasets.load_dataset('json', data_files='data/datasets/test_sets/item_num_test/tallrec_movie_7_valid.json')["train"]
    return d

def load_tallrec_movie_9_valid() -> datasets.Dataset:
    d = datasets.load_dataset('json', data_files='data/datasets/test_sets/item_num_test/tallrec_movie_9_valid.json')["train"]
    return d

def load_tallrec_movie_11_valid() -> datasets.Dataset:
    d = datasets.load_dataset('json', data_files='data/datasets/test_sets/item_num_test/tallrec_movie_11_valid.json')["train"]
    return d

# book length test -------------------------
def load_tallrec_book_3_valid() -> datasets.Dataset:
    d = datasets.load_dataset('json', data_files='data/datasets/test_sets/item_num_test/tallrec_book_3_valid.json')["train"]
    return d

def load_tallrec_book_5_valid() -> datasets.Dataset:
    d = datasets.load_dataset('json', data_files='data/datasets/test_sets/item_num_test/tallrec_book_5_valid.json')["train"]
    return d

def load_tallrec_book_7_valid() -> datasets.Dataset:
    d = datasets.load_dataset('json', data_files='data/datasets/test_sets/item_num_test/tallrec_book_7_valid.json')["train"]
    return d

def load_tallrec_book_9_valid() -> datasets.Dataset:
    d = datasets.load_dataset('json', data_files='data/datasets/test_sets/item_num_test/tallrec_book_9_valid.json')["train"]
    return d

def load_tallrec_book_11_valid() -> datasets.Dataset:
    d = datasets.load_dataset('json', data_files='data/datasets/test_sets/item_num_test/tallrec_book_11_valid.json')["train"]
    return d

def combine_instruction_input(examples):
    examples["text"] = examples["instruction"] + " " + examples["input"]
    return examples


def load_standard_val_datasets(testset_name) -> datasets.DatasetDict:
    """Loads a pre-defined set of standard val datasets."""

    if testset_name == "tallrec_movie":
        d = {
            "tallrec_ml-1m_profile_7": load_tallrec_ml_1m_profile_7_valid(),
        }
    elif testset_name == "tallrec_book":
        d = {
            "tallrec_book_profile_7": load_tallrec_book_profile_7_valid(),
        }
    elif testset_name == "collm_movie":
        d = {
            "collm_ml-1m_profile_7": load_collm_ml_1m_profile_7_valid(),
        }
    elif testset_name == "collm_book":
        d = {
            "collm_book_profile_7": load_collm_book_profile_7_valid(),
        }
    elif testset_name == "tallrec_movie_INT":
        d = {
            "tallrec_movie_3": load_tallrec_movie_3_valid(),
            "tallrec_movie_5": load_tallrec_movie_5_valid(),
            "tallrec_movie_7": load_tallrec_movie_7_valid(),
            "tallrec_movie_9": load_tallrec_movie_9_valid(),
            "tallrec_movie_11": load_tallrec_movie_11_valid(),
        }
    elif testset_name == "tallrec_book_INT":
        d = {
            "tallrec_book_3": load_tallrec_book_3_valid(),
            "tallrec_book_5": load_tallrec_book_5_valid(),
            "tallrec_book_7": load_tallrec_book_7_valid(),
            "tallrec_book_9": load_tallrec_book_9_valid(),
            "tallrec_book_11": load_tallrec_book_11_valid(),
        }
    
    else:
        d = {
            "tallrec_ml-1m_profile_7": load_tallrec_ml_1m_profile_7_valid(),
            "tallrec_book_profile_7": load_tallrec_book_profile_7_valid(),
            "collm_ml-1m_profile_7": load_collm_ml_1m_profile_7_valid(),
            "collm_book_profile_7": load_collm_book_profile_7_valid(),
        }

    d = {k: retain_dataset_columns(v, ["text"]) for k, v in d.items()}

    return datasets.DatasetDict(d)
