import os
from typing import Any, Dict
import torch
import torch.nn as nn
import transformers
from sentence_transformers import SentenceTransformer
from transformers import LlamaForCausalLM, LlamaTokenizer
from scipy.sparse import load_npz
from recAtk.models.minigpt4.common.config import Config
from recAtk.models.minigpt4.tasks import setup_task
import argparse
import json
import re



EMBEDDER_MODEL_NAMES = [
    "tallrec_movie",
    "tallrec_book",
    "tallrec_movie_VPD",
    "collm_movie",
    "collm_book",
    "bert",
    "bert__random_init",
    "contriever",
    "dpr",
    "gte_base",
    "gtr_base",
    "gtr_base__random_init",
    "medicalai/ClinicalBERT",
    "gtr_large",
    "ance_tele",
    "dpr_st",
    "gtr_base_st",
    "paraphrase-distilroberta",
    "sentence-transformers/all-MiniLM-L6-v2",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "nomic-ai/nomic-embed-text-v1",
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
]


FREEZE_STRATEGIES = ["decoder", "encoder_and_decoder", "encoder", "none"]
EMBEDDING_TRANSFORM_STRATEGIES = ["repeat"]


def get_device():
    """
    Function that checks
    for GPU availability and returns
    the appropriate device.
    :return: torch.device
    """
    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    device = torch.device(dev)
    return device


device = get_device()


def disable_dropout(model: nn.Module):
    dropout_modules = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    for m in dropout_modules:
        m.p = 0.0
    print(
        f"Disabled {len(dropout_modules)} dropout modules from model type {type(model)}"
    )


def freeze_params(model: nn.Module):
    total_num_params = 0
    for name, params in model.named_parameters():
        params.requires_grad = False
        total_num_params += params.numel()
    # print(f"Froze {total_num_params} params from model type {type(model)}")


def mean_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    B, S, D = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.sum(dim=1) / attention_mask.sum(dim=1)[:, None]
    assert pooled_outputs.shape == (B, D)
    return pooled_outputs


def max_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    B, S, D = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.max(dim=1).values
    assert pooled_outputs.shape == (B, D)
    return pooled_outputs


def stack_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    B, S, D = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.reshape((B, S * D))  # stack along seq length
    assert pooled_outputs.shape == (B, S * D)
    return pooled_outputs


def load_embedder_and_tokenizer(name: str, torch_dtype: str, **kwargs):
    """
    Load an embedding model and its corresponding tokenizer based on a given model name.
    """

    model_kwargs = {
        "low_cpu_mem_usage": True,  # Not compatible with DeepSpeed
        "output_hidden_states": False,
    }

    if name == "dpr":
        # model = SentenceTransformer("sentence-transformers/facebook-dpr-question_encoder-multiset-base")
        model = transformers.DPRContextEncoder.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        )
    elif name == "dpr_st":
        model = SentenceTransformer(
            "sentence-transformers/facebook-dpr-question_encoder-multiset-base"
        )
        tokenizer = model.tokenizer
    elif name == "contriever":
        model = transformers.AutoModel.from_pretrained(
            "facebook/contriever", **model_kwargs
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/contriever")
    elif name == "bert":
        model = transformers.AutoModel.from_pretrained(
            "bert-base-uncased", **model_kwargs
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    elif name == "bert__random_init":
        config = transformers.AutoConfig.from_pretrained("bert-base-uncased")
        model = transformers.AutoModel.from_config(config)
        tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    elif name == "gtr_base":
        model = transformers.AutoModel.from_pretrained(
            "sentence-transformers/gtr-t5-base", **model_kwargs
        ).encoder
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "sentence-transformers/gtr-t5-base"
        )
    elif name == "gtr_large":
        model = transformers.AutoModel.from_pretrained(
            "sentence-transformers/gtr-t5-large", **model_kwargs
        ).encoder
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "sentence-transformers/gtr-t5-large"
        )
    elif name == "gtr_base__random_init":
        config = transformers.AutoConfig.from_pretrained(
            "sentence-transformers/gtr-t5-base"
        )
        model = transformers.AutoModel.from_config(config).encoder
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "sentence-transformers/gtr-t5-base"
        )
    elif name == "gtr_base_st":
        model = SentenceTransformer("sentence-transformers/gtr-t5-base")
        tokenizer = model.tokenizer
    elif name == "gtr_large":
        model = SentenceTransformer("sentence-transformers/gtr-t5-large")
        tokenizer = model.tokenizer
    elif name == "gte_base":
        model = transformers.AutoModel.from_pretrained(
            "thenlper/gte-base", **model_kwargs
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "thenlper/gte-base"
        )
    elif name == "ance_tele":
        model = transformers.AutoModel.from_pretrained(
            "OpenMatch/ance-tele_nq_psg-encoder", **model_kwargs
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "OpenMatch/ance-tele_nq_psg-encoder"
        )
    elif name == "paraphrase-distilroberta":
        model = transformers.AutoModel.from_pretrained(
            "sentence-transformers/paraphrase-distilroberta-base-v1", **model_kwargs
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "sentence-transformers/paraphrase-distilroberta-base-v1"
        )
    elif name == "medicalai/ClinicalBERT":
        model = transformers.AutoModel.from_pretrained(
            "medicalai/ClinicalBERT", **model_kwargs
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
    elif name.startswith("gpt2"):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            name,
            **model_kwargs,
        )
        # model.to_bettertransformer()
        tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        tokenizer.pad_token = tokenizer.eos_token
    elif name.startswith("meta-llama/Llama-2-70b"):
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_config = transformers.AutoConfig.from_pretrained(
            name,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            name,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map="auto",
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        model.eval()


    elif name == "tallrec_movie":
        model_path = "embedders/TALLRec/tallrec_movie"
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
    
    elif name == "tallrec_movie_VPD":
        model_path = "embedders/TALLRec/tallrec_movie_VPD"
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token

    elif name == "tallrec_book":
        model_path = "embedders/TALLRec/tallrec_book"
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token

    elif name.startswith("collm_movie"):
        ckpt_path = "embedders/CoLLM/collm-mf-ml-best.pth"
        is_movie = True
        model = load_collm_model(ckpt_path, is_movie)
        model_path = "embedders/CoLLM/Qwen/Qwen2-1.5B"
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.add_special_tokens({"unk_token":"<unk>"})

    elif name.startswith("collm_book"):
        ckpt_path = "embedders/CoLLM/collm-book-best.pth"
        is_movie = False
        model = load_collm_model(ckpt_path, is_movie)
        model_path = "embedders/CoLLM/Qwen/Qwen2-1.5B"
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.add_special_tokens({"unk_token":"<unk>"})


    elif name.startswith("meta-llama/"):
        if torch_dtype == "float32":
            torch_dtype = torch.float32
        elif torch_dtype == "float16":
            torch_dtype = torch.float16
        elif torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "embedders/meta-llama/Llama-2-7b-hf",
            **model_kwargs,
            token=os.environ.get("LLAMA_TOKEN"),
            torch_dtype=torch_dtype,
            **kwargs,
        )
        # if torch_dtype is not torch.float32:
        #     model.to_bettertransformer()
        tokenizer = transformers.AutoTokenizer.from_pretrained("embedders/meta-llama/Llama-2-7b-hf")
        tokenizer.pad_token = tokenizer.eos_token

    elif name.startswith("sentence-transformers/"):
        model = SentenceTransformer(name)
        tokenizer = model.tokenizer
    elif name.startswith("nomic-ai/nomic-embed-text-v1"):
        model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1", trust_remote_code=True
        )
        tokenizer = model.tokenizer
    else:
        print(f"WARNING: Trying to initialize from unknown embedder {name}")
        model = transformers.AutoModel.from_pretrained(name, **model_kwargs)
        tokenizer = transformers.AutoTokenizer.from_pretrained(name)

    # model = torch.compile(model)
    return model, tokenizer


def load_encoder_decoder(
    model_name: str, lora: bool = False
) -> transformers.AutoModelForSeq2SeqLM:
    model_kwargs: Dict[str, Any] = {
        "low_cpu_mem_usage": True,
    }
    if lora:
        model_kwargs.update(
            {
                "load_in_8bit": True,
                "device_map": "auto",
            }
        )
    return transformers.AutoModelForSeq2SeqLM.from_pretrained(
        model_name, **model_kwargs
    )


def load_tokenizer(name: str, max_length: int) -> transformers.PreTrainedTokenizer:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        name,
        padding="max_length",
        truncation="max_length",
        max_length=max_length,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Disable super annoying warning:
    # https://github.com/huggingface/transformers/issues/22638
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    return tokenizer




def parse_args(is_movie):
    """
    Parses command-line arguments for loading the config file based on domain (movie or book).
    """
    parser = argparse.ArgumentParser(description="Training")
    if is_movie:
        parser.add_argument("--cfg-path", default='embedders/CoLLM/train_configs/collm_pretrain_movie_mf_ood.yaml', help="path to configuration file.")
    else:
        parser.add_argument("--cfg-path", default='embedders/CoLLM/train_configs/collm_pretrain_book_mf_ood.yaml', help="path to configuration file.")
       
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args([])
    return args

def load_collm_model(ckpt_path, is_movie):

    # Parse configuration based on domain
    cfg = Config(parse_args(is_movie))
    task = setup_task(cfg)

    # Load the dataset to extract max user and item IDs
    if is_movie:
        d_path = 'data/datasets/invInst/invInst_movie.json'
        user_num, item_num = get_max_user_and_item_id_from_file(d_path)
    else:
        d_path = 'data/datasets/invInst/invInst_book.json'
        user_num, item_num = get_max_user_and_item_id_from_file(d_path)

    # Inject user/item stats into config
    cfg.model_cfg.rec_config.user_num = int(user_num) 
    cfg.model_cfg.rec_config.item_num = int(item_num) 
    model = task.build_model(cfg)

    # Load model weights from checkpoint (supports 'model' wrapper or plain state_dict)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    return model



def get_max_user_and_item_id_from_file(json_path):
    """
    Parse a JSON dataset file to extract the maximum user ID and item ID mentioned.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    max_user_id = -1
    max_item_id = -1

    # Patterns to extract user ID, item list, and single item ID
    user_pattern = re.compile(r'User\s+(\d+)')
    item_list_pattern = re.compile(r'IDs?:\s*([0-9,\s]+)')  
    item_single_pattern = re.compile(r'titled\s+\\".*?\\" with its ID (\d+)')  

    for entry in dataset:
        text = entry['text']

        # Match and update max user ID
        user_match = user_pattern.search(text)
        if user_match:
            user_id = int(user_match.group(1))
            max_user_id = max(max_user_id, user_id)

        # Match item ID lists (e.g., "IDs: 3, 12, 24")
        item_matches = item_list_pattern.findall(text)
        for group in item_matches:
            ids = [int(i.strip()) for i in group.split(',') if i.strip().isdigit()]
            if ids:
                max_item_id = max(max_item_id, max(ids))

        # Match single target item (e.g., "with its ID 42")
        single_match = item_single_pattern.search(text)
        if single_match:
            rec_id = int(single_match.group(1))
            max_item_id = max(max_item_id, rec_id)

    return max_user_id, max_item_id