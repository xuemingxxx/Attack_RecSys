# Attack_RecSys

This repository accompanies the paper "Privacy Risks of LLM‑Empowered Recommender Systems: An Inversion Attack Perspective" and presents a method for reconstructing textual prompts of recommendation tasks from the next-token probabilities (logits) produced by a victim recommender model. It includes: synthetic dataset generation with automatically created cross-domain, multi-task prompts; inversion model training using a T5-base-based vec2text framework enhanced with our proposed optimization method, Similarity-Guided Refinement; and comprehensive attack evaluation using metrics such as ItemMatch, ProfileMatch, BLEU, ROUGE-L, and Token-F1. Confirmed results are documented at ./tests/result

1. Environment setup:
    pip install requirements.txt


2. Prepare victim recommendation model:
Put the victim rec models at:
if Tallrec:
    ./embedders/TALLRec
if CoLLM:
    ./embedders/CoLLM


3. Train an Inversion model: 
We have created and placed the synthetic datasets for training at ./data/datasets/invInst, just use the following commands to train inversion models:

python recAtk/run.py --per_device_train_batch_size 40 --per_device_eval_batch_size 24 --max_seq_length 256 --num_train_epochs 20  --max_eval_samples 1000 --eval_steps 50000 --warmup_steps 100000 --learning_rate 0.0002 --dataset_name invInst_movie --testset_name tallrec_movie --model_name_or_path t5-base --use_wandb=0 --embedder_model_name tallrec_movie --experiment inversion_from_logits_emb_iterative --bf16=1 --embedder_torch_dtype float16 --lr_scheduler_type constant_with_warmup --use_frozen_embeddings_as_input 1 --mock_embedder 0 --output_dir tests/trained_inversion_models/recAtk_Refined_tallrec_movie

python recAtk/run.py --per_device_train_batch_size 40 --per_device_eval_batch_size 24 --max_seq_length 256 --num_train_epochs 20  --max_eval_samples 1000 --eval_steps 50000 --warmup_steps 100000 --learning_rate 0.0002 --dataset_name invInst_book --testset_name tallrec_book --model_name_or_path t5-base --use_wandb=0 --embedder_model_name tallrec_book --experiment inversion_from_logits_emb_iterative --bf16=1 --embedder_torch_dtype float16 --lr_scheduler_type constant_with_warmup --use_frozen_embeddings_as_input 1 --mock_embedder 0 --output_dir tests/trained_inversion_models/recAtk_Refined_tallrec_book

python recAtk/run.py --per_device_train_batch_size 40 --per_device_eval_batch_size 24 --max_seq_length 256 --num_train_epochs 20  --max_eval_samples 1000 --eval_steps 50000 --warmup_steps 100000 --learning_rate 0.0002 --dataset_name invInst_movie --testset_name collm_movie --model_name_or_path t5-base --use_wandb=0 --embedder_model_name collm_movie --experiment inversion_from_logits_emb_iterative --bf16=1 --embedder_torch_dtype float16 --lr_scheduler_type constant_with_warmup --use_frozen_embeddings_as_input 1 --mock_embedder 0 --output_dir tests/trained_inversion_models/recAtk_Refined_collm_movie

python recAtk/run.py --per_device_train_batch_size 40 --per_device_eval_batch_size 24 --max_seq_length 256 --num_train_epochs 20  --max_eval_samples 1000 --eval_steps 50000 --warmup_steps 100000 --learning_rate 0.0002 --dataset_name invInst_book --testset_name collm_book --model_name_or_path t5-base --use_wandb=0 --embedder_model_name collm_book --experiment inversion_from_logits_emb_iterative --bf16=1 --embedder_torch_dtype float16 --lr_scheduler_type constant_with_warmup --use_frozen_embeddings_as_input 1 --mock_embedder 0 --output_dir tests/trained_inversion_models/recAtk_Refined_collm_book


4. Evaluation:
Refer to ./tests/eval_all.py to evaluate the inversion model's performance.