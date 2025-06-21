# Attack_RecSys

## Introduction

This repository accompanies the paper "Privacy Risks of LLM‑Empowered Recommender Systems: An Inversion Attack Perspective" and presents a method for reconstructing textual prompts of recommendation tasks from the next-token probabilities (logits) produced by a victim recommender model. It includes: synthetic dataset generation with automatically created cross-domain, multi-task prompts; inversion model training using a T5-base-based vec2text framework enhanced with our proposed optimization method, Similarity-Guided Refinement; and comprehensive attack evaluation using metrics such as ItemMatch, ProfileMatch, BLEU, ROUGE-L, and Token-F1. Confirmed results are documented at ./tests/result

## Train an Inversion Model
The file demo.mp4 provides a demonstration of the training and evaluation procedures.

### 1. Environment setup:
You can either create the environment using the provided YAML file:
```bash
conda env create -f environment.yml
conda activate recAtk
```
or manually：
```bash
conda create --name recAtk python=3.10
conda activate recAtk 
pip install -r requirements.txt
```

### 2. Prepare victim recommendation model:
Place the victim RecSys models into the following directories:

if you would like to attack TallRec, you should refer to the instruction at https://github.com/SAI990323/TALLRec to train a TallRec model and place it (e.g., tallrec_movie) at:
```bash
./embedders/TALLRec
```
Example Structure：
```bash
├── TALLRec/
│   ├── tallrec_movie
│   │   ├── config.json
│   │   ├── model-001-of-002.safetensors
│   │   ├── model-001-of-002.safetensors
│   │   ├── tokenizer_config.json
│   │   ├── tokenizer.json
│   │   └── tokenizer.model
```
if you would like to attack CoLLM, you should refer to the instruction at https://github.com/zyang1580/CoLLM to prepare and place the configuration file, collm training datasets, Matrix Factorization model and CIE model at:
```bash
./embedders/CoLLM
```
Example Structure：
```bash
├── CoLLM/
│   ├── collm-datasets
│   ├── mf_model
│   ├── train_configs
│   └── collm_movie.pth
```
and relatively modify load_collm_model, parse_args method at model_utils.py file according to your CoLLM file structure.


### 3. Train an Inversion model: 
We have created and placed the synthetic datasets for training at：
```bash
./data/datasets/invInst
```
You can use this command to train inversion models:
```bash
python recAtk/run.py --per_device_train_batch_size 40 --per_device_eval_batch_size 24 --max_seq_length 256 --num_train_epochs 20  --max_eval_samples 1000 --eval_steps 50000 --warmup_steps 100000 --learning_rate 0.0002 --dataset_name invInst_movie --testset_name tallrec_movie --model_name_or_path t5-base --use_wandb=0 --embedder_model_name tallrec_movie --experiment inversion_from_logits_emb_iterative --bf16=1 --embedder_torch_dtype float16 --lr_scheduler_type constant_with_warmup --use_frozen_embeddings_as_input 1 --mock_embedder 0 --output_dir tests/trained_inversion_models/recAtk_Refined_tallrec_movie
```
Some key parameters that can be modified or added as needed include:
```bash
--per_device_train_batch_size # Batch size used for training on each device.
--per_device_eval_batch_size # Batch size used for evaluation on each device.
--dataset_name # Name of the training dataset to be used (e.g., invInst_book, invInst_movie).
--testset_name # Name of the test dataset for evaluation (e.g., tallrec_movie, tallrec_book, collm_movie, collm_book).
--embedder_model_name # Name of the victim recommendation model (e.g., tallrec_movie, tallrec_book, collm_movie, collm_book).
--experiment # Whether to use our Similarity-Guided Refinement method (change to inversion_from_logits_emb if you don't want to use).
--use_less_data # Useing only a small subset of the training data.
--resume_from_checkpoint # Path to a checkpoint to resume training from.
```
The training metrics are reported at the end of the training process.
```bash
***** train metrics *****
  epoch                    =        5.0
  train_loss               =     9.8249
  train_runtime            = 0:00:29.78
  train_samples_per_second =     16.786
  train_steps_per_second   =      0.504
```

### 4. Evaluation:
Run the following command to evaluate the inversion model's performance. 
```bash
python ./tests/eval_all.py
```
You can modify the parameters by adding code in the python file, such as:
```bash
trainer.args.per_device_eval_batch_size = 20
```
The results will be directly returned to the terminal, and you can also find the evaluation results in 
```bash
./tests/result/
```
The results include evaluation metrics and 20 pairs of original and reconstructed prompt samples.
```bash
[pred] Analyze the user's profile, the movies the user enjoys and those they dislike, and respond "Yes." or "No." to indicate if they will like the target movie. The user is a 44-year-old male. Movies user enjoys: "E.T. the Extra-Terrestrial", "Fifth Element, The", "Lord of the Rings, The" Movies user dislikes: "Adventures of Buckaroo Banzai Across the 8th Dimension, The", "Young Sherlock Holmes", "Porky's II: The Next Day" Whether the user will like the target movie "Cheech and Chong's Up in Smoke"?
[true] Analyze the user's profile, the movies the user enjoys and those they dislike, and respond "Yes." or "No." to indicate if they will like the target movie. The user is a 45-year-old male Movies user enjoys: "E.T. the Extra-Terrestrial", "Ladyhawke", "Fifth Element, The" Movies user dislikes: "Young Sherlock Holmes", "Escape to Witch Mountain", "Adventures of Buckaroo Bonzai Across the 8th Dimension, The" Whether the user will like the target movie "Porky's II: The Next Day"?
```
```bash
pred_num_tokens: 128.0
true_num_tokens: 123.80000305175781
token_set_precision: 0.8907491402300837
token_set_recall: 0.8896512948885658
token_set_f1: 0.8897432376518939
token_set_f1_sem: 0.010402108739610004
n_ngrams_match_1: 91.1
n_ngrams_match_2: 85.25
n_ngrams_match_3: 77.45
num_true_words: 97.75
num_pred_words: 100.35
bleu_score: 81.14018531025269
bleu_score_sem: 1.1281194192075208
rouge_score: 0.8997172234510601
exact_match: 0.0
exact_match_sem: 0.0
emb_cos_sim: 0.99951171875
emb_cos_sim_sem: 0.00015440808887540915
emb_top1_equal: 0.6000000238418579
emb_top1_equal_sem: 0.16329931451750404
eval_item_match: 0.5428571428571429
eval_profile_match: 0.8
```

### 5. Attack efficiency table:
The following runtimes were measured on an A40 GPU with 40GB of memory.

| Model   | Train Time (1.5M prompts)         | Evaluate Time (1k prompts) |
|---------|----------------------------------|-----------------------------|
| TallRec | 342,174 seconds (~95 hours)      | 811 seconds                 |
| CoLLM   | 429,127 seconds (~119 hours)     | 981 seconds                 |