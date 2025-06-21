import sys
sys.path.append('./')
from recAtk import analyze_utils


experiment, trainer = analyze_utils.load_experiment_and_trainer_from_pretrained(
    name = "tests/trained_inversion_models/recAtk_Refined_tallrec_movie", use_less_data = 1
    # name = "tests/trained_inverse_models/t5-base__llama-7b__one-million-instructions__emb"
)

# trainer.model.use_frozen_embeddings_as_input = False
trainer.args.per_device_eval_batch_size = 10
results = {}
for dataset_name, dataset in trainer.eval_dataset.items():
    eval_results = trainer.evaluate(eval_dataset=dataset.select(range(20)))
