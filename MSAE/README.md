## Installation

We used Python 3.10.18. To install the required packages, run:

```bash
pip install -r requirements.txt
```

## Prepare Datasets

We worked with a local version of CC3M, for which we use the class CC3MLocalDataset in the file `my_datasets/cc3m_dataset.py`.
However, you can also use the Hugging Face version of [CC3M](https://huggingface.co/datasets/pixparse/cc3m-wds). To precompute the activations with a specific foundation model, run:

```bash
python precompute_activations.py -d cc3m -m ViT-L~14 --data-split train --image-only
python precompute_activations.py -d cc3m -m dinov2-base --data-split train --image-only
python precompute_activations.py -d cc3m -m ViT-L~14 --data-split val --image-only
python precompute_activations.py -d cc3m -m dinov2-base --data-split val --image-only
```
Note: the term "activations" may be misleading here, as it is about the embeddings and not the SAE feature activations.

### Optional:
To enable the naming of features of SAEs trained on CLIP embeddings, you can execute the following:

```bash
python -m label_assignment_strategies.labelling_and_scoring.precompute_clip_embeddings_of_captions --vocab_name clip_disect_20k --vocab_folder vocab
```

## Experiment Steps

All experiments are run using the scripts in the `scripts_experiments` directory. Using `scripts_experiments/shared_variables.sh` to set the shared variables for all experiments, you can run, which would be the first step of all experiments:

```bash
bash scripts_experiments/00a_train_saes.sh
```
For H-SAE, perform the steps described in the README of the `HSAE` directory after this first step.

Then set the set models paths in `my_config.py` to the trained models from the previous step.
Then continue with the rest of the scripts in `scripts_experiments` to run the evaluations.

Note: The script `scripts_experiments/01b_compute_monosemanticity_parallel.sh` still needs some manual updating of the GPU IDs at the moment.

## Overview of the code structure
Folders:
- `activations_preprocessing`: compute intermediate results based on the precomputed activations e.g., weighted conditional probability of activations given concepts
- `structure_evaluation`: hierarchical evaluation of SAEs
- `label_assignment_strategies`: assign labels to features and individual feature scores and statistics
- `my_datasets`: dataset classes, including the local version of CC3M
- `qualitative_eval`: visualize qualitative examples of the SAEs
- `sae_general_eval`: analyse standard SAE evaluation results
- `scripts_experiments`: scripts to run the experiments end-to-end
- `structure_evaluation`: compute the hierarchical evaluation metrics and run the hierarchical evaluation
- `structure_extraction`: create hierarchy graphs
- `ui`: user interface for visualizing individual features of SAEs and their properties
- `vocabs`: vocabularies for naming features of SAEs trained on CLIP embeddings

Embedding model data generation, SAE training and standard eval:
- `precompute_activations.py`
- `train.py`
- `config.py`
- `extract_sae_embeddings.py`
- `loss.py`
- `sae.py`
- `sae_model_loading_and_saving.py`
- `utils.py`
- `metrics.py`
- `actmsae_repo_datasets.py` (legacy dataset classes)

Own utils:
- `utils_figure_generation.py`
- `valuable_notebook_code_snippets.py`
- `utils_sae_feature_properties.py`
- `path_hub.py`
- `my_utils.py`
- `my_config.py`
- `execute_precompute_sae_data.py`


## Acknowledgements
The code for training and standard evaluation of SAEs is based on https://github.com/WolodjaZ/MSAE. However, we have added additional SAEs variants and, of course, all aspects related to hierarchical evaluation.