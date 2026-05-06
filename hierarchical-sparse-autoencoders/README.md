## Installation

We used Python 3.10.18. To install the required packages, run:

```bash
pip install -r requirements.txt
```
## Steps
Execute code from within the `sae` directory.
The code assumes that you created the necessary embeddings to train the SAE on as described in the README of the MSAE folder.

We show execution for a single seed for DINO. For using CLIP, use the config file `config_vit_361_16.json` instead of `config_dino_361_16.json`. For using a different seed, change the seed in the config file and add a suffix to the name field in the config file.

### Training the SAE
To train the SAE, run:

```bash
python run_moe_eqx.py config_dino_361_16.json
```

### Saving the SAE feature activations
Move the checkpoints into a checkpoints directory in the root of the repo.
To save the feature activations for a set of sentences, run:

```bash
python save_activations.py --config_file ../checkpoints/dino_361_16_v2.json --name dino_361_16_v2
python save_activations.py --config_file ../checkpoints/dino_361_16_v2.json --name dino_361_16_v2 --use_val
```

## Acknowledgements
This code is based on https://github.com/muchanem/hierarchical-sparse-autoencoders.
