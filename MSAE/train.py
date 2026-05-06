import torch
import inspect
import argparse
import logging
from tqdm import tqdm
from dataclasses import asdict
import wandb

from metrics import calculate_similarity_metrics, identify_dead_neurons, orthogonal_decoder, cknna, explained_variance
from sae_model_loading_and_saving import save_model
from utils import SAEDataset, set_seed, get_device, geometric_median, calculate_vector_mean, LinearDecayLR, CosineWarmupScheduler
from config import get_config
from sae import EWGSAE, Autoencoder, MatryoshkaAutoencoder, ArchitecturalMatryoshkaSAE, MatchingPursuitSAE, MultiSAE
from loss import SAELoss

"""
Sparse Autoencoder (SAE) Training Script

This script provides a complete pipeline for training various types of sparse autoencoder models,
including standard SAEs with different activation functions and Matryoshka SAEs with nested
feature hierarchies. It handles training, evaluation, and model saving with configurable
hyperparameters.
"""

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def _to_serializable(value):
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _to_log_scalar(value):
    if isinstance(value, torch.Tensor):
        return value.detach().item()
    return value


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the SAE training script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments with the following fields:
            - dataset_train: Path to the training dataset
            - dataset_test: Path to the testing/validation dataset
            - model: Model architecture to train (e.g., "ReLUSAE", "TopKSAE")
            - activation: Activation function to use
            - epochs: Number of training epochs
            - learning_rate: Initial learning rate
            - expansion_factor: Ratio of latent dimensions to input dimensions
    """
    parser = argparse.ArgumentParser(description="Train Sparse Autoencoder (SAE) models")
    parser.add_argument("-dt", "--dataset_train", type=str, required=True, 
                       help="Path to training dataset file (.npy)")
    parser.add_argument("-ds", "--dataset_test", type=str, required=True, 
                       help="Path to testing/validation dataset file (.npy)")
    parser.add_argument("-dm", "--dataset_second_modality", type=str, default=None,
                       help="Path to second modality dataset file (.npy)")
    parser.add_argument("-m", "--model", type=str, required=True, 
                       choices=["ReLUSAE", "TopKSAE", "BatchTopKSAE", "MSAE_UW", "MSAE_RW", "ArchMSAE_UW", "ArchMSAE_FW", "MPSAE", "MultiSAE", "EWGSAE"],
                       help="Model architecture to train")
    parser.add_argument("-a", "--activation", type=str, required=True, 
                       help="Activation function (e.g., 'ReLU_003', 'TopKReLU_64')")
    parser.add_argument("-e", "--epochs", type=int, default=100, 
                       help="Number of training epochs")
    parser.add_argument("-ef", "--expansion_factor", type=float, default=1.0, 
                       help="Ratio of latent dimensions to input dimensions")
    parser.add_argument("-ms", "--max_dataset_size", type=int, required=True,
                       help="Maximum number of samples to use from training dataset")
    parser.add_argument("-s", "--seed", type=int, default=42,
                       help="Random seed for reproducibility")

    return parser.parse_args()


def eval(model, eval_loader, loss_fn, device, cfg):
    # Evaluation phase
    loss_all = 0.0
    recon_loss_all = 0.0
    sparse_loss_all = 0.0
    cknna_score_sparse_sum = 0.0
    cknna_score_all_sum = 0.0
    fvu_score_all_sum = 0.0
    fvu_score_sparse_sum = 0.0
    diagonal_cs_sparse_sum = 0.0
    diagonal_cs_all_sum = 0.0
    mae_distance_sparse_sum = 0.0
    mae_distance_all_sum = 0.0
    od_sum = 0.0
    sparsity_sparse_sum = 0.0
    sparsity_all_sum = 0.0
    
    # Switch to evaluation mode
    model.eval()
    for step, embeddings in enumerate(tqdm(eval_loader, desc="Evaluation")):
        embeddings = embeddings.to(device)

        # forward pass
        with torch.no_grad():
            recons_sparse, repr_sparse, recons_all, repr_all = model(embeddings)
        
        # postprocess return values of forward and loss function computation based on model type
        if cfg.model.use_matryoshka:
            with torch.no_grad():
                loss, recon_loss, sparse_loss = loss_fn(recons_all, embeddings, repr_all)
            recons_sparse = recons_sparse[0] # use highest sparsity level for metrics
            repr_sparse = repr_sparse[0] # use highest sparsity level for metrics
        elif cfg.model.use_mpsae:
            with torch.no_grad():
                loss, recon_loss, sparse_loss = loss_fn(recons_all, embeddings, repr_all)
        elif cfg.model.use_arch_matryoshka:
            recons_sparse_per_prefix = recons_sparse # just for naming consistency
            with torch.no_grad():
                loss, recon_loss, sparse_loss = loss_fn(recons_all, embeddings, repr_all)
            # Modify variables for metric calculation and comparability with other models
            recons_sparse = recons_sparse_per_prefix[-1] # use reconstructions of prefix with all latents
        elif cfg.model.use_multisae:
            SAE_POSITION = 0 # TODO: set which sae to use for evaluation
            recons_sparse = recons_sparse[SAE_POSITION]
            recons_all = recons_all[SAE_POSITION]
            repr_sparse = repr_sparse[SAE_POSITION]
            repr_all = repr_all[SAE_POSITION]
            with torch.no_grad():
                loss, recon_loss, sparse_loss = loss_fn(recons_all, embeddings, repr_all)
        else:
            with torch.no_grad():
                loss, recon_loss, sparse_loss = loss_fn(recons_all, embeddings, repr_all)
        
        # Accumulate loss metrics
        loss_all += loss.item()
        recon_loss_all += recon_loss.item()
        sparse_loss_all += sparse_loss.item()
        
        # Accumulate CKNNA scores
        cknna_score_sparse_sum += cknna(recons_sparse, embeddings)
        cknna_score_all_sum += cknna(recons_all, embeddings)
        
        # Accumulate similarity metrics
        fvu_score_sparse_sum += explained_variance(embeddings, recons_sparse)
        fvu_score_all_sum += explained_variance(embeddings, recons_all)
        distance_sparse = calculate_similarity_metrics(embeddings, recons_sparse)
        distance_all = calculate_similarity_metrics(embeddings, recons_all)
        diagonal_cs_sparse_sum += distance_sparse[0]
        mae_distance_sparse_sum += distance_sparse[1]
        diagonal_cs_all_sum += distance_all[0]
        mae_distance_all_sum += distance_all[1]
        
        # Accumulate orthogonality measure
        if not model.tied:
            od_sum += orthogonal_decoder(model.decoder)
        else:
            od_sum += 0.0
        
        # Accumulate sparsity measures
        sparsity_sparse_sum += (repr_sparse == 0.0).float().mean(axis=-1).mean()
        sparsity_all_sum += (repr_all == 0.0).float().mean(axis=-1).mean()
    
    # Log evaluation metrics (averaged over batches)
    logger.info("Evaluation results:")
    logger.info(f"  Loss: {loss_all / len(eval_loader):.6f}")
    logger.info(f"  Reconstruction Loss: {recon_loss_all / len(eval_loader):.6f}")
    logger.info(f"  Sparsity Loss: {sparse_loss_all / len(eval_loader):.6f}")
    logger.info(f"  FVU Sparse: {fvu_score_sparse_sum / len(eval_loader):.4f}")
    logger.info(f"  FVU All: {fvu_score_all_sum / len(eval_loader):.4f}")
    logger.info(f"  CKNNA Sparse: {cknna_score_sparse_sum / len(eval_loader):.4f}")
    logger.info(f"  CKNNA All: {cknna_score_all_sum / len(eval_loader):.4f}")
    logger.info(f"  Cosine Similarity Sparse: {diagonal_cs_sparse_sum / len(eval_loader):.4f}")
    logger.info(f"  MAE Distance Sparse: {mae_distance_sparse_sum / len(eval_loader):.4f}")
    logger.info(f"  Cosine Similarity All: {diagonal_cs_all_sum / len(eval_loader):.4f}")
    logger.info(f"  MAE Distance All: {mae_distance_all_sum / len(eval_loader):.4f}")
    logger.info(f"  Sparsity Sparse: {sparsity_sparse_sum / len(eval_loader):.4f}")
    logger.info(f"  Sparsity All: {sparsity_all_sum / len(eval_loader):.4f}")
    logger.info(f"  Orthogonal Decoder Loss: {od_sum / len(eval_loader):.6f}")

    return {
        "loss": loss_all / len(eval_loader),
        "reconstruction_loss": recon_loss_all / len(eval_loader),
        "sparsity_loss": sparse_loss_all / len(eval_loader),
        "fvu_sparse": fvu_score_sparse_sum / len(eval_loader),
        "fvu_all": fvu_score_all_sum / len(eval_loader),
        "cknna_sparse": cknna_score_sparse_sum / len(eval_loader),
        "cknna_all": cknna_score_all_sum / len(eval_loader),
        "cosine_similarity_sparse": diagonal_cs_sparse_sum / len(eval_loader),
        "mae_distance_sparse": mae_distance_sparse_sum / len(eval_loader),
        "cosine_similarity_all": diagonal_cs_all_sum / len(eval_loader),
        "mae_distance_all": mae_distance_all_sum / len(eval_loader),
        "sparsity_sparse": sparsity_sparse_sum / len(eval_loader),
        "sparsity_all": sparsity_all_sum / len(eval_loader),
        "orthogonal_decoder_loss": od_sum / len(eval_loader),
    }

def arch_matryoshka_loss_helper(loss_fn, recons_sparse_per_prefix, recons_all, 
                                repr_sparse, repr_all, embeddings, model, device):
    # Weight reconstruction losses by relative importance
    loss_recon_all = torch.tensor(0., requires_grad=True, device=device)
    for i in range(len(recons_sparse_per_prefix)):
        current_loss = loss_fn(recons_sparse_per_prefix[i], embeddings, repr_sparse)[1]
        loss_recon_all = loss_recon_all + current_loss * model.relative_importance[i]

    # Normalize by sum of weights
    loss = loss_recon_all / sum(model.relative_importance)
    
    recons_sparse = recons_sparse_per_prefix[-1] # use reconstructions of prefix with all latents
    sparse_loss = loss_fn(recons_all, embeddings, repr_all)[-1]
    recon_loss = loss_fn(recons_sparse, embeddings, repr_all)[1]
    return loss, recon_loss, sparse_loss

def act_matryoshka_loss_helper(loss_fn, recons_sparse, recons_all, repr_sparse, repr_all, embeddings, model, device):
    # For Matryoshka models, compute weighted loss across all nesting levels
    # Weight reconstruction losses by relative importance
    loss_recon_all = torch.tensor(0., requires_grad=True, device=device)
    for i in range(len(recons_sparse)):
        current_loss = loss_fn(recons_sparse[i], embeddings, repr_sparse[i])[1]
        loss_recon_all = loss_recon_all + current_loss * model.relative_importance[i]

    # Normalize by sum of weights
    loss = loss_recon_all / sum(model.relative_importance)

    # Separate loss components (NOT USED FOR ACTUAL TRAINING!) just for evaluation/logging
    sparse_loss = loss_fn(recons_all, embeddings, repr_all)[-1]
    recon_loss = loss_fn(recons_sparse[0], embeddings, repr_all)[1]

    return loss, recon_loss, sparse_loss

def multisae_loss_helper(loss_fn, recons_sparse, recons_all, repr_sparse, repr_all, embeddings, model, device):
    # For MultiSAE, compute independent losses and sum them
    
    loss = torch.tensor(0., requires_grad=True, device=device)

    for i in range(len(recons_sparse)):
        current_loss, _ , _ = loss_fn(recons_sparse[i], embeddings, repr_sparse[i])
        loss = loss + current_loss * 1.0

    # Normalization skipped for MultiSAE these are independent models
    
    # Separate loss components (NOT USED FOR ACTUAL TRAINING!) just for evaluation/logging
    # Use first SAE for metrics
    recons_sparse = recons_sparse[0]
    repr_sparse = repr_sparse[0]
    _, recon_loss, sparse_loss = loss_fn(recons_sparse, embeddings, repr_sparse)
    return loss, recon_loss, sparse_loss

def eval_during_train(repr_sparse, repr_all, recons_sparse, recons_all, embeddings, model):
    # Calculate evaluation metrics
    # CKNNA (Centered Kernel Nearest Neighbor Alignment) scores
    cknna_score_sparse = cknna(recons_sparse, embeddings)
    cknna_score_all = cknna(recons_all, embeddings)
    
    # FVU (Explained Variance) metric
    fvu_score_sparse = explained_variance(embeddings, recons_sparse)
    fvu_score_all = explained_variance(embeddings, recons_all)
    
    # Reconstruction quality metrics
    diagonal_cs_sparse, mae_distance_sparse = calculate_similarity_metrics(recons_sparse, embeddings)
    diagonal_cs_all, mae_distance_all = calculate_similarity_metrics(recons_all, embeddings)
    
    # Orthogonality of decoder features
    if not model.tied:
        od = orthogonal_decoder(model.decoder)
    else:
        od = torch.tensor(0.0)
    
    # Sparsity measurements
    sparsity_sparse = (repr_sparse == 0.0).float().mean(axis=-1).mean()
    sparsity_all = (repr_all == 0.0).float().mean(axis=-1).mean()

    # Representation Metrics
    repr_norm = repr_all.norm(dim=-1).mean().item()
    repr_max = repr_all.max(dim=-1).values.mean().item()

    # return as a dictionary
    return {
        "cknna_score_sparse": cknna_score_sparse,
        "cknna_score_all": cknna_score_all,
        "fvu_score_sparse": fvu_score_sparse,
        "fvu_score_all": fvu_score_all,
        "diagonal_cs_sparse": diagonal_cs_sparse,
        "diagonal_cs_all": diagonal_cs_all,
        "mae_distance_sparse": mae_distance_sparse,
        "mae_distance_all": mae_distance_all,
        "od": od,
        "sparsity_sparse": sparsity_sparse,
        "sparsity_all": sparsity_all,
        "repr_norm": repr_norm,
        "repr_max": repr_max,
        }

def display_eval_metrics(epoch, step, loss, recon_loss, sparse_loss, eval_metrics):
    logger.info(f"Epoch: {epoch+1}, Step: {step}, Loss: {loss.item():.6f}, Recon Loss: {recon_loss.item():.6f}, Sparse Loss: {sparse_loss.item():.6f}")
    
    # using eval_metrics dictionary
    logger.info(f"FVU Sparse: {eval_metrics['fvu_score_sparse']:.4f}, FVU All: {eval_metrics['fvu_score_all']:.4f}")
    logger.info(f"CKNNA Sparse: {eval_metrics['cknna_score_sparse']:.4f}, CKNNA All: {eval_metrics['cknna_score_all']:.4f}")
    logger.info(f"Cosine Similarity Sparse: {eval_metrics['diagonal_cs_sparse']:.4f}, MAE Distance Sparse: {eval_metrics['mae_distance_sparse']:.4f}")
    logger.info(f"Cosine Similarity All: {eval_metrics['diagonal_cs_all']:.4f}, MAE Distance All: {eval_metrics['mae_distance_all']:.4f}")
    logger.info(f"Sparsity Sparse: {eval_metrics['sparsity_sparse']:.4f}, Sparsity All: {eval_metrics['sparsity_all']:.4f}")
    logger.info(f"Orthogonal Decoder Loss: {eval_metrics['od']:.6f}, Representation norm {eval_metrics['repr_norm']:.4f} and max {eval_metrics['repr_max']:.2f}")


def load_datasets(args, cfg):
    # Load datasets
    train_ds = SAEDataset(
        args.dataset_train, 
        dtype=cfg.training.dtype, 
        mean_center=cfg.training.mean_center, 
        target_norm=cfg.training.target_norm,
        max_size=args.max_dataset_size if hasattr(args, 'max_dataset_size') else None
    )
    eval_ds = SAEDataset(
        args.dataset_test, 
        dtype=cfg.training.dtype, 
        mean_center=cfg.training.mean_center, 
        target_norm=cfg.training.target_norm
    )
    logger.info(f"Training dataset length: {len(train_ds)}, Evaluation dataset length: {len(eval_ds)}, Embedding size: {train_ds.vector_size}")
    logger.info(f"Training dataset mean center: {train_ds.mean.mean()}, Scaling factor: {train_ds.scaling_factor} with target norm {train_ds.target_norm}")
    logger.info(f"Evaluation dataset mean center: {eval_ds.mean.mean()}, Scaling factor: {eval_ds.scaling_factor} with target norm {eval_ds.target_norm}")
    assert train_ds.vector_size == eval_ds.vector_size, "Training and evaluation datasets must have the same embedding size"
    eval_ds_second = None
    if args.dataset_second_modality is not None:
        eval_ds_second = SAEDataset(
            args.dataset_second_modality,
            dtype=cfg.training.dtype,
            mean_center=cfg.training.mean_center,
            target_norm=cfg.training.target_norm
        )
        logger.info(f"Second modality dataset mean center: {eval_ds_second.mean.mean()}, Scaling factor: {eval_ds_second.scaling_factor} with target norm {eval_ds_second.target_norm}")
        assert train_ds.vector_size == eval_ds_second.vector_size, "Training and second modality datasets must have the same embedding size"
    return train_ds, eval_ds, eval_ds_second
def get_dataloaders(train_ds, eval_ds, eval_ds_second, cfg, args):
    # Prepare the dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=cfg.training.batch_size, 
        num_workers=cfg.training.num_workers, 
        shuffle=True
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_ds, 
        batch_size=cfg.training.batch_size, 
        num_workers=cfg.training.num_workers, 
        shuffle=False
    )
    eval_loader_second = None
    if args.dataset_second_modality is not None:
        eval_loader_second = torch.utils.data.DataLoader(
            eval_ds_second,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
            shuffle=False
        )
    return train_loader, eval_loader, eval_loader_second

def load_scheduler_and_optimizer(model, cfg, device):
    # Prepare the optimizer with adaptive settings based on device
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and "cuda" in device.type
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.training.lr, 
        betas=(cfg.training.beta1, cfg.training.beta2), 
        eps=cfg.training.eps, 
        weight_decay=cfg.training.weight_decay, 
        fused=use_fused
    )
    
    # Prepare the learning rate scheduler
    if cfg.training.scheduler == 1:
        # Linear decay scheduler
        scheduler = LinearDecayLR(optimizer, cfg.training.epochs, decay_time=cfg.training.decay_time)
    elif cfg.training.scheduler == 2:
        # Cosine annealing with warmup
        scheduler = CosineWarmupScheduler(
            optimizer, 
            max_lr=cfg.training.lr, 
            warmup_epoch=1, 
            final_lr_factor=0.1, 
            total_epochs=cfg.training.epochs
        )
    else:
        # No scheduler
        scheduler = None
    
    return optimizer, scheduler

def override_cfg_with_args_and_calculate_config_values(cfg, args):
    cfg.training.epochs = args.epochs
    cfg.training.seed = args.seed

    # # Set model parameters based on dataset and arguments
    # cfg.model.n_inputs = train_ds.vector_size # COMMNENTED OUT AND EXPECTED TO BE SET CORRECTLY IN CONFIG ALREADY
    
    # Calculate number of latent dimensions using expansion factor
    cfg.model.n_latents = int(args.expansion_factor * cfg.model.n_inputs)
    logger.info(f"Expansion factor: {args.expansion_factor}, Latent dimensions: {cfg.model.n_latents}")
    
    # Extract l1 from ReLU if applied
    if args.model == "ReLUSAE" and "_" in args.activation:
        args.activation, sparse_weight = args.activation.split("_")
        cfg.loss.sparse_weight = float(f"0.{sparse_weight}")
        logger.info(f"Changing sparsity weight value to {cfg.loss.sparse_weight}")
        
    # Override activation if specified in arguments
    if args.activation:
        cfg.model.activation = args.activation
    
    # Configure Matryoshka SAE parameters if applicable
    if cfg.model.use_matryoshka:
        # Max nesting list
        if cfg.model.nesting_list > cfg.model.max_nesting:
            max_nesting = cfg.model.n_latents
        else:
            max_nesting = cfg.model.max_nesting
        
        # Generate nesting list if a single value was provided
        if isinstance(cfg.model.nesting_list, int):
            logger.info(f"Generating nesting list from {cfg.model.nesting_list} to {max_nesting}")
            start = [cfg.model.nesting_list]
            while start[-1] < max_nesting:
                new_k = start[-1] * 2
                if new_k > max_nesting:
                    break
                start.append(new_k)
            
            if max_nesting not in start:
                start.append(max_nesting)
            cfg.model.nesting_list = start
        
        # Set importance weights for different nesting levels
        if cfg.model.relative_importance == "RW":
            # Reverse weighting - higher weight for lower k values
            cfg.model.relative_importance = list(reversed(list(range(1, len(cfg.model.nesting_list)+1))))
        elif cfg.model.relative_importance == "UW":
            # Uniform weighting - equal weight for all k values
            cfg.model.relative_importance = [1.0] * len(cfg.model.nesting_list)
        logger.info(f"Using Matryoshka with nesting list: {cfg.model.nesting_list} and weighting function: {cfg.model.relative_importance}")
    elif cfg.model.use_ewgsae:
        # Max width list for EWGSAE is determined by ewg_sae_groups parameter
        max_width = cfg.model.n_latents   
        # Generate non-nested list if a single value was provided
        if isinstance(cfg.model.ewg_sae_groups, int):
            logger.info(f"Generating nesting list from {cfg.model.ewg_sae_groups} to {max_width}")
            base_size = cfg.model.ewg_sae_groups
            # create cumulative list like [64, 64 + 128, 64 + 128 + 256] until max_width is reached or exceeded
            start = []
            current_size = 0
            current_group = 0
            while current_size < max_width:
                current_group += 1
                current_size += base_size * (2 ** (current_group - 1))
                if current_size > max_width:
                    break
                start.append(current_size)
            if max_width not in start:
                start.append(max_width)
            cfg.model.ewg_sae_groups = start
            
    elif cfg.model.use_multisae:
        # Generate nesting list if a single value was provided

        if not isinstance(cfg.model.nesting_list, list):
            raise ValueError("nesting_list must be a list for MultiSAE")
        
        highest_nesting = max(cfg.model.nesting_list)
        if highest_nesting > cfg.model.n_latents:
            raise ValueError("Highest nesting level cannot be greater than number of latents")
        
        # Set importance weights for different nesting levels
        if cfg.model.relative_importance == "RW":
            raise ValueError("Relative Weighting not supported for MultiSAE as it the losses are computed independently.")
        elif cfg.model.relative_importance == "UW":
            # Uniform weighting - equal weight for all k values
            cfg.model.relative_importance = [1.0] * len(cfg.model.nesting_list)

        logger.info(f"Using Multi SAE with nesting list: {cfg.model.nesting_list} and weighting function: {cfg.model.relative_importance}")
    elif cfg.model.use_arch_matryoshka:
        # Set importance weights for different nesting levels
        if cfg.model.relative_importance == "RW":
            raise NotImplementedError("Relative weighting not implemented for Architectural Matryoshka SAE")
        elif cfg.model.relative_importance == "UW":
            # Uniform weighting - equal weight for all k values
            cfg.model.relative_importance = [1.0] * len(cfg.model.group_fractions)
        elif cfg.model.relative_importance == "FW":
            # Forward weighting - increasing weight for higher group fractions
            cfg.model.relative_importance = list(range(1, len(cfg.model.group_fractions) + 1))
        logger.info(f"Using Architectural Matryoshka with group fractions: {cfg.model.group_fractions} and weighting function: {cfg.model.relative_importance}")
    elif cfg.model.use_mpsae == True:
        logger.info(f"Using Matching Pursuit SAE")
    else:
        logger.info(f"Using standard SAE with {cfg.model.activation} activation")

def main(args):
    """
    Main training function for Sparse Autoencoders.
    
    This function handles the complete training pipeline:
    1. Setting up configuration based on model type and arguments
    2. Loading and preparing datasets
    3. Initializing the appropriate model (standard or Matryoshka SAE)
    4. Setting up loss function, optimizer, and learning rate scheduler
    5. Executing the training loop with periodic evaluation
    6. Tracking metrics including reconstruction quality, sparsity, and dead neurons
    7. Saving the trained model with relevant metadata
    
    Args:
        args (argparse.Namespace): Command line arguments from parse_args()
    """
    logger.info("Starting training with the following arguments:")
    logger.info(args)
    
    # Get configuration based on model type
    cfg = get_config(args.model)
    override_cfg_with_args_and_calculate_config_values(cfg, args)

    wandb_run = None
    try:
        wandb_config = {
            "args": _to_serializable(vars(args)),
            "training": _to_serializable(asdict(cfg.training)),
            "model": _to_serializable(asdict(cfg.model)),
            "loss": _to_serializable(asdict(cfg.loss)),
        }
        wandb_run = wandb.init(
            project="SAE-training",
            config=wandb_config,
        )
    except Exception as exc:
        logger.warning(f"W&B initialization failed. Continuing without W&B logging. Error: {exc}")
    
    
    # Set the random seed for reproducibility
    set_seed(cfg.training.seed)
    
    # Set the device (GPU/CPU)
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load datasets
    train_ds, eval_ds, eval_ds_second = load_datasets(args, cfg)
    # Prepare dataloaders
    train_loader, eval_loader, eval_loader_second = get_dataloaders(train_ds, eval_ds, eval_ds_second, cfg, args)
    
    # Calculate bias initialization (median or zero)
    logger.info(f"Calculating bias initialization with median: {cfg.training.bias_init_median}")
    bias_init = 0.0
    if cfg.training.bias_init_median:
        # Use geometric median of a subset of data points for robustness
        bias_init = geometric_median(train_ds, device=device, max_number=len(train_ds)//10)
    logger.info(f"Bias initialization: {bias_init}")
    
    # Initialize the appropriate model type
    if cfg.model.use_matryoshka:
        model = MatryoshkaAutoencoder(bias_init=bias_init, **asdict(cfg.model))
    elif cfg.model.use_arch_matryoshka:
        model = ArchitecturalMatryoshkaSAE(bias_init=bias_init, **asdict(cfg.model))
    elif cfg.model.use_mpsae:
        model = MatchingPursuitSAE(bias_init=bias_init, **asdict(cfg.model))
    elif cfg.model.use_multisae:
        model = MultiSAE(bias_init=bias_init, **asdict(cfg.model))
    elif cfg.model.use_ewgsae:
        model = EWGSAE(bias_init=bias_init, **asdict(cfg.model))
    else:
        model = Autoencoder(bias_init=bias_init, **asdict(cfg.model))
    model = model.to(device)
    
    # Prepare loss function
    # Use zeros or calculate mean from dataset depending on config
    mean_input = torch.zeros((cfg.model.n_inputs,), dtype=cfg.training.dtype)
    if not cfg.training.mean_center:
        mean_input = calculate_vector_mean(train_ds, num_workers=cfg.training.num_workers)
    
    mean_input = mean_input.to(device)
    loss_fn = SAELoss(
        reconstruction_loss=cfg.loss.reconstruction_loss,
        sparse_loss=cfg.loss.sparse_loss,
        sparse_weight=cfg.loss.sparse_weight,
        mean_input=mean_input,
        ewg_sae_groups=cfg.model.ewg_sae_groups if cfg.model.use_ewgsae else None,
    )
    base_sparse_weight = loss_fn.sparse_weight

    optimizer, scheduler = load_scheduler_and_optimizer(model, cfg, device)
    
    # Training loop
    global_step = 0
    numb_of_dead_neurons = 0
    dead_neurons = []
    
    for epoch in range(cfg.training.epochs):
        model.train()
        logger.info(f"Epoch {epoch+1}/{cfg.training.epochs}")
        
        # Training loop for current epoch
        for step, embeddings in enumerate(tqdm(train_loader, desc="Training")):
            optimizer.zero_grad()
            global_step += 1
            embeddings = embeddings.to(device)
            # Forward pass through model
            recons_sparse, repr_sparse, recons_all, repr_all = model(embeddings)
            
            # Postprocess return values of forward and compute loss based on model type
            if cfg.model.use_matryoshka:
                loss, recon_loss, sparse_loss = act_matryoshka_loss_helper(
                    loss_fn, recons_sparse, recons_all, repr_sparse, repr_all, embeddings, model, device
                    )
                # Use first nesting level for metrics
                repr_sparse = repr_sparse[0]
                recons_sparse = recons_sparse[0]
            elif cfg.model.use_multisae:
                loss, recon_loss, sparse_loss = multisae_loss_helper(
                    loss_fn, recons_sparse, recons_all, repr_sparse, repr_all, embeddings, model, device
                    )
                # Use first nesting level for metrics
                first_sae_id = 0
                recons_sparse = recons_sparse[first_sae_id]
                recons_all = recons_all[first_sae_id]
                repr_sparse = repr_sparse[first_sae_id]
                repr_all = repr_all[first_sae_id]
            elif cfg.model.use_arch_matryoshka:
                recons_sparse_per_prefix = recons_sparse # just for naming consistency
                loss, recon_loss, sparse_loss = arch_matryoshka_loss_helper(
                    loss_fn, recons_sparse_per_prefix, recons_all, repr_sparse, repr_all, embeddings, model, device
                    )
                # Modify variables for metric calculation and comparability with other models
                recons_sparse = recons_sparse_per_prefix[-1] # use reconstructions of prefix with all latents
            elif cfg.model.use_mpsae:
                loss, recon_loss, sparse_loss = loss_fn(recons_sparse, embeddings, repr_sparse)
            else:
                if cfg.model.use_ewgsae and cfg.training.ewg_warmup_steps > 0:
                    warmup_factor = min(1.0, global_step / cfg.training.ewg_warmup_steps)
                    loss_fn.sparse_weight = base_sparse_weight * warmup_factor
                    # model._grad_proj_warmup_factor = warmup_factor # No longer used
                loss, recon_loss, sparse_loss = loss_fn(recons_sparse, embeddings, repr_sparse)
            
            # Backpropagation
            loss.backward()
            
            # Weight normalization and gradient projection
            model.scale_to_unit_norm()
            model.project_grads_decode()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.clip_grad)
            
            # Update model parameters
            optimizer.step()
            
            # Detach tensors for metric calculation
            recons_sparse, recons_all, embeddings = recons_sparse.detach(), recons_all.detach(), embeddings.detach()

            eval_metrics = eval_during_train(
                repr_sparse, repr_all, recons_sparse, recons_all, embeddings, model
            )

            # Check for dead neurons periodically
            if global_step % cfg.training.check_dead == 0:
                activations = model.get_and_reset_stats()
                dead_neurons = identify_dead_neurons(activations).numpy().tolist()
                numb_of_dead_neurons = len(dead_neurons)
                logger.info(f"Number of dead neurons: {numb_of_dead_neurons}")
            
            # Log metrics periodically
            if global_step % cfg.training.print_freq == 0:
                display_eval_metrics(epoch, step, loss, recon_loss, sparse_loss, eval_metrics)
                if wandb_run is not None:
                    wandb_train_metrics = {
                        "train/loss": loss.item(),
                        "train/recon_loss": recon_loss.item(),
                        "train/sparse_loss": sparse_loss.item(),
                        "train/epoch": epoch + 1,
                        "train/step": step,
                        "train/global_step": global_step,
                    }
                    wandb_train_metrics.update(
                        {f"train/{k}": _to_log_scalar(v) for k, v in eval_metrics.items()}
                    )
                    wandb.log(wandb_train_metrics, step=global_step)

        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        # Log epoch summary
        lr_rate = scheduler.get_last_lr()[0] if scheduler else cfg.training.lr
        logger.info(f"Epoch: {epoch+1}, Learning Rate: {lr_rate:.6f}, Loss: {loss.item():.6f}, Recon Loss: {recon_loss.item():.6f}, Sparse Loss: {sparse_loss.item():.6f}, Dead neurons: {numb_of_dead_neurons}")
        if wandb_run is not None:
            wandb.log({
                "epoch/lr": lr_rate,
                "epoch/loss": loss.item(),
                "epoch/recon_loss": recon_loss.item(),
                "epoch/sparse_loss": sparse_loss.item(),
                "epoch/dead_neurons": numb_of_dead_neurons,
            }, step=global_step)
    
        # Evaluate the model on the validation set
        eval_metrics_primary = eval(model, eval_loader, loss_fn, device, cfg)
        if wandb_run is not None:
            wandb.log(
                {f"val/{k}": _to_log_scalar(v) for k, v in eval_metrics_primary.items()},
                step=global_step,
            )

        if args.dataset_second_modality is not None:
            # Evaluate on the second modality dataset
            eval(model, eval_loader_second, loss_fn, device, cfg)
    

        
    # Save the trained model
    save_path = save_model(model, cfg, train_ds, args)
    if wandb_run is not None:
        model_artifact = wandb.Artifact(name=f"{args.model}-checkpoint", type="model")
        model_artifact.add_file(save_path)
        wandb.log_artifact(model_artifact)
        wandb.finish()
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
