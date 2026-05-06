from sae import EWGSAE, ArchitecturalMatryoshkaSAE, Autoencoder, MatchingPursuitSAE, MatryoshkaAutoencoder, MultiSAE
import torch
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    'Autoencoder': Autoencoder,
    'MatryoshkaAutoencoder': MatryoshkaAutoencoder,
    'ArchitecturalMatryoshkaSAE': ArchitecturalMatryoshkaSAE,
    'MatchingPursuitSAE': MatchingPursuitSAE,
    'MultiSAE': MultiSAE,
    'EWGSAE': EWGSAE,
}

def get_model_class(class_name: str):
    """
    Get model class from registry.

    Args:
        class_name (str): Name of the model class

    Returns:
        class: The corresponding model class

    Raises:
        ValueError: If class_name is not in registry
    """
    if class_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model class: {class_name}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[class_name]


def load_model(path):
    """
    Load a saved sparse autoencoder model from a checkpoint.

    Args:
        path (str): Path to the saved model file (.pth or .pt)

    Returns:
        tuple: (model, mean_center, scaling_factor, target_norm)
    """
    import logging
    logger = logging.getLogger(__name__)

    # Load checkpoint
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(path, map_location=device)

    # Check version
    version = checkpoint.get('checkpoint_version', '0.0')

    if version == '0.0':
        # Legacy format (old checkpoints)
        logger.warning(f"Loading legacy checkpoint from {path}")
        return _load_legacy_checkpoint(path)

    # New format (v1.0+)
    logger.info(f"Loading checkpoint v{version} from {path}")

    # Get model class and config
    model_class_name = checkpoint['model_class']
    model_config = checkpoint['model_config']

    logger.info(f"  Model class: {model_class_name}")
    logger.info(f"  Config: {model_config}")

    # Get model class from registry
    ModelClass = get_model_class(model_class_name)

    # Prepare config for initialization
    init_config = model_config.copy()

    # Handle bias_init: use actual tensor if available, else use config value
    if checkpoint.get('bias_init_tensor') is not None:
        init_config['bias_init'] = checkpoint['bias_init_tensor']
    # else: use the float value from config (already in init_config)

    # Instantiate model
    try:
        model = ModelClass(**init_config)
    except TypeError as e:
        logger.error(f"Failed to instantiate {model_class_name} with config {init_config}")
        raise e

    # Load state dict
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    except RuntimeError as e:
        logger.warning(f"Strict loading failed: {e}")
        result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        # Log what was missed
        logger.info(f"Missing keys: {result.missing_keys}")
        logger.info(f"Unexpected keys: {result.unexpected_keys}")

    # Move to device
    model = model.to(device)

    logger.info(f"Successfully loaded {model_class_name}")
    preprocessing = checkpoint['preprocessing']
    return model, preprocessing["mean_center"], preprocessing["scaling_factor"], preprocessing["target_norm"]


def _load_legacy_checkpoint(path):
    """
    Handle old checkpoint format (pre-v1.0).

    This parses the filename to extract config parameters.
    Provides backward compatibility with existing saved models.

    Args:
        path (str): Path to checkpoint file

    Returns:
        Same as load_model()
    """
    import logging
    logger = logging.getLogger(__name__)

    logger.warning("Using legacy filename parsing - consider re-saving in new format")

    try:
        # Extract configuration from filename
        path_head = path.split("/")[-1]
        path_name = path_head[:path_head.find(".pt")]
        path_name_spited = path_name.split("_")
        
        n_latents = int(path_name_spited.pop(0))
        n_inputs = int(path_name_spited.pop(0))
        activation = path_name_spited.pop(0)
        if "TopK" in activation:
            activation += "_" + path_name_spited.pop(0)
        elif "ReLU" == activation:
            path_name_spited.pop(0)
        if "UW" in path_name_spited[0] or "RW" in path_name_spited[0] or "FW" in path_name_spited[0]:
            path_name_spited.pop(0)
        tied = False if path_name_spited.pop(0) == "False" else True
        normalize = False if path_name_spited.pop(0) == "False" else True
        latent_soft_cap = float(path_name_spited.pop(0))
        
        # Create and load the model
        model = Autoencoder(n_latents, n_inputs, activation, tied=tied, normalize=normalize, latent_soft_cap=latent_soft_cap)
        model_state_dict = torch.load(path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(model_state_dict['model'])
        mean_center = model_state_dict['mean_center']
        scaling_factor = model_state_dict['scaling_factor']
        target_norm = model_state_dict['target_norm']
        return model, mean_center, scaling_factor, target_norm

    except Exception as e:
        logger.error(f"Failed to parse legacy checkpoint {path}: {e}")
        raise ValueError(f"Could not load legacy checkpoint. Consider manual migration.") from e


def generate_model_filename(model, cfg, args, train_ds):
    """
    Generate concise, informative filename.

    Format: {model_type}_{key_params}_{dataset}_{timestamp}.pth
    Example: msae_rw_k64_x8_cc3m_vit_20231219_143022.pth
    """
    from datetime import datetime

    # Model type mapping
    model_type_map = {
        'ReLUSAE': 'relu',
        'TopKSAE': 'topk',
        'BatchTopKSAE': 'btopk',
        'MSAE_UW': 'msae_uw',
        'MSAE_RW': 'msae_rw',
        'ArchMSAE_UW': 'archmsae_uw',
        'ArchMSAE_FW': 'archmsae_fw',
        'MPSAE': 'mpsae',
        'MultiSAE': 'multisae',
    }
    model_type = model_type_map.get(args.model, args.model.lower())

    # Key parameters
    parts = [model_type]

    # Add activation info
    activation = cfg.model.activation
    if 'TopK' in activation and '_' in activation:
        # Extract k value: "TopKReLU_64" -> "k64"
        k_val = activation.split('_')[-1]
        parts.append(f'k{k_val}')

    # Expansion factor
    parts.append(f'x{int(args.expansion_factor)}')

    # Dataset identifier (simplified)
    dataset_name = args.dataset_train.split('/')[-1]
    # Extract: cc3m, ViT-L~14 -> "cc3m_vit"
    if 'cc3m' in dataset_name.lower():
        parts.append('cc3m')
    elif 'imagenet' in dataset_name.lower():
        parts.append('imagenet')

    if 'vit' in dataset_name.lower():
        parts.append('vit')
    elif 'dinov2' in dataset_name.lower():
        parts.append('dinov2')


    # Timestamp for uniqueness
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    parts.append(timestamp)

    filename = '_'.join(parts) + '.pth'
    return filename


def save_model(model, cfg, train_ds, args, save_dir="saved_models"):
    """
    Save model with complete config embedded in checkpoint.

    Checkpoint structure:
    {
        'model_state_dict': OrderedDict(...),
        'model_class': 'MatryoshkaAutoencoder',
        'model_config': {...},
        'preprocessing': {...},
        'bias_init_tensor': tensor(...) or None,
        'training_metadata': {...},
        'checkpoint_version': '1.0',
    }
    """
    import os
    from datetime import datetime

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Generate concise filename
    filename = generate_model_filename(model, cfg, args, train_ds)
    save_path = os.path.join(save_dir, filename)

    # Get model config (stored in __init__)
    if not hasattr(model, '_init_config'):
        raise ValueError(
            f"Model {type(model).__name__} does not have _init_config. "
            "Make sure all model classes store config in __init__."
        )

    model_config = model._init_config.copy()

    # Prepare checkpoint
    checkpoint = {
        # Essential for reconstruction
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'model_config': model_config,

        # Preprocessing (essential for inference)
        'preprocessing': {
            'mean_center': train_ds.mean,
            'scaling_factor': train_ds.scaling_factor,
            'target_norm': train_ds.target_norm,
        },

        # Store actual bias_init tensor if it was computed
        'bias_init_tensor': model.pre_bias.data.clone() if isinstance(model.bias_init, torch.Tensor) else None,

        # Metadata for tracking (not needed for loading)
        'training_metadata': {
            'command_line_args': vars(args),
            'final_epoch': cfg.training.epochs,
            'dataset_train': args.dataset_train,
            'dataset_train_size': len(train_ds),
            'dataset_test': args.dataset_test,
            'save_timestamp': datetime.now().isoformat(),
            'expansion_factor': args.expansion_factor,
            'sparse_weight': cfg.loss.sparse_weight,
            'seed': cfg.training.seed,
        },

        # Version for backward compatibility
        'checkpoint_version': '1.0',
    }

    # Save
    torch.save(checkpoint, save_path)

    logger.info(f"Model saved to {save_path}")
    logger.info(f"  Model class: {model.__class__.__name__}")
    logger.info(f"  Config: {model_config}")

    return save_path