import os
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from sae_model_loading_and_saving import load_model
from sae import EWGSAE
from valuable_notebook_code_snippets import load_clip_embeddings_memmap, load_precomputed_sae_representations_memmap
from utils import SAEDataset, set_seed, get_device
from metrics import (
    explained_variance_full,
    normalized_mean_absolute_error,
    l0_messure,
    cknna
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the representation extraction and evaluation script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Extract and evaluate representations from Sparse Autoencoder models")
    parser.add_argument("-m", "--model", type=str, required=True, 
                        help="Path to the trained model file (.pt)")
    parser.add_argument("-d", "--data", type=str, required=True, 
                        help="Path to the dataset file (.npy)")
    parser.add_argument("-b", "--batch-size", type=int, default=10000, 
                        help="Batch size for processing data")
    parser.add_argument("-o", "--output-path", type=str, default=".", 
                        help="Directory path to save extracted representations")
    parser.add_argument("-s", "--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--subset-size", type=str, default="",
                        help="Size of the random subset of the dataset to use")
    parser.add_argument("--no-save", action="store_true",
                        help="If set, do not save the extracted representations to disk")
    parser.add_argument("--save-metrics-csv", action="store_true",
                        help="If set, save evaluation metrics to a CSV file")

    return parser.parse_args()



def construct_output_filename(repr_file_name, embedding_type, num_data_points, embedding_dim):
    return f"{repr_file_name}_{embedding_type}_{num_data_points}_{embedding_dim}.npy"

def construct_output_path_prefix(output_path, model_path, data_path):
    model_file_name = model_path.split("/")[-1].replace(".pt","")
    data_file_name = data_path.split("/")[-1].replace(".npy","")
    repr_path_prefix = os.path.join(output_path, f"{data_file_name}_{model_file_name}")
    return repr_path_prefix


def evaluate_representations(representations_memmap, activations_memmap, outputs_memmap,):
    representations_memmap = load_clip_embeddings_memmap(representations_memmap)
    activations_memmap = load_precomputed_sae_representations_memmap(activations_memmap)
    outputs_memmap = load_clip_embeddings_memmap(outputs_memmap)

    l0 = []
    mse = []
    mae = []
    fvu = []
    cs = []
    cknnas = []
    sparse_l0 = []
    sparse_mse = []
    sparse_mae = []
    sparse_fvu = []
    sparse_cs = []
    sparse_cknnas = []
    dead_neurons_count = None
    dead_neurons_count_sparse = None
    
    # Process data in batches
    BATCH_SIZE = 10000
    for idx in tqdm(range((activations_memmap.shape[0] + BATCH_SIZE - 1) // BATCH_SIZE), desc="Evaluating representations"): 
        if idx * BATCH_SIZE > 100000:
            break
        
        start = BATCH_SIZE * idx
        end = min(start + BATCH_SIZE, outputs_memmap.shape[0])

        batch = torch.tensor(representations_memmap[start:end])
        outputs = torch.tensor(outputs_memmap[start:end])
        sae_representations = torch.tensor(activations_memmap[start:end])

        # Calculate and collect metrics
        fvu.append(explained_variance_full(batch, outputs))
        mse.append(torch.nn.functional.mse_loss(batch, outputs, reduction='none').mean(dim=1))
        mae.append(normalized_mean_absolute_error(batch, outputs))
        cs.append(torch.nn.functional.cosine_similarity(batch, outputs))
        l0.append(l0_messure(sae_representations))
        # Only calculate the cknna if it even to the number of the batch
        if batch.shape[0] == BATCH_SIZE:
            cknnas.append(cknna(batch, sae_representations, topk=10))

        # Track neurons that are activated at least once
        if dead_neurons_count is None:
            dead_neurons_count = (sae_representations > 0).sum(dim=0).cpu().long()
        else:
            dead_neurons_count += (sae_representations > 0).sum(dim=0).cpu().long()


    # Aggregate metrics across all batches
    mse = torch.cat(mse, dim=0).cpu().numpy()
    mae = torch.cat(mae, dim=0).cpu().numpy()
    cs = torch.cat(cs, dim=0).cpu().numpy()
    l0 = torch.cat(l0, dim=0).cpu().numpy()
    fvu = torch.cat(fvu, dim=0).cpu().numpy()
    cknnas = np.array(cknnas)
    
    # Count neurons that were never activated
    number_of_dead_neurons = torch.where(dead_neurons_count == 0)[0].shape[0]
    # Log final metrics
    logger.info(f"Fraction of Variance Unexplained (FVU): {np.mean(fvu)} +/- {np.std(fvu)}")
    logger.info(f"MSE: {np.mean(mse)} +/- {np.std(mse)}")
    logger.info(f"Normalized MAE: {np.mean(mae)} +/- {np.std(mae)}")
    logger.info(f"Cosine similarity: {np.mean(cs)} +/- {np.std(cs)}")
    logger.info(f"L0 messure: {np.mean(l0)} +/- {np.std(l0)}")
    logger.info(f"CKNNA: {np.mean(cknnas)} +/- {np.std(cknnas)}")
    logger.info(f"Number of dead neurons: {number_of_dead_neurons}")


    metrics_dict = {
        'fvu': fvu,
        'mse': mse,
        'mae': mae,
        'cosine_similarity': cs,
        'l0': l0,
        'cknna': cknnas,
        "sparse_fvu": fvu,
        "sparse_mse": mse,
        "sparse_mae": mae,
        "sparse_cosine_similarity": cs,
        "sparse_l0": l0,
    }
    return metrics_dict

def get_representation(model, dataset, repr_path_prefix, batch_size, save_outputs=True):
    """
    Extract representations from the model for the given dataset and evaluate model performance.
    
    Extracts both output reconstructions and latent representations from the model,
    saves them to disk as memory-mapped files, and computes various performance metrics.
    
    Args:
        model: The Sparse Autoencoder model to evaluate
        dataset: Dataset to process
        repr_path_prefix (str): Base path prefix for saving representations
        batch_size (int): Number of samples to process at once
        save_outputs (bool): Whether to save the extracted representations to disk
        
    Metrics computed:
        - Fraction of Variance Unexplained (FVU) using normalized MSE
        - Normalized Mean Absolute Error (MAE)
        - Cosine similarity between inputs and outputs
        - L0 measure (average number of active neurons per sample)
        - CKNNA (Cumulative k-Nearest Neighbor Accuracy)
        - Number of dead neurons (neurons that never activate)
    """
    device = get_device()
    logger.info(f"Using device: {device}")
    model.eval()
    model.to(device)
    with torch.no_grad():
        if save_outputs:
            # Prepare memory-mapped file for output reconstructions
            repr_path_output = construct_output_filename(repr_path_prefix, "output", len(dataset), model.n_inputs)
            memmap_output = np.memmap(repr_path_output, dtype='float32', mode='w+', 
                                    shape=(len(dataset), model.n_inputs))
            logger.info(f"Data output with shape {memmap_output.shape} will be saved to {repr_path_output}")

            # Prepare memory-mapped file for latent representations
            repr_path_repr = construct_output_filename(repr_path_prefix, "repr", len(dataset), model.n_latents)
            memmap_repr = np.memmap(repr_path_repr, dtype='float32', mode='w+', 
                                    shape=(len(dataset), model.n_latents))
            logger.info(f"Data repr with shape {memmap_repr.shape} will be saved to {repr_path_repr}")

        # Create dataloader for batch processing
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                               shuffle=False, num_workers=0)
        
        # Lists to collect metrics for each batch
        l0 = []
        mse = []
        mae = []
        fvu = []
        cs = []
        cknnas = []
        sparse_l0 = []
        sparse_mse = []
        sparse_mae = []
        sparse_fvu = []
        sparse_cs = []
        sparse_cknnas = []
        dead_neurons_count = None
        dead_neurons_count_sparse = None
        
        is_ewgsae = isinstance(model, EWGSAE)

        # Process data in batches
        for idx, batch in enumerate(tqdm(dataloader, desc="Extracting representations")):
            start = batch_size * idx
            end = start + batch.shape[0]
            batch = batch.to(device)

            # Forward pass through the model
            with torch.no_grad():
                forward_output = model(batch)
                sparse_outputs, sparse_representation, outputs, representations = model.get_primary_outputs(forward_output)
                # EWGSAE has no inherent sparsity gate — its sparse_* outputs equal
                # the full ones. For evaluation we replace them with a hard top-64
                # filter on the activations so the sparse_* metrics report a
                # meaningful sparsity probe. Eval-only; never seen by training.
                if is_ewgsae:
                    sparse_outputs, sparse_representation = model.forward_topk_eval(batch, k=64)

            # Post-process outputs and batch
            # Handle case where dataset might be a Subset
            base_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
            batch = base_dataset.unprocess_data(batch.cpu()).to(device)
            outputs = base_dataset.unprocess_data(outputs.cpu()).to(device)
            sparse_outputs = base_dataset.unprocess_data(sparse_outputs.cpu()).to(device)
            
            if save_outputs:
                # Save the outputs and representations to the memmap files
                outputs_numpy = outputs.cpu().numpy()
                memmap_output[start:end] = outputs_numpy
                memmap_output.flush()
                representations_numpy = representations.cpu().numpy()
                memmap_repr[start:end] = representations_numpy
                memmap_repr.flush()

            # Calculate and collect metrics
            fvu.append(explained_variance_full(batch, outputs))
            mse.append(torch.nn.functional.mse_loss(batch, outputs, reduction='none').mean(dim=1))
            mae.append(normalized_mean_absolute_error(batch, outputs))
            cs.append(torch.nn.functional.cosine_similarity(batch, outputs))
            l0.append(l0_messure(representations))
            # Only calculate the cknna if it even to the number of the batch
            if batch.shape[0] == batch_size:
                cknnas.append(cknna(batch, representations, topk=10))
            
            sparse_fvu.append(explained_variance_full(batch, sparse_outputs))
            sparse_mse.append(torch.nn.functional.mse_loss(batch, sparse_outputs, reduction='none').mean(dim=1))
            sparse_mae.append(normalized_mean_absolute_error(batch, sparse_outputs))
            sparse_cs.append(torch.nn.functional.cosine_similarity(batch, sparse_outputs))
            sparse_l0.append(l0_messure(sparse_representation))
            # Only calculate the cknna if it even to the number of the batch
            if batch.shape[0] == batch_size:
                sparse_cknnas.append(cknna(batch, sparse_representation, topk=10))
            
            # Track neurons that are activated at least once
            if dead_neurons_count is None:
                dead_neurons_count = (representations != 0).sum(dim=0).cpu().long()
            else:
                dead_neurons_count += (representations != 0).sum(dim=0).cpu().long()

            if dead_neurons_count_sparse is None:
                dead_neurons_count_sparse = (sparse_representation != 0).sum(dim=0).cpu().long()
            else:
                dead_neurons_count_sparse += (sparse_representation != 0).sum(dim=0).cpu().long()

        # Aggregate metrics across all batches
        mse = torch.cat(mse, dim=0).cpu().numpy()
        mae = torch.cat(mae, dim=0).cpu().numpy()
        cs = torch.cat(cs, dim=0).cpu().numpy()
        l0 = torch.cat(l0, dim=0).cpu().numpy()
        fvu = torch.cat(fvu, dim=0).cpu().numpy()
        cknnas = np.array(cknnas)
        sparse_mse = torch.cat(sparse_mse, dim=0).cpu().numpy()
        sparse_mae = torch.cat(sparse_mae, dim=0).cpu().numpy()
        sparse_cs = torch.cat(sparse_cs, dim=0).cpu().numpy()
        sparse_l0 = torch.cat(sparse_l0, dim=0).cpu().numpy()
        sparse_fvu = torch.cat(sparse_fvu, dim=0).cpu().numpy()
        sparse_cknnas = np.array(sparse_cknnas)
        
        # Count neurons that were never activated
        number_of_dead_neurons = torch.where(dead_neurons_count == 0)[0].shape[0]
        number_of_dead_neurons_sparse = torch.where(dead_neurons_count_sparse == 0)[0].shape[0]

        # Log final metrics
        logger.info(f"Fraction of Variance Unexplained (FVU): {np.mean(fvu)} +/- {np.std(fvu)}")
        logger.info(f"MSE: {np.mean(mse)} +/- {np.std(mse)}")
        logger.info(f"Normalized MAE: {np.mean(mae)} +/- {np.std(mae)}")
        logger.info(f"Cosine similarity: {np.mean(cs)} +/- {np.std(cs)}")
        logger.info(f"L0 messure: {np.mean(l0)} +/- {np.std(l0)}")
        logger.info(f"CKNNA: {np.mean(cknnas)} +/- {np.std(cknnas)}")
        logger.info(f"Number of dead neurons: {number_of_dead_neurons}")
        logger.info(f"\nSparse Fraction of Variance Unexplained (FVU): {np.mean(sparse_fvu)} +/- {np.std(sparse_fvu)}")
        logger.info(f"Sparse MSE: {np.mean(sparse_mse)} +/- {np.std(sparse_mse)}")
        logger.info(f"Sparse Normalized MAE: {np.mean(sparse_mae)} +/- {np.std(sparse_mae)}")
        logger.info(f"Sparse Cosine similarity: {np.mean(sparse_cs)} +/- {np.std(sparse_cs)}")
        logger.info(f"Sparse L0 messure: {np.mean(sparse_l0)} +/- {np.std(sparse_l0)}")
        logger.info(f"Sparse CKNNA: {np.mean(sparse_cknnas)} +/- {np.std(sparse_cknnas)}")
        logger.info(f"Number of sparse dead neurons: {number_of_dead_neurons_sparse}")


        metrics_dict = {
            'fvu': fvu,
            'mse': mse,
            'mae': mae,
            'cosine_similarity': cs,
            'l0': l0,
            'cknna': cknnas,
            'sparse_fvu': sparse_fvu,
            'sparse_mse': sparse_mse,
            'sparse_mae': sparse_mae,
            'sparse_cosine_similarity': sparse_cs,
            'sparse_l0': sparse_l0,
            'sparse_cknna': sparse_cknnas,
        }
        return metrics_dict

import pandas as pd
from datetime import datetime

def save_metrics_to_csv(metrics_dict, model_path, data_path, num_samples, seed):
    """
    Save evaluation metrics to a CSV file, appending to existing file if present.
    
    Args:
        metrics_dict (dict): Dictionary containing metric arrays with keys like 'fvu', 'mae', etc.
        model_path (str): Path to the model file
        data_path (str): Path to the data file
        num_samples (int): Number of samples evaluated
        seed (int): Random seed used
    """
    # Extract model and data names
    model_name = os.path.basename(model_path).replace('.pth', '')
    data_name = os.path.basename(data_path).replace('.npy', '')

    output_csv_path = get_metrics_output_path(model_name, data_name)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    # Create a single row with all metrics
    row_data = {
        'timestamp': datetime.now().isoformat(),
        'model_name': model_name,
        'model_path': model_path,
        'data_name': data_name,
        'data_path': data_path,
        'num_samples': num_samples,
        'seed': seed,
    }
    
    # Add mean and std for each metric
    for metric_name, metric_values in metrics_dict.items():
        row_data[f'{metric_name}_mean'] = np.mean(metric_values)
        row_data[f'{metric_name}_std'] = np.std(metric_values)
    
    # Convert to DataFrame
    df_new = pd.DataFrame([row_data])
    
    # # Append to existing CSV or create new one
    # if os.path.exists(output_csv_path):
    #     df_existing = pd.read_csv(output_csv_path)
    #     df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    #     df_combined.to_csv(output_csv_path, index=False)
    #     logger.info(f"Metrics appended to {output_csv_path}")
    # else:
    df_new.to_csv(output_csv_path, index=False)
    logger.info(f"Metrics saved to new file {output_csv_path}")

def get_metrics_output_path(model_name, data_name):
    path = os.path.join("output", model_name, "standard_evals", f"{data_name}_metrics.csv")
    return path
def main(args):
    """
    Main function to load model and dataset, then extract and evaluate representations.
    
    Args:
        args (argparse.Namespace): Command line arguments.
    """
    # # Set random seed for reproducibility
    set_seed(args.seed)

    if "hsae" in args.model:
        # # representations_memmap_path = "/home/ngrandien/thesis/MSAE/data/cc3m_ViT-L~14_train_image_2820737_768.npy"
        # # activations_memmap_path = "/home/ngrandien/thesis/MSAE/sae_activations/cc3m_ViT-L~14_train_image_2820737_768_hsae_361_16_6144_cc3m_vith_repr_2820737_6144.npy"
        # # outputs_memmap_path = "/home/ngrandien/thesis/MSAE/sae_activations/cc3m_ViT-L~14_train_image_2820737_768_hsae_361_16_6144_cc3m_vith_output_2820737_768.npy"

        # # representations_memmap_path = "/home/ngrandien/thesis/MSAE/data/cc3m_ViT-L~14_validation_image_13002_768.npy"
        # # activations_memmap_path = "/home/ngrandien/thesis/MSAE/sae_activations/cc3m_ViT-L~14_validation_image_13002_768_hsae_361_16_6144_cc3m_vith_repr_13002_6144.npy"
        # # outputs_memmap_path = "/home/ngrandien/thesis/MSAE/sae_activations/cc3m_ViT-L~14_validation_image_13002_768_hsae_361_16_6144_cc3m_vith_output_13002_768.npy"

        # # representations_memmap_path = "/home/ngrandien/thesis/MSAE/data/cc3m_dinov2-base_train_image_2820737_768.npy"
        # # outputs_memmap_path = "/home/ngrandien/thesis/MSAE/sae_activations/cc3m_dinov2-base_train_image_2820737_768_hsae_361_16_6144_cc3m_dinov2h_output_2820737_768.npy"
        # # activations_memmap_path = "/home/ngrandien/thesis/MSAE/sae_activations/cc3m_dinov2-base_train_image_2820737_768_hsae_361_16_6144_cc3m_dinov2h_repr_2820737_6144.npy"

        # representations_memmap_path = "/home/ngrandien/thesis/MSAE/data/cc3m_dinov2-base_validation_image_13002_768.npy"
        # outputs_memmap_path = "/home/ngrandien/thesis/MSAE/sae_activations/cc3m_dinov2-base_validation_image_13002_768_hsae_361_16_6144_cc3m_dinov2h_output_13002_768.npy"
        # activations_memmap_path = "/home/ngrandien/thesis/MSAE/sae_activations/cc3m_dinov2-base_validation_image_13002_768_hsae_361_16_6144_cc3m_dinov2h_repr_13002_6144.npy"

        model_str = os.path.basename(args.model).replace(".pth", "")
        foundation_model = "ViT-L~14" if "vit" in args.model else "dinov2-base"
        split = "validation"
        size = "13002"
        representations_memmap_path = f"/home/ngrandien/thesis/MSAE/data/cc3m_{foundation_model}_{split}_image_{size}_768.npy"
        activations_memmap_path = f"/home/ngrandien/thesis/MSAE/sae_activations/cc3m_{foundation_model}_{split}_image_{size}_768_{model_str}h_repr_{size}_6144.npy"
        outputs_memmap_path = f"/home/ngrandien/thesis/MSAE/sae_activations/cc3m_{foundation_model}_{split}_image_{size}_768_{model_str}h_output_{size}_768.npy"


        metrics_dict = evaluate_representations(
            representations_memmap_path,
            activations_memmap_path,
            outputs_memmap_path,
        )
        assert os.path.basename(args.data)in representations_memmap_path, "Data path in arguments does not match the data used for evaluation"
        if args.save_metrics_csv:
            save_metrics_to_csv(metrics_dict, args.model, args.data, size, args.seed)
        return
    
    #Load the trained model
    model, mean_center, scaling_factor, target_norm = load_model(args.model)
    logger.info("Model loaded")
    
    # Load the dataset with appropriate preprocessing
    if ("text" in args.model and "text" in args.data) or ("image" in args.model and "image" in args.data):
        logger.info("Using model mean and scalling factor")    
        dataset = SAEDataset(args.data)
        dataset.mean = mean_center.cpu()
        dataset.scaling_factor = scaling_factor
    else:    
        logger.info("Computing mean and scalling factor")    
        dataset = SAEDataset(args.data, mean_center=True if mean_center.sum() != 0.0 else False, target_norm=target_norm)
        
    logger.info(f"Dataset loaded with length: {len(dataset)}")
    logger.info(f"Dataset mean center: {dataset.mean.mean()}, Scaling factor: {dataset.scaling_factor} with target norm {dataset.target_norm}")

    # Apply subset if requested
    if args.subset_size:
        subset_size = int(args.subset_size)
        if subset_size >= len(dataset):
            logger.warning(f"Requested subset size {subset_size} >= dataset size {len(dataset)}. Using full dataset.")
        else:
            indices = torch.arange(subset_size)
            dataset = torch.utils.data.Subset(dataset, indices)
            logger.info(f"Using prefix subset of {subset_size} samples from {len(dataset.dataset)} total samples")

    # Construct output filename from model and data names
    repr_path_prefix = construct_output_path_prefix(args.output_path, args.model, args.data)
    # Extract representations and compute metrics
    metrics_dict = get_representation(model, dataset, repr_path_prefix, args.batch_size, save_outputs=not args.no_save)
    # Save metrics to CSV if requested
    if args.save_metrics_csv:
        save_metrics_to_csv(metrics_dict, args.model, args.data, len(dataset), args.seed)

if __name__ == "__main__":
    args = parse_args()
    main(args)