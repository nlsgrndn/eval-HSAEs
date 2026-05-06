import os
import warnings
import torch
import argparse
import logging
import numpy as np
from tqdm import tqdm
from transformers import AutoImageProcessor, Dinov2Model

import clip
from torch.utils.data import Dataset
from actmsae_repo_datasets import CelebAMy, VocabDataset, CC3MDataset, ImageNetDataset
from my_datasets.cc3m_dataset import CC3MLocalDataset
from my_datasets.imagenet1k_local import ImageNet1kLocalDataset
from utils import get_device, set_seed



"""
CLIP Embedding Extraction Utility

This script extracts embeddings from various datasets using the CLIP model.
It supports multiple datasets and CLIP model variants, and saves the extracted
embeddings as memory-mapped numpy arrays for efficient storage and access.
"""

# Supported CLIP model variants
SUPPORTED_MODELS = [
    "ViT-B~32",
    "ViT-B~16",
    "RN50",
    "ViT-L~14",
    "dinov2-base"
]

# Supported image datasets
SUPPORTED_DATASETS = [
    "imagenet",
    "cc3m",
]

# Supported text vocabulary sources with their file paths
SUPPORTED_VOCABS = {
    "mscoco": "vocab/mscoco_unigram.txt",
    "laion_unigram": "vocab/laion_400_unigram.txt",
    "laion_bigrams": "vocab/laion_400_bigram.txt",
    "laion": ["laion_unigram", "laion_bigrams"],  # Combined vocabulary
    "disect": "vocab/clip_disect_20k.txt",
}

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
    Parse command line arguments for the embedding extraction script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Extract embeddings from the dataset")
    parser.add_argument("-d", "--dataset", type=str, required=True, 
                       help="Dataset to use (one of: imagenet, cc3m, celeba, or a supported vocabulary)")
    parser.add_argument("-m", "--model", type=str, required=True, 
                       help="CLIP model variant to use (e.g., 'ViT-B~32' for 'ViT-B/32')")
    parser.add_argument("-b", "--batch-size", type=int, default=4096, 
                       help="Batch size for embedding extraction")
    parser.add_argument("-ds", "--data-split", choices=["train", "val", "test"], required=True,
                       help="Data split to use (train, val, or test)")
    parser.add_argument("-v", "--vocab-size", type=int, default=-1, 
                       help="Vocabulary size limit (-1 for full vocabulary)")
    parser.add_argument("-w", "--workers", type=int, default=12, 
                       help="Number of workers for data loading")
    parser.add_argument("-io", "--image-only", action="store_true",
                       help="Extract only image embeddings (no text)")
    # Dataset subset arguments
    parser.add_argument("--subset-size", type=int, default=None,
                       help="Number of samples to process (subset of dataset)")
    parser.add_argument("--subset-start", type=int, default=0,
                       help="Starting index for dataset subset")
    return parser.parse_args()



class SubsetDataset(Dataset):
    """
    Wrapper for any dataset to handle subsets.
    
    Args:
        base_dataset (Dataset): The base dataset to subset
        start_idx (int): Starting index for the subset
        size (int): Size of the subset (None for all remaining samples)
    """
    def __init__(self, base_dataset: Dataset, start_idx: int = 0, size: int = None):
        self.base_dataset = base_dataset
        self.start_idx = start_idx
        
        max_size = len(base_dataset) - start_idx
        if size is None or size > max_size:
            self.size = max_size
        else:
            self.size = size
            
        logger.info(f"Created subset dataset: indices {start_idx} to {start_idx + self.size - 1} ({self.size} samples)")
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        if idx >= self.size:
            raise IndexError(f"Index {idx} out of range for subset of size {self.size}")
        return self.base_dataset[self.start_idx + idx]
    
    def __getattr__(self, name):
        # Delegate any other attribute access to the base dataset
        return getattr(self.base_dataset, name)






class DINOv2ModelWrapper(torch.nn.Module):
    def __init__(self, model_name="dinov2-base", device=torch.device("cuda")):
        super().__init__()
        self.model = Dinov2Model.from_pretrained(f"facebook/{model_name}")
        self.to(device)
        self.eval()
    
    def encode_image(self, pixel_values):
        # Automatically use the device where model parameters are
        device = next(self.parameters()).device
        outputs = self.model(pixel_values = pixel_values.to(device))
        return outputs.pooler_output

class EmbeddingExtractor:
    """
    Utility for extracting CLIP embeddings from images and text.
    
    Handles loading the CLIP model and preprocessing pipeline, and provides
    methods for embedding both images and text.
    
    Args:
        model_name (str): Name of the CLIP model variant to use
        device (str): Device to run inference on ('cuda' or 'cpu')
    """
    def __init__(self, model_name, device) -> None:
        self.model_name = model_name
        self.device = device
        # Load the model, preprocessor, and tokenizer
        self.model, self.preprocessor, self.tokenizer, self.token_max_length = self.load_model(
            model_name, self.device)
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    @staticmethod
    def load_model(model_name, device, model_path=None):
        """
        Load a CLIP model, preprocessor, and tokenizer.
        
        Args:
            model_name (str): Name of the CLIP model variant
            device (str): Device to load the model on
            model_path (str, optional): Custom path for model weights
            
        Returns:
            tuple: (model, preprocessor, tokenizer, token_max_length)
            
        Raises:
            ValueError: If the model variant is not supported
        """
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Model {model_name} not supported, please use one of {SUPPORTED_MODELS}")
        
        if "dino" in model_name.lower():
            # Load the model and preprocessor
            model = DINOv2ModelWrapper(model_name, device)
            base_preprocessor = AutoImageProcessor.from_pretrained(f"facebook/{model_name}", return_tensors="pt")
            # modify preprocessor to return torch tensors to perform the following operation
            # img = img.pixel_values #list of numpy arrays
            # img = torch.stack([torch.tensor(i) for i in img], dim=0)
            def preprocessor_wrapper(img):
                result = base_preprocessor(img, return_tensors="pt")
                if len(result.pixel_values.shape) >= 4: # batch of images
                    return result.pixel_values.squeeze(0)  # Remove batch dimension for single images
                return result.pixel_values

            # DINOv2 does not use a tokenizer - set to None
            tokenizer = None
            token_max_length = None 
            return model, preprocessor_wrapper, tokenizer, token_max_length
        else: # clip models
            # Replace '~' with '/' in model name (to handle command line limitations)
            model_name = model_name.replace("~", "/")
            
            # Load the model and preprocessor
            model, preprocessor = clip.load(model_name, device=device, download_root=model_path)
            
            # Create a tokenizer that truncates by default
            original_tokenizer = clip.tokenize
            tokenizer = lambda x: original_tokenizer(x, truncate=True)
            token_max_length = 77  # Standard context length for CLIP
                
            return model, preprocessor, tokenizer, token_max_length
    
    def embed_text(self, text):
        """
        Extract text embeddings using the CLIP model.
        
        Args:
            text (list or str or torch.Tensor): Text input(s) to embed
            
        Returns:
            tuple: (text_features, tokenized_text)
                - text_features: Normalized text embeddings
                - tokenized_text: Tokenized text inputs
        """
        
        if "dino" in self.model_name.lower():
            raise NotImplementedError("DINOEmbeddingExtractor does not support text embedding")
        # Handle different input types
        if not isinstance(text, torch.Tensor):
            text_embeddings = self.tokenizer(text if isinstance(text, list) else [text]).to(self.device)
        else:
            text_embeddings = text.to(self.device)

        # Extract features with mixed precision
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.model.encode_text(text_embeddings)
            
        return text_features, text_embeddings.detach().cpu()
    
    def embed_image(self, img):
        """
        Extract image embeddings using the CLIP model.
        
        Args:
            img (list or PIL.Image or torch.Tensor): Image input(s) to embed
            
        Returns:
            tuple: (image_features, preprocessed_images)
                - image_features: Normalized image embeddings
                - preprocessed_images: Preprocessed image tensors
        """
        # Handle different input types
        if isinstance(img, list):
            if not isinstance(img[0], torch.Tensor):
                img = [self.preprocessor(i).to(self.device) for i in img]
            img = torch.stack(img, dim=0).to(self.device)
        else:
            if not isinstance(img, torch.Tensor):
                img = self.preprocessor(img)
            
        if img.ndim == 3:
            img = img.unsqueeze(0)
        
        try:
            img = img.to(self.device)
        except Exception as e:
            pass

        # Extract features with mixed precision
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(img)
            
        return image_features, img.detach().cpu()
    

def load_data(dataset, preprocess, data_split="train"):
    """
    Load a dataset by name with appropriate preprocessing.
    
    Args:
        dataset (str): Name of the dataset to load
        preprocess (callable): Image preprocessing function
        data_split (str): Which split to load (train, val, test)
        
    Returns:
        IterableDataset: The loaded dataset
        
    Raises:
        ValueError: If the dataset is not supported
    """
    if dataset == "imagenet":
        # custom_data_split_name = data_split if data_split != "val" else "validation"
        # dataset = ImageNetDataset(preprocess, custom_data_split_name)
        custom_data_split_name = data_split
        dataset = ImageNet1kLocalDataset(data_split=custom_data_split_name, preprocess=preprocess, return_None_label=True, load_image=True)
    elif dataset == "cc3m":
        #dataset = CC3MDataset(preprocess, "train" if train else "validation")
        custom_data_split_name = data_split
        dataset = CC3MLocalDataset(preprocess, split=custom_data_split_name, return_file_path=False)
    elif dataset == "celeba":
        if data_split == "val":
            raise ValueError("The current CelebA implementation does not have a 'val' split, please use 'train' or 'test' or see whether a validation is available in principle")
        custom_data_split_name = data_split if data_split != "val" else "validation"
        dataset = CelebAMy(download=True, split=custom_data_split_name,
                          transform=preprocess, target_type="attr")
        raise ValueError(f"Dataset {dataset} not supported, please use one of {SUPPORTED_DATASETS}")

    # Add reverse class mapping if available
    if hasattr(dataset, "class_to_idx"):
        dataset.idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    
    return dataset


def load_vocab(vocab, vocab_size=-1):
    """
    Load a text vocabulary from files.
    
    Args:
        vocab (str): Name of the vocabulary to load
        vocab_size (int): Maximum number of items to include (-1 for all)
        
    Returns:
        VocabDataset: Dataset containing the vocabulary items
        
    Raises:
        ValueError: If the vocabulary is not supported
    """
    if vocab not in SUPPORTED_VOCABS.keys():
        raise ValueError(f"Vocab {vocab} not supported, please use one of {SUPPORTED_VOCABS.keys()}")
    
    path = SUPPORTED_VOCABS[vocab]
    
    # Handle composite vocabularies (e.g., "laion" = unigrams + bigrams)
    if isinstance(path, list):
        current_vocab = None
        for x in path:
            if current_vocab is None:
                current_vocab = load_vocab(x, vocab_size // 2 if vocab_size > 0 else -1)
            else:
                current_vocab += load_vocab(x, vocab_size // 2 if vocab_size > 0 else -1)
        return current_vocab

    # Load vocabulary from file
    vocab_data = []
    with open(path, 'r') as f:
        lines = f.readlines()
        if vocab_size > 0:
            # Apply vocabulary size limit if specified
            if vocab_size > len(lines):
                warnings.warn(f"Vocab size {vocab_size} is greater than the actual vocab size {len(lines)}. Using full vocab.")
            else:
                lines = lines[-vocab_size:]  # Take most frequent terms (assuming frequency-sorted lists)

        for line in lines:
            line = line.strip()
            vocab_data.append(line)
    
    return VocabDataset(vocab_data)


def safe_collate(batch):
    # Filter out None values
    batch = [(img, txt) for img, txt in batch if img is not None or txt is not None]
    if not batch:
        return None, None  # Handle empty batch case
    
    # If any item is None, set it to None
    images, texts = zip(*batch)
    if any(x is None for x in images):
        images = None
    else:
        images = torch.stack(images)
        
    if any(x is None for x in texts):
        texts = None
        
    # Process valid items
    return images, texts


def construct_filename(dataset, model, split_name, dataset_size, modality, file_ext, embedding_dim=None):
    """
    Construct a filename for saving embeddings based on parameters.
    
    """
    prefix = f"{dataset}_{model}_{split_name}_{modality}_{dataset_size}"
    if embedding_dim is not None:
        prefix += f"_{embedding_dim}"
    return f"{prefix}.{file_ext}"

def main(args):
    """
    Main function for extracting and saving embeddings.
    
    Loads the specified dataset and model, extracts embeddings for all samples,
    and saves them to disk as memory-mapped arrays.
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
    set_seed(42)
    logger.info(f"Extracting embeddings for {args.dataset} using {args.model} model")
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load CLIP model and embedding extractor
    extractor = EmbeddingExtractor(args.model, device)
    
    # Load appropriate dataset
    if args.dataset in SUPPORTED_DATASETS:
        logger.info(f"Loading dataset {args.dataset}")
        base_dataset = load_data(args.dataset, extractor.preprocessor, args.data_split)
        split_name = args.data_split if args.data_split != "val" else "validation"
    elif args.dataset in SUPPORTED_VOCABS.keys():
        logger.info(f"Loading vocab {args.dataset}")
        base_dataset = load_vocab(args.dataset, args.vocab_size)
        split_name = str(args.vocab_size)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported, please use one of {SUPPORTED_DATASETS + list(SUPPORTED_VOCABS.keys())}")
    
    # Create subset if requested
    if args.subset_size is not None:
        dataset = SubsetDataset(base_dataset, args.subset_start, args.subset_size)
        # Add subset info to split name
        split_name = f"{split_name}_subset_{args.subset_start}_{args.subset_size}"
    else:
        dataset = base_dataset
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        # Get a sample to determine embedding dimension
        dataset_size = len(dataset)
        sample = dataset[0][0]
        if sample is None:
            sample = dataset[0][1]
            if sample is None:
                logger.error("Sample is None, skipping embedding extraction.")
                return
            features, _ = extractor.embed_text(sample)
        else:
            features, _ = extractor.embed_image(sample)
        embedding_dim = features.shape[-1]
        
        logger.info(f"Creating embeddings Memmap with length {dataset_size} and shape {embedding_dim}")
        
        # Prepare memory-mapped arrays for storing embeddings
        memmap_image_path = os.path.join(
            "data", 
            construct_filename(args.dataset, args.model, split_name, dataset_size, "image", "npy", embedding_dim)
        )
        image_memmap = np.memmap(
            memmap_image_path, 
            dtype=np.float32, 
            mode='w+', 
            shape=(dataset_size, embedding_dim)
        )
        logger.info(f"Saving image embeddings to {memmap_image_path}")
        
        memmap_text_path = os.path.join(
            "data", 
            construct_filename(args.dataset, args.model, split_name, dataset_size, "text", "npy", embedding_dim)
        )
        text_memmap = np.memmap(
            memmap_text_path, 
            dtype=np.float32, 
            mode='w+', 
            shape=(dataset_size, embedding_dim)
        )
        logger.info(f"Saving text embeddings to {memmap_text_path}")
        
        # Also save the original text for reference
        text_output_path = os.path.join(
            "data", 
            construct_filename(args.dataset, args.model, split_name, dataset_size, "text", "txt")
        )
        logger.info(f"Saving original text to {text_output_path}")
        
        # Process data in batches
        text_full = []
        dl = torch.utils.data.DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            num_workers=args.workers,
            pin_memory=True, 
            drop_last=False,
            collate_fn=safe_collate
        )
        
        logger.info("Extracting embeddings...")
        start_idx = 0
        successful_count = 0
        for images, texts in tqdm(dl, total=len(dl), desc="Extracting embeddings"):
            if images is None and texts is None:
                continue
            
            # Calculate batch indices
            end_idx = start_idx + len(images if images is not None else texts)
            
            # Extract and save image embeddings
            if images is not None:
                image_embeddings, _ = extractor.embed_image(images)
                image_memmap[start_idx:end_idx] = image_embeddings.detach().cpu().numpy().astype(np.float32)
                image_memmap.flush()
            
            # Extract and save text embeddings
            if texts is not None and args.image_only == False:
                
                # Convert numerical class indices to text labels if needed
                texts = list(texts)
                if isinstance(texts, list) and isinstance(texts[0], int):
                    texts = [dataset.idx_to_class[x] for x in texts]
                
                text_embeddings, _ = extractor.embed_text(texts)
                text_memmap[start_idx:end_idx] = text_embeddings.detach().cpu().numpy().astype(np.float32)
                text_memmap.flush()
            
                # Collect original text
                text_full.extend(texts)
            
            # Update indices for next batch
            start_idx = end_idx
            successful_count += len(images if images is not None else texts)
        
        # Save original text to file
        with open(text_output_path, 'w') as f:
            f.write("\n".join(text_full))
        
        #Correct end index for memmap if needed
        if successful_count < dataset_size:
            logger.info(f"Resizing memmaps to {successful_count} items")
            
            # Create new memmaps with correct size
            final_image_path = os.path.join(
                "data", 
                construct_filename(args.dataset, args.model, split_name, successful_count, "image", "npy", embedding_dim)
            )
            final_text_path = os.path.join(
                "data", 
                construct_filename(args.dataset, args.model, split_name, successful_count, "text", "npy", embedding_dim)
            )
            
            # Copy data to new memmaps
            final_image_memmap = np.memmap(
                final_image_path, 
                dtype=np.float32, 
                mode='w+', 
                shape=(successful_count, embedding_dim)
            )
            final_image_memmap[:] = image_memmap[:successful_count]
            final_image_memmap.flush()
            
            final_text_memmap = np.memmap(
                final_text_path, 
                dtype=np.float32, 
                mode='w+', 
                shape=(successful_count, embedding_dim)
            )
            final_text_memmap[:] = text_memmap[:successful_count]
            final_text_memmap.flush()
            
            # Also update the text file
            final_text_output_path = os.path.join(
                "data", 
                f"{args.dataset}_{args.model}_{split_name}_text_{successful_count}.txt"
            )
            with open(final_text_output_path, 'w') as f:
                f.write("\n".join(text_full))
            
            logger.info(f"Resized memmaps saved at {final_image_path} and {final_text_path} and text to {final_text_output_path}")
                        
            # Cleanup original files
            os.remove(memmap_image_path)
            os.remove(memmap_text_path)
            os.remove(text_output_path)
            logger.info(f"Removed original memmaps and text file from {memmap_image_path}, {memmap_text_path}, and {text_output_path}")
            
            memmap_image_path = final_image_path
            image_memmap = final_image_memmap
            memmap_text_path = final_text_path
            text_memmap = final_text_memmap
            
        if image_memmap.sum() == 0:
            os.remove(memmap_image_path)
            logger.info(f"Removed empty memmap file {memmap_image_path}")
        
        if text_memmap.sum() == 0:
            os.remove(memmap_text_path)
            logger.info(f"Removed empty memmap file {memmap_text_path}")
        
        logger.info("Embedding extraction complete")
        
        
if __name__ == "__main__":
    args = parse_args()
    main(args)
