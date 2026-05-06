from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision.datasets import CelebA
import torch


class HFDataset(Dataset):
    """
    Base class for Hugging Face dataset wrappers.

    Provides common functionality for loading and preprocessing datasets
    from the Hugging Face hub.

    Args:
        dataset (str): Dataset identifier on Hugging Face hub
        preprocess (callable): Image preprocessing function
        split (str): Dataset split to use
        download_full (bool): Whether to download the full dataset at once or stream it
        **kwargs: Additional arguments to pass to load_dataset
    """
    def __init__(self, dataset, preprocess, split, download_full=False, **kwargs):
        stream = not download_full
        self.dataset = load_dataset(dataset, split=split, streaming=stream,
                                    trust_remote_code=True, **kwargs)
        self.preprocess = preprocess
        self.len: int = 0  # Will be set by child classes

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.len

    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        item = self.dataset[idx]
        sample, target = item['jpg'], item['txt']
        if self.preprocess:
            sample = self.preprocess(sample)
        return sample, target


class ImageNetDataset(HFDataset):
    """
    Wrapper for the ImageNet dataset from Hugging Face.

    Handles ImageNet-specific data format and preprocessing.

    Args:
        preprocess (callable): Image preprocessing function
        split (str): Dataset split to use ('train' or 'validation')
    """
    def __init__(self, preprocess, split):
        super().__init__("ILSVRC/imagenet-1k", preprocess, split, True)
        # Set dataset length based on split info
        self.len = self.dataset.info.splits[self.dataset.split].num_examples
        # Create mapping from class names to indices
        self.class_to_idx = {}
        for idx, class_name in enumerate(self.dataset.info.features['label'].names):
            self.class_to_idx[class_name] = idx

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            sample, target = item['image'], item['label']
            if target == -1:
                target = ""  # Handle missing labels
            else:
                target = self.dataset.info.features['label'].int2str(target)

            if isinstance(sample, Image.Image):
                sample = sample.convert("RGB")
            if self.preprocess:
                sample = self.preprocess(sample)
            return sample, target
        except (UnicodeDecodeError, OSError, SyntaxError) as e:
            return None, None


class CC3MDataset(HFDataset):
    """
    Wrapper for the Conceptual Captions 3M dataset.

    Handles CC3M-specific data format and preprocessing.

    Args:
        preprocess (callable): Image preprocessing function
        split (str): Dataset split to use ('train' or 'validation')
        download_full (bool): Whether to download the full dataset at once
    """
    def __init__(self, preprocess, split, download_full=False):
        download_full=True
        super().__init__("pixparse/cc3m-wds", preprocess, split, download_full)
        # Hardcoded dataset sizes since they're not always available from the API
        if split == "train":
            self.len = 2905954
        else:
            self.len = 13443

    def __getitem__(self, idx):
        """Return preprocessed image and caption pairs."""
        try:
            item = self.dataset[idx]
            sample, target = item['jpg'], item['txt']
            if isinstance(sample, Image.Image):
                sample = sample.convert("RGB")

            if self.preprocess:
                sample = self.preprocess(sample)
            return sample, target
        except (UnicodeDecodeError, OSError, SyntaxError) as e:
            return None, None


class CelebAMy(Dataset):
    """
    Custom wrapper for the CelebA dataset.

    Combines multiple attribute labels into a comma-separated string
    to use as the text component for CLIP.

    Args:
        root (str): Root directory for CelebA data
        split (str): Dataset split ('train', 'valid', or 'test')
        **kwargs: Additional arguments to pass to CelebA constructor
    """
    def __init__(self, root, split, **kwargs):
        self.celeba = CelebA(root, split=split, **kwargs)
        self.attr_names = self.celeba.attr_names[:40]  # Using first 40 attributes

    def __getitem__(self, index):
        """Yield image samples with concatenated attribute labels as text."""
        sample, target = self.celeba[index]
        # Get the indices of attributes that are True for this sample
        labels_by_target = torch.nonzero(target)[:, 0]
        # Convert attribute indices to attribute names and join with commas
        target = ','.join([str(self.attr_names[x]) for x in labels_by_target])
        return sample, target

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.celeba)


class VocabDataset(Dataset):
    """
    Dataset for processing text vocabulary items.

    This dataset treats each vocabulary entry as a text item to be embedded,
    with no corresponding image.

    Args:
        data (list): List of vocabulary items (strings)
    """
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        """Yield vocabulary items with None as placeholder for images."""
        return None, self.data[index]

    def __len__(self):
        """Return the number of vocabulary items."""
        return len(self.data)

    def __add__(self, other):
        """Concatenate two VocabDataset instances."""
        if isinstance(other, VocabDataset):
            return VocabDataset(self.data + other.data)
        else:
            raise TypeError("Can only concatenate VocabDataset instances")