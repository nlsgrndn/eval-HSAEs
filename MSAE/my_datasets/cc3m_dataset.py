import csv
import os
import pickle
import logging

from PIL import Image
from torch.utils.data import Dataset
INDEX_BASE_PATH = "dataset_datastructures/general_indices"
MAX_LEN_FOR_TESTING= 10_000_000

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CC3MLocalDataset(Dataset):
    def __init__(self, preprocess, split, root_path="../datasets/cc3m/cc3m", return_file_path = False):
        self.preprocess = preprocess
        self.split = split
        self.root_path = root_path
        self.return_file_path = return_file_path

        local_split = 'val' if split == 'validation' else split
        self.split_path = os.path.join(root_path, "cc_data", local_split)

        if not os.path.exists(self.split_path):
            raise FileNotFoundError(f"Split path {self.split_path} does not exist")

        # Set CSV path
        if local_split == 'val':
            self.csv_path = os.path.join(root_path, "Validation_GCC-1.1.0-Validation_output.csv")
        elif local_split == 'train':
            self.csv_path = os.path.join(root_path, "Train_GCC-training_output.csv")
        else:
            raise ValueError(f"Unsupported split: {split}. Use 'train' or 'val'.")

        # Build or load file position index
        index_path = os.path.join(INDEX_BASE_PATH, f"cc3m_{local_split}_{MAX_LEN_FOR_TESTING}_index.pkl")

        if os.path.exists(index_path):
            print(f"Loading pre-built index from {index_path}")
            with open(index_path, 'rb') as f:
                self.file_positions = pickle.load(f)
            self.len = len(self.file_positions)
        else:
            print(f"Building index for {self.csv_path}")
            self._build_index(index_path)

        print(f"Loaded CC3M dataset with {self.len} entries")

    def _build_index(self, index_path):
        """Build index of file positions for each row."""
        self.file_positions = []

        with open(self.csv_path, 'r', encoding='utf-8') as f:
            # Skip header and record its position
            header_pos = f.tell()
            header = f.readline()

            # Record position of each data row
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if len(self.file_positions) >= MAX_LEN_FOR_TESTING:
                    print(f"Reached max testing length of {MAX_LEN_FOR_TESTING}, stopping index build")
                    break
                self.file_positions.append(pos)

        self.len = len(self.file_positions)

        # Save index for future use
        if not os.path.exists(INDEX_BASE_PATH):
            os.makedirs(INDEX_BASE_PATH, exist_ok=True)
        with open(index_path, 'wb') as f:
            pickle.dump(self.file_positions, f)
        print(f"Built index with {self.len} entries, saved to {index_path}")

        # Also save header info for parsing
        self.header_path = index_path.replace('_index.pkl', '_header.txt')
        with open(self.header_path, 'w', encoding='utf-8') as f:
            f.write(header.strip())

    def _get_row_data(self, idx):
        """Fast lookup using file position index."""
        if idx >= len(self.file_positions):
            raise IndexError(f"Index {idx} out of range")

        import csv
        from io import StringIO

        with open(self.csv_path, 'r', encoding='utf-8') as f:
            # Seek to the specific row
            f.seek(self.file_positions[idx])
            line = f.readline().strip()

            # Parse the line using csv reader
            reader = csv.DictReader(StringIO(line), delimiter='\t',
                                  fieldnames=['title', 'filepath'])
            row = next(reader)

            rel_path = row['filepath']
            abs_path = os.path.join(self.root_path, rel_path)
            return (row['title'], abs_path)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        try:
            caption, image_path = self._get_row_data(idx)
            if self.return_file_path:
                return image_path, caption
            if not image_path:
                raise ValueError(f"No caption found for index {idx}")

            image = Image.open(image_path).convert("RGB")

            width, height = image.size

            # Rescale tiny images to minimum acceptable size
            width, height = image.size
            if width < 2 or height < 2:
                logger.warning(f"Rescaling small image at index {idx}: size {image.size}, path: {image_path}")
                # Resize maintaining aspect ratio, then center crop to 224x224
                scale = max(2 / width, 2 / height)
                new_size = (int(width * scale), int(height * scale))
                image = image.resize(new_size, Image.BILINEAR)

            if self.preprocess:
                image = self.preprocess(image)

            return image, caption

        except (UnicodeDecodeError, OSError, SyntaxError, IOError) as e:
            logger.warning(f"Error loading sample {idx}: {e}")
            return None, None