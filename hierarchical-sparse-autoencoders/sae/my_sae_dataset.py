import numpy as np

"""
Sparse Autoencoder (SAE) Utilities

This module provides utility functions and classes for training and using
Sparse Autoencoders, including dataset handling, learning rate schedulers,
custom activation functions, and various mathematical operations.
"""

class SAEDataset:
    """
    Memory-efficient dataset implementation for Sparse Autoencoders.
    
    This class loads data from memory-mapped numpy arrays to efficiently handle
    large datasets without loading everything into memory at once. It also
    handles preprocessing like mean centering and normalization.
    
    The class automatically parses dataset dimensions from the filename,
    which is expected to contain the data shape as the last two underscored
    components (e.g., "dataset_name_10000_768.npy" for 10000 vectors of size 768).
    
    Args:
        data_path (str): Path to the memory-mapped numpy array file
        dtype (np.dtype, optional): Data type for arrays. Defaults to np.float32.
        mean_center (bool, optional): Whether to center the data by subtracting the mean.
                                     Defaults to False.
        target_norm (float, optional): Target norm for normalization. If None, uses sqrt(vector_size).
                                     If 0.0, no normalization is applied. Defaults to None.
        max_size (int, optional): Maximum number of samples to use from dataset. If None, uses all.
                                 Defaults to None.
    """
    def __init__(self, data_path: str, dtype: np.dtype = np.float32, mean_center: bool = False, target_norm: float = None, max_size: int = None):
        # Parse vector dimensions from filename
        parts = data_path.split("/")[-1].split(".")[0].split("_")
        self.len, self.vector_size = map(int, parts[-2:])
        
        # Set core attributes
        self.dtype = np.dtype(dtype)
        self.data = np.memmap(data_path, dtype="float32", mode="r", 
                             shape=(self.len, self.vector_size))
        

        # Apply max_size limit if specified
        if max_size is not None:
            self.len = min(self.len, max_size)
            self.data = self.data[:self.len]
        
        # Special case for representation files (already preprocessed)
        if "repr" in data_path:
            self.mean = np.zeros(self.vector_size, dtype=self.dtype)
            self.mean_center = False
            self.scaling_factor = 1.0
            return

        # Set preprocessing configuration
        self.mean_center = mean_center
        self.target_norm = np.sqrt(self.vector_size) if target_norm is None else target_norm

        # Compute statistics if needed
        if self.mean_center or self.target_norm != 0.0:
            self._compute_statistics()
        else:
            self.mean = np.zeros(self.vector_size, dtype=self.dtype)
            self.scaling_factor = 1.0

    def _compute_statistics(self, batch_size: int = 10000):
        """
        Compute dataset statistics (mean and scaling factor) in memory-efficient batches.
        
        Args:
            batch_size (int, optional): Number of samples to process at once. Defaults to 10000.
        """
        # Compute mean if mean centering is enabled
        if self.mean_center:
            mean_acc = np.zeros(self.vector_size, dtype=np.float32)
            total = 0

            for start in range(0, self.len, batch_size):
                end = min(start + batch_size, self.len)
                batch = self.data[start:end].copy()
                mean_acc += np.sum(batch, axis=0)
                total += (end - start)

            self.mean = np.asarray(mean_acc / total, dtype=self.dtype)
        else:
            self.mean = np.zeros(self.vector_size, dtype=self.dtype)

        # Compute scaling factor if normalization is enabled
        if self.target_norm != 0.0:
            squared_norm_sum = 0.0
            total = 0

            for start in range(0, self.len, batch_size):
                end = min(start + batch_size, self.len)
                batch = self.data[start:end].copy()
                # Center the batch if needed
                batch = batch - self.mean
                squared_norm_sum += np.sum(np.square(batch))
                total += (end - start)

            avg_squared_norm = squared_norm_sum / total
            self.scaling_factor = float(self.target_norm / np.sqrt(avg_squared_norm))
        else:
            self.scaling_factor = 1.0

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.len
    
    def process_data(self, data: np.ndarray) -> np.ndarray:
        """
        Process data for the autoencoder (subtract mean and apply scaling).
        
        Args:
            data (np.ndarray): Input data array
            
        Returns:
            np.ndarray: Processed data array
        """
        return (np.asarray(data, dtype=self.dtype) - self.mean) * self.scaling_factor
    
    def unprocess_data(self, data: np.ndarray) -> np.ndarray:
        """
        Reverse the processing of data (apply inverse scaling and add mean).
        
        Args:
            data (np.ndarray): Input data array
            
        Returns:
            np.ndarray: Unprocessed data array
        """
        return np.asarray(data, dtype=self.dtype) / self.scaling_factor + self.mean

    def __getitem__(self, idx):
        """
        Get a preprocessed data sample at the specified index.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            np.ndarray: Preprocessed data sample
        """
        return self.process_data(self.data[idx])