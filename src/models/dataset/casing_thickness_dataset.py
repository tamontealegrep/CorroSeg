
import numpy as np
import torch
from torch.utils.data import Dataset

#---------------------------------------------------------------------------------------------------

class CasingThicknessDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray = None, transform=None, **kwargs):
        """
        A dataset class for handling casing thickness images and their segmentation masks.

        This class allows loading images and masks, applying optional transformations, 
        and retrieving samples for training or inference.

        Args:
            X (np.ndarray): Input data with shape (n, h, w).
            y (np.ndarray, optional): Segmentation masks with shape (n, h, w).
            transform (callable, optional): Transformations to apply to X and y.
            kwargs: Additional arguments for the transformation function.

        Raises:
            ValueError: If y is provided and its shape does not match the shape of X.

        """
        self.X = X
        self.y = y
        self.transform = transform
        self.kwargs = kwargs

        if y is not None and X.shape != y.shape:
            raise ValueError("X and y must have the same shape")
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = self.X[idx]
        
        if self.transform:
            y = self.y[idx] if self.y is not None else np.zeros_like(X)
            X, y = self.transform(X, y, **self.kwargs)
            y = torch.tensor(y, dtype=torch.float32).unsqueeze(0) if self.y is not None else None
        else:
            y = torch.tensor(self.y[idx], dtype=torch.float32).unsqueeze(0) if self.y is not None else None

        X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # (h, w) -> (1, h, w)

        return X, y 

#---------------------------------------------------------------------------------------------------