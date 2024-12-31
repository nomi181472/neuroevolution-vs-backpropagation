import torch
import random
import numpy as np
import os
class Utils:
    @staticmethod
    def get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    @staticmethod
    def set_seed(seed):
        """Set the random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def create_directory(path):
        """Create a directory if it does not exist."""
        if not os.path.exists(path):
            os.makedirs(path)
