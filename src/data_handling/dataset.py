import torch
from torch_geometric.data import Dataset
import os

class AirfoilDataset(Dataset):
    """
    Loads pre-processed .pt files for the AirfRANS dataset.
    """
    def __init__(self, root, transform=None, pre_transform=None):
        """
        Args:
            root (str): The root directory of your project.
        """
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed_data')

    @property
    def processed_file_names(self):
        if not hasattr(self, '_processed_files'):
             self._processed_files = sorted([
                f for f in os.listdir(self.processed_dir) if f.endswith('.pt')
            ])
        return self._processed_files

    def download(self):
        pass

    def process(self):
        print(f"Skipping processing, loading pre-processed files from {self.processed_dir}")

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        """
        Loads and returns a single pre-processed graph from the disk.
        """
        file_path = os.path.join(self.processed_dir, self.processed_file_names[idx])
        data = torch.load(file_path, weights_only=False)
        return data