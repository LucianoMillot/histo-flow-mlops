import torch
import zarr
import numpy as np
import h5py
from torch.utils.data import Dataset

class PCamZarrDataset(Dataset):

    def __init__(self, zarr_path, labels_path=None, transform=None):
        self.zarr_path = zarr_path
        self.data = None
        
        self.labels = None
        if labels_path:
            with h5py.File(labels_path, 'r') as f:
                self.labels = torch.tensor(f['y'][:]).long().squeeze()
            
        self.transform = transform

    def __len__(self):
        if self.data is None:
            # Ouvrir temporairement juste pour récupérer la taille
            return zarr.open(self.zarr_path, mode='r').shape[0]
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Lazy loading pour le multiprocessing (num_workers > 0)
        if self.data is None:
            self.data = zarr.open(self.zarr_path, mode='r')
            
        image = np.array(self.data[idx])
        
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        if self.labels is not None:
            label = self.labels[idx]
            return image, label
            
        return image