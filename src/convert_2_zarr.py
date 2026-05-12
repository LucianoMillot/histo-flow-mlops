import h5py
import zarr
import os

H5_PATH = '../data/raw/pcam/training_split.h5'
ZARR_PATH = '../data/raw/pcam/training_split.zarr'

print("Chargement des données HDF5 en RAM...")
with h5py.File(H5_PATH, 'r') as f:
    data = f['x'][:]

print("Création de l'architecture Zarr...")
z = zarr.open(
    ZARR_PATH, 
    mode='w', 
    shape=data.shape, 
    chunks=(2000, 96, 96, 3), 
    dtype=data.dtype
)

print("Écriture des données...")
z[:] = data

print(f"Conversion terminée. Le dossier {ZARR_PATH} a été créé.")