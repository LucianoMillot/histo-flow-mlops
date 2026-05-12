import h5py
import zarr
import os

import sys

if len(sys.argv) != 3:
    print("Usage: python convert_2_zarr.py <input_h5> <output_zarr>")
    sys.exit(1)

H5_PATH = sys.argv[1]
ZARR_PATH = sys.argv[2]

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