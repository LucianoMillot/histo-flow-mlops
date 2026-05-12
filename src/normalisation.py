import numpy as np
import torchstain
import cv2
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
import os
import warnings

warnings.filterwarnings('ignore')
np.seterr(all='ignore')

ZARR_RAW_PATH = '../data/raw/pcam/training_split.zarr'
ZARR_PROCESSED_PATH = '../data/processed/train_x_norm.zarr'

def is_patch_useful(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if np.mean(gray) > 235 or np.var(gray) < 10:
        return False
    return True

def normalize_patch(patch, normalizer):
    if not is_patch_useful(patch): return patch
    try:
        curr_norm, _, _ = normalizer.normalize(I=patch, stains=True)
        return curr_norm.astype(np.uint8)
    except:
        return patch

def process_chunk(block, normalizer=None):
    return np.array([normalize_patch(p, normalizer) for p in block])

def main():
    dask.config.set(scheduler='processes')
    data_x = da.from_zarr(ZARR_RAW_PATH)
    
    ref_patch = data_x[100].compute() 
    
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='numpy')
    normalizer.fit(ref_patch)
    print(f"Normaliseur Macenko prêt.")

    norm_data = data_x.map_blocks(
        process_chunk, 
        normalizer=normalizer, 
        dtype=np.uint8
    )

    print(f"Écriture vers : {ZARR_PROCESSED_PATH}")
    with ProgressBar():
        da.to_zarr(norm_data, ZARR_PROCESSED_PATH, overwrite=True)

if __name__ == "__main__":
    main()