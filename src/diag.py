import dask.array as da
import numpy as np
import matplotlib.pyplot as plt
import zarr
import os
import logging

import sys

if len(sys.argv) != 4:
    print("Usage: python diag.py <input_zarr> <norm_zarr> <output_dir>")
    sys.exit(1)

ZARR_RAW_PATH = sys.argv[1]
ZARR_PROCESSED_PATH = sys.argv[2]
OUTPUT_DIR = sys.argv[3]

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{OUTPUT_DIR}/diagnostic_report.log"),
        logging.StreamHandler()
    ]
)

def check_integrity(raw_data, norm_data):
    logging.info("--- ÉTAPE 1 : VÉRIFICATION D'INTÉGRITÉ ---")
    
    if raw_data.shape == norm_data.shape:
        logging.info(f"[PASS] Dimensions identiques : {norm_data.shape}")
    else:
        logging.error(f"[FAIL] Incohérence des dimensions. Raw: {raw_data.shape}, Norm: {norm_data.shape}")

    if norm_data.dtype == np.uint8:
        logging.info("[PASS] Type de données correct : uint8")
    else:
        logging.warning(f"[WARN] Type inattendu : {norm_data.dtype}. Devrait être uint8.")

    logging.info("Scan d'intégrité sur un échantillon de 10 000 images...")
    sample = norm_data[:10000].compute()
    
    if np.isnan(sample).any():
        logging.error("[FAIL] Des NaNs ont été détectés dans les données normalisées !")
    else:
        logging.info("[PASS] Aucun NaN détecté dans l'échantillon.")
        
    min_val, max_val = sample.min(), sample.max()
    if min_val >= 0 and max_val <= 255:
        logging.info(f"[PASS] Valeurs dans les bornes correctes [0, 255]. (Min: {min_val}, Max: {max_val})")
    else:
        logging.error(f"[FAIL] Valeurs hors limites ! (Min: {min_val}, Max: {max_val})")

def generate_visual_report(raw_data, norm_data, num_samples=5):
    logging.info("--- ÉTAPE 2 : RAPPORT VISUEL ---")
    logging.info(f"Génération d'une grille comparative de {num_samples} patches...")
    
    indices = np.random.randint(0, raw_data.shape[0], size=num_samples)
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 3 * num_samples))
    
    for i, idx in enumerate(indices):
        img_raw = raw_data[idx].compute()
        img_norm = norm_data[idx].compute()
        
        axes[i, 0].imshow(img_raw)
        axes[i, 0].set_title(f"Patch {idx} - Brut")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(img_norm)
        axes[i, 1].set_title(f"Patch {idx} - Normalisé")
        axes[i, 1].axis('off')
        
    plt.tight_layout()
    vis_path = f"{OUTPUT_DIR}/visual_comparison.png"
    plt.savefig(vis_path, dpi=150)
    plt.close()
    logging.info(f"[PASS] Grille visuelle sauvegardée : {vis_path}")

def generate_statistical_report(raw_data, norm_data):
    logging.info("--- ÉTAPE 3 : RAPPORT STATISTIQUE ---")
    logging.info("Calcul des histogrammes colorimétriques sur un échantillon...")
    
    sample_raw = raw_data[:500].compute()
    sample_norm = norm_data[:500].compute()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    colors = ['red', 'green', 'blue']
    
    for i, color in enumerate(colors):
        hist_raw, bins = np.histogram(sample_raw[..., i], bins=256, range=(0, 256))
        ax1.plot(bins[:-1], hist_raw, color=color, alpha=0.7)
        
        hist_norm, _ = np.histogram(sample_norm[..., i], bins=256, range=(0, 256))
        ax2.plot(bins[:-1], hist_norm, color=color, alpha=0.7)
        
    ax1.set_title("Distribution des Couleurs (Données Brutes)")
    ax1.set_xlim(0, 255)
    ax2.set_title("Distribution des Couleurs (Après Macenko)")
    ax2.set_xlim(0, 255)
    
    stat_path = f"{OUTPUT_DIR}/statistical_histograms.png"
    plt.savefig(stat_path, dpi=150)
    plt.close()
    logging.info(f"[PASS] Graphique statistique sauvegardé : {stat_path}")

def main():
    logging.info("DÉMARRAGE DU DIAGNOSTIC DATA QUALITY")
    
    try:
        raw_data = da.from_zarr(ZARR_RAW_PATH)
        norm_data = da.from_zarr(ZARR_PROCESSED_PATH)
        
        check_integrity(raw_data, norm_data)
        generate_visual_report(raw_data, norm_data)
        generate_statistical_report(raw_data, norm_data)
        
        logging.info("DIAGNOSTIC TERMINÉ AVEC SUCCÈS.")
        
    except Exception as e:
        logging.error(f"Erreur critique lors du diagnostic : {e}")

if __name__ == "__main__":
    main()