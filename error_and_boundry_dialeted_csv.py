import os
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
import json
from scipy.ndimage import binary_dilation
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Compute uncertainty statistics around segmentation borders.")
    parser.add_argument("--n_dilations", type=int, default=1, help="Number of dilation steps to apply to each label mask.")
    parser.add_argument("--input_dir", required=True, help="Base directory containing subject subfolders.")
    parser.add_argument("--output_dir", required=True, help="Directory to write summary CSV and plots.")
    return parser.parse_args()

def compute_stats(seg, unc, err=None, n_dilations=1):
    labels = np.unique(seg)
    labels = labels[labels != 0]
    struct_elem = np.ones((3, 3, 3), dtype=bool)

    results = []
    for label in labels:
        mask = seg == label
        if not np.any(mask):
            continue

        dilated = mask.copy()
        for _ in range(n_dilations):
            dilated = binary_dilation(dilated, structure=struct_elem)

        border = np.logical_and(dilated, ~mask)

        stat = {
            "Label": int(label),
            "MeanVarianceInDilatedBorder": float(np.mean(unc[dilated])),
            "StdVarianceInDilatedBorder": float(np.std(unc[dilated])),
            "MinVariance": float(np.min(unc[dilated])),
            "MaxVariance": float(np.max(unc[dilated])),
            "n_voxels": int(np.sum(dilated))
        }

        if err is not None:
            stat["MeanSquaredErrorInDilatedBorder"] = float(np.mean(err[border]))

        results.append(stat)

    return results

def main():
    args = parse_args()

    base_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    UNC_NAME = "uncertainty_map.nii.gz"
    SEG_NAME = "groundtruth_segm.nii.gz"
    ERROR_NAME = "squared_error.nii.gz"
    OUT_JSON_NAME = "dilated_uncertainty_stats.json"

    subject_dirs = [
        os.path.join(base_dir, d) for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    all_results = []

    for subj_dir in subject_dirs:
        subj_id = os.path.basename(subj_dir)
        unc_path = os.path.join(subj_dir, UNC_NAME)

        if not os.path.exists(unc_path):
            print(f"[{subj_id}] Skipping: Uncertainty map not found.")
            continue

        seg_path = None
        err_path = None
        for root, _, files in os.walk(subj_dir):
            if SEG_NAME in files:
                seg_path = os.path.join(root, SEG_NAME)
            if ERROR_NAME in files:
                err_path = os.path.join(root, ERROR_NAME)

        if not seg_path or not err_path:
            print(f"[{subj_id}] Skipping: Missing segmentation or error map.")
            continue

        try:
            seg = nib.load(seg_path).get_fdata().astype(np.int32)
            unc = nib.load(unc_path).get_fdata().astype(np.float32)
            err = nib.load(err_path).get_fdata().astype(np.float32)
        except Exception as e:
            print(f"[{subj_id}] Error loading data: {e}")
            continue

        print(f"[{subj_id}] Processing...")

        stats = compute_stats(seg, unc, err, n_dilations=args.n_dilations)

        out_json_path = os.path.join(subj_dir, OUT_JSON_NAME)
        with open(out_json_path, "w") as f:
            json.dump({int(row["Label"]): row for row in stats}, f, indent=2)

        for row in stats:
            row["SubjectID"] = subj_id
            all_results.append(row)

    if all_results:
        df = pd.DataFrame(all_results)
        df_clean = df.dropna(subset=["MeanVarianceInDilatedBorder", "MeanSquaredErrorInDilatedBorder"])

        pearson_corr, _ = pearsonr(df_clean["MeanVarianceInDilatedBorder"], df_clean["MeanSquaredErrorInDilatedBorder"])
        spearman_corr, _ = spearmanr(df_clean["MeanVarianceInDilatedBorder"], df_clean["MeanSquaredErrorInDilatedBorder"])

        print("\n--- Correlation Results ---")
        print(f"Pearson:  {pearson_corr:.4f}")
        print(f"Spearman: {spearman_corr:.4f}")

        summary = {
            "Label": "ALL",
            "SubjectID": "combined",
            "MeanVarianceInDilatedBorder": df_clean["MeanVarianceInDilatedBorder"].mean(),
            "MeanSquaredErrorInDilatedBorder": df_clean["MeanSquaredErrorInDilatedBorder"].mean(),
            "PearsonCorr": pearson_corr,
            "SpearmanCorr": spearman_corr
        }
        df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

        csv_out = os.path.join(output_dir, "combined_uncertainty_error_stats.csv")
        df.to_csv(csv_out, index=False)

        # Plot
        plt.figure(figsize=(6, 5))
        plt.scatter(df_clean["MeanVarianceInDilatedBorder"], df_clean["MeanSquaredErrorInDilatedBorder"], alpha=0.7)
        plt.xlabel("Uncertainty (Mean Variance in Dilated Border)")
        plt.ylabel("Error (Mean Squared Error in Dilated Border)")
        plt.title("Error vs Uncertainty")
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "error_vs_uncertainty_all.png")
        plt.savefig(plot_path)
        plt.close()

        print("\n Output saved:")
        print(f"- CSV:  {csv_out}")
        print(f"- Plot: {plot_path}")
    else:
        print(" No valid subjects processed.")

if __name__ == "__main__":
    main()
