import argparse
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Compute Pearson and Spearman correlations per label.")
    parser.add_argument("--input_csv", required=True, help="Path to combined input CSV with uncertainty and error stats.")
    parser.add_argument("--output_csv", required=True, help="Where to save the per-label correlation results.")
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.input_csv):
        print(f" Input CSV not found: {args.input_csv}")
        return

    # Load data
    df = pd.read_csv(args.input_csv)

    # Remove summary row if present
    df = df[df["Label"] != "ALL"]

    # Ensure label is treated as integer
    df["Label"] = df["Label"].astype(int)

    # Remove incomplete rows
    df_clean = df.dropna(subset=["MeanVarianceInDilatedBorder", "MeanSquaredErrorInDilatedBorder"])

    # Compute correlations per label
    results = []
    for label, group in df_clean.groupby("Label"):
        if len(group) < 2:
            continue  # Not enough data to compute correlation

        pearson_corr, _ = pearsonr(group["MeanVarianceInDilatedBorder"], group["MeanSquaredErrorInDilatedBorder"])
        spearman_corr, _ = spearmanr(group["MeanVarianceInDilatedBorder"], group["MeanSquaredErrorInDilatedBorder"])

        results.append({
            "Label": label,
            "NumSubjects": len(group),
            "PearsonCorrelation": round(pearson_corr, 4),
            "SpearmanCorrelation": round(spearman_corr, 4)
        })

    if not results:
        print(" No labels with sufficient data for correlation.")
        return

    out_df = pd.DataFrame(results)
    out_df.sort_values("Label", inplace=True)
    out_df.to_csv(args.output_csv, index=False)

    print(f" Saved correlation results per label to:\n{args.output_csv}")

if __name__ == "__main__":
    main()
