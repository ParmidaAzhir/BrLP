import os
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import binary_dilation, binary_erosion
import argparse

def extract_border_and_dilation(mask, struct_elem_size=3):
    struct_elem = np.ones((struct_elem_size,) * 3, dtype=bool)
    eroded = binary_erosion(mask, structure=struct_elem)
    border = mask & (~eroded)
    dilated_border = binary_dilation(border, structure=struct_elem)
    return dilated_border

def main(base_dir, struct_elem_size, output_csv):

    unc_name = "uncertainty_map.nii.gz"
    seg_name = "groundtruth_segm.nii.gz"

    results = []

    subject_dirs = [
        os.path.join(base_dir, d) for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    for subj_dir in subject_dirs:
        subj_id = os.path.basename(subj_dir)
        unc_path = os.path.join(subj_dir, unc_name)

        if not os.path.exists(unc_path):
            print(f" No uncertainty map in {subj_dir}")
            continue

        seg_path = None
        for root, _, files in os.walk(subj_dir):
            if seg_name in files:
                seg_path = os.path.join(root, seg_name)
                break

        if seg_path is None:
            print(f" No segmentation found in {subj_dir}")
            continue

        print(f"\n Processing: {subj_id}")
        try:
            seg = nib.load(seg_path).get_fdata().astype(np.int32)
            unc = nib.load(unc_path).get_fdata().astype(np.float32)
            variance_map = unc

            labels = np.unique(seg)
            labels = labels[labels != 0]

            for label in labels:
                mask = seg == label
                if not np.any(mask):
                    continue

                dilated_border = extract_border_and_dilation(mask, struct_elem_size)
                selected_variance = variance_map[dilated_border]
                mean_var = float(np.mean(selected_variance)) if selected_variance.size > 0 else np.nan

                results.append({
                    "SubjectID": subj_id,
                    "Label": int(label),
                    "MeanVarianceInDilatedBorder": mean_var
                })

        except Exception as e:
            print(f" Error processing {subj_id}: {e}")

    print(f"\n Total regions collected: {len(results)}")
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f" Saved results to {output_csv}")
    else:
        print(" No data collected — check your segmentation files or masks.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute uncertainty in dilated segmentation borders.")
    parser.add_argument('--n_dilations', type=int, default=3, help='Size of structuring element for dilation')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the input directory containing subject folders')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to save the output CSV')

    args = parser.parse_args()

    main(args.input_csv, args.n_dilations, args.output_csv)
