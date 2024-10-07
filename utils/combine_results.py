import argparse
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_dir",
        type=str,
        help="result directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="output directory",
    )
    args = parser.parse_args()

    cls_names = [
        "Noise",
        "Zipper",
        "Positioning",
        "Banding",
        "Motion",
        "Contrast",
        "Distortion",
    ]
    dfs = [
        pd.read_csv(
            f"{args.result_dir}/{cls_name.lower()}/predicted_output_{cls_name}.csv"
        )
        for cls_name in cls_names
    ]
    combined_df = dfs[0]
    for df in dfs[1:]:
        combined_df = combined_df.merge(df, on="Subject ID")

    col_names = ["Subject ID"]
    col_names.extend([f"Pred_label_{cls_name}" for cls_name in cls_names])
    columns = {col_name: i for i, col_name in enumerate(col_names)}
    combined_df = combined_df.rename(columns=columns)
    combined_df.to_csv(args.output_path)
