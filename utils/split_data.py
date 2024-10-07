import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        type=str,
        help="csv file path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="output directory",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.csv_path)
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    train_data.to_csv(f"{args.output_dir}/LISA_LF_QC_split_train.csv", index=False)
    val_data.to_csv(f"{args.output_dir}/LISA_LF_QC_split_val.csv", index=False)
