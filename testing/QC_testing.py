# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 19:25:59 2024
This code was written based on Monai tutorial available at:
https://github.com/Project-MONAI/tutorials/blob/main/3d_classification/torch/densenet_evaluation_dict.py

This code is to test the trained model from QC_training.py
@author: rrouhi
If you have any questions, please contact Rahimeh Rouhi at rrouhi@chla.usc.edu
"""
import argparse
import logging
import os
import sys
import glob
import pandas as pd
import torch
import monai
from monai.data import decollate_batch, DataLoader
from monai.transforms import (
    Activations,
    Compose,
    SpatialPadd,
    LoadImaged,
    NormalizeIntensityd,
    ToTensord,
    EnsureChannelFirstd,
)
from tqdm import tqdm

from models.densenet import AxisDenseNet264
from utils.utils import set_seed


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        help="data directory",
    )
    parser.add_argument(
        "--class_name",
        type=str,
        help="class name (e.g. Distortion)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="model name (e.g. densenet254, axis_densenet264)",
    )
    parser.add_argument(
        "--weight_dir",
        type=str,
        help="weight directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="output directory",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="random seed",
    )
    args = parser.parse_args()
    return args


def extract_filenames(paths):
    filenames = []
    for path in paths:
        filename = os.path.basename(path)  # Get the base filename with extension
        filenames.append(filename)
        # filename_no_ext = os.path.splitext(filename)[0]  # Remove extension
        # filenames.append(filename_no_ext)
    return filenames


def main():
    args = get_args()

    set_seed(args.random_seed)

    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    class_name = args.class_name  # Change this based on different classes
    n_classes = 3

    # x, y, z = 128, 128, 128 #Adjust this based on available GPU memory
    x, y, z = 256, 256, 256  # Adjust this based on available GPU memory

    data_dirte = args.data_dir  # SE_HF/SE_LF

    images = glob.glob(os.path.join(data_dirte, "*.nii.gz"))
    te_files = [{"img": img} for img in zip(images)]

    te_transforms = Compose(
        [
            LoadImaged(keys=["img"], reader="nibabelreader"),
            EnsureChannelFirstd(keys=["img"]),
            NormalizeIntensityd(keys=["img"], nonzero=False, channel_wise=True),
            SpatialPadd(keys=["img"], method="symmetric", spatial_size=(x, y, z)),
            ToTensord(keys=["img"]),
        ]
    )

    post_pred = Compose([Activations(softmax=True)])

    # create a validation data loader
    te_ds = monai.data.Dataset(data=te_files, transform=te_transforms)
    te_loader = DataLoader(
        te_ds,
        batch_size=1,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    if args.model_name == "densenet264":
        model = monai.networks.nets.DenseNet264(
            spatial_dims=3,
            in_channels=1,
            out_channels=n_classes,
        ).to(device)
    elif args.model_name == "axis_densenet264":
        model = AxisDenseNet264(
            spatial_dims=3,
            in_channels=1,
            out_channels=n_classes,
        ).to(device)

    model.eval()
    state_dict = torch.load(
        f"{args.weight_dir}/best_metric_model_LISA_LF_" + class_name + ".pth"
    )
    if any(key.startswith("module.") for key in state_dict.keys()):
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    # auc_metric = ROCAUCMetric()
    with torch.no_grad():
        y_pred = torch.tensor([], dtype=torch.float32, device=device)

        for te_data in tqdm(te_loader, total=len(te_loader)):
            te_images = te_data["img"].to(device)
            te_outputs = model(te_images)
            if args.model_name == "axis_densenet264":
                te_outputs = te_outputs[0]

            y_pred = torch.cat([y_pred, te_outputs], dim=0)

        y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]

        # Assuming y_pred_act is a list of tensors
        # Move to CPU if needed
        y_pred_act = [tensor.to("cpu") for tensor in y_pred_act]
        y_pred_act_tensor = torch.stack(y_pred_act)  # Stack into a single tensor

        # Get the index of the highest probability for each row
        pre_labels_dig = torch.argmax(y_pred_act_tensor, dim=1)

    # Write the results to the excell file
    predicted_labels_list = pre_labels_dig.tolist()
    filenames = extract_filenames(images)
    DF = pd.DataFrame(
        {"Subject ID": filenames, "Pred_label_" + class_name: predicted_labels_list}
    )
    print()
    DF = DF.sort_values(by=["Subject ID"])
    DF.to_csv(
        f"{args.output_dir}/predicted_output_" + str(class_name) + ".csv",
        index=False,
    )
    print(" ******The output was generated in predicted_output...csv******")


if __name__ == "__main__":
    main()
