# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 19:25:04 2024
This code is to train a classifier based on DenseNet264 from Monai

This code was written based on the Monai tutorial available at:
https://github.com/Project-MONAI/tutorials/blob/main/3d_classification/torch/densenet_evaluation_dict.py

The code trains a DenseNet264 model to classify LF images into different grades for each category:
- Noise, Zipper, Positioning, Banding, Motion, Contrast, and Distortion
Each category has three classes (0, 1, and 2), requiring a separate model to be trained for each category

@author: rrouhi
If you have any questions, please contact Rahimeh Rouhi at rrouhi@chla.usc.edu.

"""
import argparse
import logging
import os
import sys

import monai
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from monai.data import decollate_batch, DataLoader
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    SpatialPadd,
    LoadImaged,
    CenterSpatialCropd,
    NormalizeIntensityd,
    ToTensord,
    EnsureChannelFirstd,
)
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import (
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)

from models.densenet import AxisDenseNet264
from utils.data_utils import RandRotateByAxisd
from utils.utils import set_seed


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        help="data directory",
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        help="split csv file for train",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        help="split csv file for validation",
    )
    parser.add_argument(
        "--class_name",
        type=str,
        help="class name (e.g. Distortion)",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        default=0,
        help="rotate degree",
    )
    parser.add_argument(
        "--n_epoch",
        type=int,
        help="the number of epoch",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="batch size",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="model name (e.g. densenet254, axis_densenet264)",
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


def metrics_func(probability_tensors, true_tensor):

    true_classes = [torch.argmax(label) for label in true_tensor]
    probability_classes = [torch.argmax(label) for label in probability_tensors]

    true_combined_tensor = torch.tensor([t.item() for t in true_classes])
    prob_combined_tensor = torch.tensor([t.item() for t in probability_classes])

    # Initialize metric calculators
    # USe the weighted to handle the imbalance labels
    precision_metric = MulticlassPrecision(average="weighted", num_classes=3)
    recall_metric = MulticlassRecall(average="weighted", num_classes=3)
    f1_metric = MulticlassF1Score(average="weighted", num_classes=3)

    precision_metric.update(prob_combined_tensor, true_combined_tensor)
    recall_metric.update(prob_combined_tensor, true_combined_tensor)
    f1_metric.update(prob_combined_tensor, true_combined_tensor)

    recall = recall_metric.compute()
    f1_value = f1_metric.compute()
    precision = precision_metric.compute()

    # Print other metrics, but just return f1-score #Switch to another one, as preferred
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_value:.4f}")

    return f1_value


def main():
    args = get_args()

    set_seed(args.random_seed)

    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    class_name = args.class_name
    # Load the CSV file for labels
    DF_tr = pd.read_csv(args.train_csv)
    DF_val = pd.read_csv(args.val_csv)

    n_epoch = args.n_epoch
    n_classes = 3
    df_tr = DF_tr[["filename", class_name]]
    df_val = DF_val[["filename", class_name]]
    # Nominal to numerical
    data_tr = df_tr.to_numpy()
    data_val = df_val.to_numpy()
    # Define features and labels
    label_tr = data_tr.squeeze()
    label_val = data_val.squeeze()

    x, y, z = 256, 256, 256  # Change based on your GPU memory

    data_dir = args.data_dir

    img_tr = []
    label_tr = []

    img_val = []
    label_val = []

    # for train data
    AXIS_LABELS = {"axi": 0, "cor": 1, "sag": 2}
    for i in range(0, data_tr.shape[0]):
        subj_name = df_tr["filename"][i]

        # Check if image exists in HYP01 folder
        img_path = os.path.join(data_dir, "images", subj_name)
        if os.path.exists(img_path):
            img = img_path
        else:
            # If image doesn't exist in HYP01, check in HYP02 folder
            img_path = os.path.join(data_dir, "images", subj_name)
            if os.path.exists(img_path):
                img = img_path
            else:
                # If image doesn't exist in either folder, handle as needed
                print(f"Image not found for subject {subj_name}")
                continue  # Skip this iteration

        img_tr.append(img)
        if args.model_name == "densenet264":
            label_tr.append(df_tr[class_name][i])
        elif args.model_name == "axis_densenet264":
            axis_label = AXIS_LABELS[subj_name.split(".")[0].split("_")[-1]]
            label_tr.append(np.array([df_tr[class_name][i], axis_label]))
        print(subj_name)

    # for validation data
    for i in range(0, data_val.shape[0]):
        subj_name = df_val["filename"][i]

        # Check if image exists in HYP01 folder
        img_path = os.path.join(data_dir, "images", subj_name)
        if os.path.exists(img_path):
            img = img_path
        else:
            img_path = os.path.join(data_dir, "images", subj_name)
            if os.path.exists(img_path):
                img = img_path
            else:
                print(f"Image not found for subject {subj_name}")
                continue

        img_val.append(img)
        if args.model_name == "densenet264":
            label_val.append(df_val[class_name][i])
        elif args.model_name == "axis_densenet264":
            axis_label = AXIS_LABELS[subj_name.split(".")[0].split("_")[-1]]
            label_val.append(np.array([df_val[class_name][i], axis_label]))
        print(subj_name)

    # Contrast
    train_files = [{"img": img, "label": label} for img, label in zip(img_tr, label_tr)]
    val_files = [{"img": img, "label": label} for img, label in zip(img_val, label_val)]
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")

    # Define transforms for image
    train_transforms = Compose(
        [
            LoadImaged(keys=["img"], reader="nibabelreader"),
            EnsureChannelFirstd(keys=["img"]),
            NormalizeIntensityd(keys=["img"], nonzero=False, channel_wise=True),
            CenterSpatialCropd(keys=["img"], roi_size=(x, y, z)),
            SpatialPadd(keys=["img"], method="symmetric", spatial_size=(x, y, z)),
            RandRotateByAxisd(
                keys=["img"],
                prob=0.5,
                degrees=args.rotate,
                keep_size=True,
            ),
            ToTensord(keys=["img"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["img"], reader="nibabelreader"),
            EnsureChannelFirstd(keys=["img"]),
            NormalizeIntensityd(keys=["img"], nonzero=False, channel_wise=True),
            SpatialPadd(keys=["img"], method="symmetric", spatial_size=(x, y, z)),
            ToTensord(keys=["img"]),
        ]
    )

    post_pred = Compose([Activations(softmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=n_classes)])

    check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(
        check_ds,
        batch_size=2,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    check_data = monai.utils.misc.first(check_loader)
    print(check_data["img"].shape, check_data["label"])

    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    if args.model_name == "densenet264":
        model = monai.networks.nets.DenseNet264(
            spatial_dims=3,
            in_channels=1,
            out_channels=n_classes,
        )
    elif args.model_name == "axis_densenet264":
        model = AxisDenseNet264(
            spatial_dims=3,
            in_channels=1,
            out_channels=n_classes,
        )

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    loss_function_cls = torch.nn.CrossEntropyLoss()
    if args.model_name == "axis_densenet264":
        loss_function_axis = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)

    # start a typical PyTorch training
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    writer = SummaryWriter(args.output_dir)

    epoch_loss_values = []
    epoch_loss_cls_values = []
    if args.model_name == "axis_densenet264":
        epoch_loss_axis_values = []
    metric_values = []

    for epoch in range(n_epoch):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{n_epoch}")
        model.train()
        epoch_loss = 0
        epoch_loss_cls = 0
        if args.model_name == "axis_densenet264":
            epoch_loss_axis = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs = batch_data["img"].to(device)
            labels = batch_data["label"].to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            if args.model_name == "densenet264":
                loss_cls = loss_function_cls(outputs, labels)
                loss = loss_cls
            elif args.model_name == "axis_densenet264":
                outputs_score, outputs_axis = outputs
                loss_cls = loss_function_cls(outputs_score, labels[:, 0])
                loss_axis = loss_function_axis(outputs_axis, labels[:, 1])
                loss = loss_cls + loss_axis
            loss.backward()

            optimizer.step()

            epoch_len = len(train_ds) // train_loader.batch_size
            epoch_loss += loss.item()
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")

            epoch_loss_cls += loss_cls.item()
            writer.add_scalar(
                "train_loss_cls",
                loss_cls.item(),
                epoch_len * epoch + step,
            )
            print(f"{step}/{epoch_len}, train_loss_cls: {loss_cls.item():.4f}")

            if args.model_name == "axis_densenet264":
                epoch_loss_axis += loss_axis.item()
                writer.add_scalar(
                    "train_loss_axis",
                    loss_axis.item(),
                    epoch_len * epoch + step,
                )
                print(f"{step}/{epoch_len}, train_loss_axis: {loss_axis.item():.4f}")

        lr_scheduler.step()

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        epoch_loss_cls /= step
        epoch_loss_cls_values.append(epoch_loss_cls)
        print(f"epoch {epoch + 1} average loss_cls: {epoch_loss_cls:.4f}")
        if args.model_name == "axis_densenet264":
            epoch_loss_axis /= step
            epoch_loss_axis_values.append(epoch_loss_axis)
            print(f"epoch {epoch + 1} average loss_axis: {epoch_loss_axis:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images = val_data["img"].to(device)
                    val_labels = val_data["label"].to(device)
                    val_outputs = model(val_images)
                    if args.model_name == "axis_densenet264":
                        val_labels = val_labels[:, 0]
                        val_outputs = val_outputs[0]

                    y_pred = torch.cat([y_pred, val_outputs], dim=0)
                    y = torch.cat([y, val_labels], dim=0)

                y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
                y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]

                metric_result = metrics_func(y_pred_act, y_onehot)

                metric_values.append(metric_result.item())

                del y_pred_act, y_onehot
                if metric_result >= best_metric:
                    best_metric = metric_result
                    best_metric_epoch = epoch + 1
                    torch.save(
                        model.state_dict(),
                        f"{args.output_dir}/best_metric_model_LISA_LF_"
                        + class_name
                        + ".pth",
                    )
                    print("saved new best metric model")
                print(
                    "current epoch: {} current F1-score: {:.4f} best F1-score: {:.4f} at epoch {}".format(
                        epoch + 1,
                        metric_result,
                        best_metric,
                        best_metric_epoch,
                    )
                )
                writer.add_scalar("val_auc", metric_result, epoch + 1)

        np.save(f"{args.output_dir}/loss_tr_" + class_name + ".npy", epoch_loss_values)
        np.save(f"{args.output_dir}/val_mean_" + class_name + ".npy", metric_values)

    print(
        f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
    )
    writer.close()


if __name__ == "__main__":
    main()
