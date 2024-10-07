# Task 1: Quality Assessment

## Task Description

This task involves assessing seven categories of artifacts in ultra-low-field (0.064T) MRI images taken by Hyperfine SWOOP across three sites. The artifacts to be evaluated are: motion, contrast, noise, distortion, zipper, positioning, and banding. The goal is to assess the quality of these lower-resolution scans and predict their diagnostic reliability, especially in detecting gross anatomical abnormalities. Performance of the algorithms will be measured using five key metrics: Accuracy, F1-score, F2-score, Precision, and Recall.

## Data Structure

The data is organized as follows:

```
data
├──train
│  ├──images
│  │  ├──LISA_0001_LF_axi.nii.gz
│  │  ├──...
│  │  └──LISA_0001_LF_sag.nii.gz
│  └──labels
│     ├──labels.csv
│     └──split
│        ├──LISA_LF_QC_split_train.csv
│        └──LISA_LF_QC_split_val.csv
└──val
   └──images
      ├──LISA_VALIDATION_0001_LF_axi.nii.gz
      ├──...
      └──LISA_VALIDATION_0014_LF_sag.nii.gz
```

## Codes

### Docker Environment

All code is tested and runs within a Docker container. To build and run the container, use the following commands:


```
docker build . -t wooks527/lisa2024_task1_qa
```

```
docker run -itd \
  --name lisa2024_task1_qa \
  -v [DATA_DIR]:/data/lisa2024 \
  -v [PROJECT_DIR]:/workspace \
  --device=nvidia.com/gpu=all \
  --shm-size=320g \
  wooks527/lisa2024_task1_qa:latest
```
- `DATA_DIR`: Path to the directory where the data is located.
- `PROJECT_DIR`: Path to the directory where the code is stored.

### Train

```python
PYTHONPATH=. python ./training/QC_training.py \
  --data_dir /data/lisa2024/task1/train \
  --train_csv /data/lisa2024/task1/train/labels/split/LISA_LF_QC_split_train.csv \
  --val_csv /data/lisa2024/task1/train/labels/split/LISA_LF_QC_split_val.csv \
  --class_name Noise \
  --rotate 0 \
  --n_epoch 100 \
  --batch_size 4 \
  --model_name densenet264 \
  --output_dir ./results/baseline/noise
```

### Test

```python
PYTHONPATH=. python ./testing/QC_testing.py \
  --data_dir [DATA_DIR]/val/images \
  --class_name Noise \
  --model_name densenet264 \
  --weight_dir ./results/baseline/noise \
  --output_dir ./results/baseline/noise
```
- `DATA_DIR`: Path to the directory where the data is located.

### Combine Results

After training and testing for 7 artifactes, combine the results using the following command:

```python
python ./utils/combine_results.py \
  --result_dir ./results/baseline \
  --output_path ./results/baseline/LISA_LF_QC_predictions.csv
```

## References

- https://www.synapse.org/Synapse:syn55249552/wiki/626951
- https://github.com/LISA2024Challenge/Task1
