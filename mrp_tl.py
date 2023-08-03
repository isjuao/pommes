#!/usr/bin/env python
# coding: utf-8

"""
    Run from project directory (one above the DLC project mrp-scorer-{date}).
"""


import deeplabcut as dlc
import os
from pathlib import Path
from datetime import date
import pandas as pd
import subprocess
import datetime
import argparse


# Add argument to specify creation of new DLC project folder
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--newday", action='store_true', help="Copy data into new folder since it is a new day")
args = parser.parse_args()

# Set basic settings
video_name = "dummy-video"      # name of .mp4 file (can be dummy)
video_path = Path(os.getcwd() + f"/{video_name}.mp4")
assert video_path.is_file()

today = date.today()
data_dir = Path(os.getcwd()) / f"mrp-scorer-{today}/labeled-data/{video_name}"
print(data_dir)

print("~~~~~~~~~~~~~~~~~~~~~~~")
print(f"{datetime.datetime.now().strftime('%Y-%m-%d [ %H:%M:%S ]')}")
print(" Creating pretrained project ...")
print("~~~~~~~~~~~~~~~~~~~~~~~")

# LOAD DATA

# Load baseline model
config_path, pose_config_path_it0 = dlc.create_pretrained_project(
    "mrp",
    "scorer",
    [video_path],
    videotype="mp4",
    model="full_macaque",
    analyzevideo=False,
    createlabeledvideo=True,
    copy_videos=True,           # from Colab: must leave copy_videos=True
)
print(f"Path to config.yaml: {config_path}")
assert Path(config_path).is_file()

# Set iteration to value +1
dlc.auxiliaryfunctions.edit_config(config_path, {"iteration": 1})

print("~~~~~~~~~~~~~~~~~~~~~~~")
print(f"{datetime.datetime.now().strftime('%Y-%m-%d [ %H:%M:%S ]')}")
print(" Loading COCO annotations into project ...")
print("~~~~~~~~~~~~~~~~~~~~~~~")

# Load annotations
anno_file = Path("coco2017_all.h5")
assert anno_file.is_file()
process = subprocess.Popen(f"cp {str(anno_file)} {data_dir / 'CollectedData_scorer.h5'}", shell=True)
process.wait()

df = pd.read_hdf(data_dir / "CollectedData_scorer.h5")

# Extract train/test information

# Define training set
df_train = df[df["istrain", "", ""] == True]
is_train = df["istrain", "", ""].to_list()
train_indices = [index for index, flag in enumerate(is_train) if flag]

# Define validation set
df_val = df[df["istrain", "", ""] == False]
is_val = [not flag for flag in is_train]
val_indices = [index for index, flag in enumerate(is_val) if flag]

# Drop flag column after defining
df = df.drop(columns=["istrain"], axis=1, level=0)
df.to_hdf(data_dir / "CollectedData_scorer.h5", key="df", mode="w")

if args.newday:
    # New project folder to be created

    print("~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d [ %H:%M:%S ]')}")
    print(" Loading COCO images into project ...")
    print("~~~~~~~~~~~~~~~~~~~~~~~")

    # Copy images

    def copy_imgs(dataset_dir, file_paths, txt_file: str):
        with open(txt_file, "w") as outfile:
            outfile.writelines(map(lambda path: str(dataset_dir / os.path.basename(path)) + "\n", file_paths))
        process = subprocess.Popen(f"cat files-to-copy_train.txt | xargs -I % cp % {data_dir}", shell=True)
        process.wait()

    file_paths = df_train.index.to_list()
    dataset_dir = Path("../coco/train2017")
    copy_imgs(dataset_dir, file_paths, "files-to-copy_train.txt")

    file_paths = df_val.index.to_list()
    dataset_dir = Path("../coco/val2017")
    copy_imgs(dataset_dir, file_paths, "files-to-copy_val.txt")

    dlc.check_labels(config_path)
    dlc.create_training_dataset(
        config=config_path,
        trainIndices=[train_indices,],
        testIndices=[val_indices,],
        net_type="resnet_50",   # set to the same neural network as pre-trained model
    )

# PREPARE TRAINING

print("~~~~~~~~~~~~~~~~~~~~~~~")
print(f"{datetime.datetime.now().strftime('%Y-%m-%d [ %H:%M:%S ]')}")
print(" Preparing training ...")
print("~~~~~~~~~~~~~~~~~~~~~~~")

# Define TrainingFraction as not saved by DLC when using indices for train/val splitting
train_frac = int(round(len(df_train)/len(df), 2)*100)

# Get training config
pose_config_path = Path(os.getcwd()) / f"mrp-scorer-{today}/dlc-models/iteration-1/mrp{today.strftime('%b%e').replace(' ','')}" \
                   f"-trainset{str(train_frac)}shuffle1/train/pose_cfg.yaml"
print(f"Path to pose_cfg.yaml: {pose_config_path}")
assert pose_config_path.is_file()

# Ensure correct project_path in pose_cfg.yaml
dlc.auxiliaryfunctions.edit_config(pose_config_path, {"project_path": str(Path(os.getcwd()) / f"mrp-scorer-{today}")})

# In config.yaml, update the TrainingFraction to the true split
train_frac_from_config = int(dlc.auxiliaryfunctions.read_plainconfig(config_path)["TrainingFraction"][0] * 100)
dlc.auxiliaryfunctions.edit_config(config_path, {"TrainingFraction": [float(train_frac)/100]})

# In pose_cfg.yaml of the latest iteration, change init_weights to the last snapshot of the pre-trained model
init_weights_path = Path(
    os.getcwd()) / f"mrp-scorer-{today}/dlc-models/iteration-0/mrp{today.strftime('%b%e').replace(' ','')}" \
                   f"-trainset{str(train_frac_from_config)}shuffle1/train/snapshot-1030000"
dlc.auxiliaryfunctions.edit_config(pose_config_path, {"init_weights": str(init_weights_path)})

# In pose_cfg.yaml of the latest iteration, change dataset to the new data
mat_path = str(f"training-datasets/iteration-1/UnaugmentedDataSet_mrp"
                                  f"{today.strftime('%b%e').replace(' ','')}/mrp_scorer{train_frac}shuffle1.mat")
dlc.auxiliaryfunctions.edit_config(pose_config_path, {"dataset": str(mat_path)})

# EXECUTE TRAINING

print("~~~~~~~~~~~~~~~~~~~~~~~")
print(f"{datetime.datetime.now().strftime('%Y-%m-%d [ %H:%M:%S ]')}")
print(" Starting training ...")
print("~~~~~~~~~~~~~~~~~~~~~~~")

dlc.train_network(
    config_path,
    keepdeconvweights=False,
    maxiters=10000,
    gputouse=0,
)

# GENERATE PREDICTIONS

print("~~~~~~~~~~~~~~~~~~~~~~~")
print(f"{datetime.datetime.now().strftime('%Y-%m-%d [ %H:%M:%S ]')}")
print(" Generate predictions ...")
print("~~~~~~~~~~~~~~~~~~~~~~~")

dlc.evaluate_network(
    config_path,
    gputouse=0,
)

print("~~~~~~~~~~~~~~~~~~~~~~~")
print(f"{datetime.datetime.now().strftime('%Y-%m-%d [ %H:%M:%S ]')}")
print(" All done!")
print("~~~~~~~~~~~~~~~~~~~~~~~")