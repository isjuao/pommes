#!/usr/bin/env python
# coding: utf-8

"""
    Run from project directory (one above the DLC project mrpbase-basescorer-{date}).
    Loads baseline model and generates predictions in evaluation-results subdir in the DLC project directory.
"""


import deeplabcut as dlc
import os
from pathlib import Path
from datetime import date
import pandas as pd
import subprocess
import datetime


# Set basic settings
video_name = "dummy-video"      # name of .mp4 file (can be dummy)
is_coco = True                  # use COCO for validation
if is_coco:
    anno_file = "coco2017_all.h5"
    # for final evaluation: anno_file = "coco2017_val-test.h5"
else:
    anno_file = "macaque_val_wide.h5"

today = date.today()
video_path = Path(os.getcwd() + f"/{video_name}.mp4")
data_dir = Path(os.getcwd()) / f"mrpbase-basescorer-{today}/labeled-data/{video_name}"

# Ensure video file exists
assert video_path.is_file()

print("~~~~~~~~~~~~~~~~~~~~~~~")
print(f"{datetime.datetime.now().strftime('%Y-%m-%d [ %H:%M:%S ]')}")
print(" Creating pretrained project ...")
print("~~~~~~~~~~~~~~~~~~~~~~~")

# LOAD DATA

# Load baseline model
config_path, pose_config_path_it0 = dlc.create_pretrained_project(
    "mrpbase",
    "basescorer",
    [video_path],
    videotype="mp4",
    model="full_macaque",
    analyzevideo=False,
    createlabeledvideo=True,
    copy_videos=True,           # from Colab: must leave copy_videos=True
)
print(f"Path to config.yaml: {config_path}")
assert Path(config_path).is_file()

# Set batch size to 1
dlc.auxiliaryfunctions.edit_config(config_path, {"batch_size": 1})

print("~~~~~~~~~~~~~~~~~~~~~~~")
print(f"{datetime.datetime.now().strftime('%Y-%m-%d [ %H:%M:%S ]')}")
print(" Loading annotations into project ...")
print("~~~~~~~~~~~~~~~~~~~~~~~")

# Load annotations (if desired file exists)
assert Path(anno_file).is_file()
assert data_dir.is_dir()
process = subprocess.Popen(f"cp {anno_file} {data_dir / 'CollectedData_basescorer.h5'}", shell=True)
process.wait()

df = pd.read_hdf(data_dir / "CollectedData_basescorer.h5")

# Extract train(= 0, since no re-training necessary for baseline)/test information
val_indices = []
if is_coco:
    if anno_file == "coco2017_all.h5":
        # When using all data (coco2017_all.h5), filter to validation images
        is_val = [not flag for flag in df["istrain", "", ""].to_list()]
        df = df.iloc[is_val].drop(columns=["istrain"], axis=1, level=0)

    val_indices = df.reset_index().index.to_list()

    df = df.rename({"scorer": "basescorer"}, axis=1)
    df.to_hdf(data_dir / "CollectedData_basescorer.h5", key="df", mode="w")
else:
    val_indices = df.reset_index().index.to_list()
    df.to_hdf(data_dir / "CollectedData_basescorer.h5", key="df", mode="w")

print("~~~~~~~~~~~~~~~~~~~~~~~")
print(f"{datetime.datetime.now().strftime('%Y-%m-%d [ %H:%M:%S ]')}")
print(" Loading images into project ...")
print("~~~~~~~~~~~~~~~~~~~~~~~")

# Copy images
file_paths = df.index.to_list()
txt_file = ""
if is_coco:
    dataset_dir = Path("../coco/val2017")
    if anno_file == "coco2017_all.h5":
        # Using all data (coco2017_all.h5)
        txt_file = "files-to-copy_val.txt"
    else:
        # Using test set
        txt_file = "files-to-copy_test.txt"

    with open(txt_file, "w") as outfile:
        outfile.writelines(map(lambda path: str(dataset_dir / os.path.basename(path)) + "\n", file_paths))
    process = subprocess.Popen(f"cat {txt_file} | xargs -I % cp % {data_dir}", shell=True)
    process.wait()
else:
    dataset_dir = Path("../coco/v1/images")
    txt_file = "testfiles-to-copy.txt"

    with open(txt_file, "w") as outfile:
        outfile.writelines(map(lambda path: str(dataset_dir / os.path.basename(path)) + "\n", file_paths))
    process = subprocess.Popen(f"cat {txt_file} | xargs -I % cp % {data_dir}", shell=True)
    process.wait()

train_frac = 0
train_indices = [val_indices[-1]]   # needs to be non-empty for DLC to create .pickle file in training-datasets subdir
val_indices = val_indices[:-1]

# Safety check
dlc.check_labels(config_path)
print(f"Amount of training pictures: {len(train_indices)}")
print(f"Amount of validation pictures: {len(val_indices)}")

dlc.create_training_dataset(
    config=config_path,
    trainIndices=[train_indices,],
    testIndices=[val_indices,],
    net_type="resnet_50",   # set to the same neural network as pre-trained model
)

# ENSURE CORRECT SETTINGS

pose_config_path = Path(os.getcwd()) / f"mrpbase-basescorer-{today}/dlc-models/iteration-0/" \
                                       f"mrpbase{today.strftime('%b%e').replace(' ','')}" \
                   f"-trainset95shuffle1/train/pose_cfg.yaml"
print(f"Path to pose_cfg.yaml: {pose_config_path}")
assert pose_config_path.is_file()

# Ensure correct project_path in pose_cfg.yaml
dlc.auxiliaryfunctions.edit_config(pose_config_path,
                                   {"project_path": str(Path(os.getcwd()) / f"mrpbase-basescorer-{today}")})

# Rename files because DLC uses config to find model (TrainingFraction of ModelZoo model = 95 != 0)
picklefile_path = f"mrpbase-basescorer-{today}/training-datasets/iteration-0/UnaugmentedDataSet_mrpbase" \
                  f"{today.strftime('%b%e').replace(' ','')}/Documentation_data-mrpbase_0shuffle1.pickle"
print(f"... Renaming file at {picklefile_path}")
assert os.path.isfile(Path(picklefile_path))
process = subprocess.Popen(
    f"mv {picklefile_path}"
    f" mrpbase-basescorer-{today}/training-datasets/iteration-0/UnaugmentedDataSet_mrpbase"
    f"{today.strftime('%b%e').replace(' ','')}/Documentation_data-mrpbase_95shuffle1.pickle", shell=True
)
process.wait()
matfile_path = f"mrpbase-basescorer-{today}/training-datasets/iteration-0/UnaugmentedDataSet_mrpbase" \
               f"{today.strftime('%b%e').replace(' ','')}/mrpbase_basescorer0shuffle1.mat"
print(f"... Renaming file at {matfile_path}")
assert os.path.isfile(Path(matfile_path))
process = subprocess.Popen(
    f"mv {matfile_path}"
    f" mrpbase-basescorer-{today}/training-datasets/iteration-0/UnaugmentedDataSet_mrpbase"
    f"{today.strftime('%b%e').replace(' ','')}/mrpbase_basescorer95shuffle1.mat", shell=True
)
process.wait()

# GENERATE PREDICTIONS

print("~~~~~~~~~~~~~~~~~~~~~~~")
print(f"{datetime.datetime.now().strftime('%Y-%m-%d [ %H:%M:%S ]')}")
print(" Generating predictions ...")
print("~~~~~~~~~~~~~~~~~~~~~~~")

dlc.evaluate_network(
    config_path,
    gputouse=0,
)

print("~~~~~~~~~~~~~~~~~~~~~~~")
print(f"{datetime.datetime.now().strftime('%Y-%m-%d [ %H:%M:%S ]')}")
print(" All done!")
print("~~~~~~~~~~~~~~~~~~~~~~~")