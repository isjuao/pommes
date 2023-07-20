#!/usr/bin/env python
# coding: utf-8

"""
    Run from project directory (one above the DLC project mrpbase-basescorer-{date}).
    Loads and generates prediction for baseline model.
"""


import os
import sys
# os.system('pip install deeplabcut[tf,modelzoo]')
import deeplabcut as dlc
import tensorflow
import os
from pathlib import Path
from datetime import date
import pandas as pd
import subprocess
import datetime
import argparse


use_coco = True

today = date.today()

video_path = Path(os.getcwd() + "/dummy-video.mp4")
data_dir = Path(os.getcwd()) / f"mrpbase-basescorer-{today}/labeled-data/dummy-video"

# Ensure video file exists
assert video_path.is_file()

print("~~~~~~~~~~~~~~~~~~~~~~~")
print(f"{datetime.datetime.now().strftime('%Y-%m-%d [ %H:%M:%S ]')}")
print(" Creating pretrained project ...")
print("~~~~~~~~~~~~~~~~~~~~~~~")

config_path, pose_config_path_it0 = dlc.create_pretrained_project(
    "mrpbase",
    "basescorer",
    [video_path],
    videotype="mp4",
    model="full_macaque",
    analyzevideo=False,         # if True: a labeled video is created, else only weights downloaded
    createlabeledvideo=True,    # ? no documentation available
    copy_videos=True,           # from Colab: must leave copy_videos=True (?)
)
print(f"Path to config.yaml: {config_path}")
assert Path(config_path).is_file()

# TODO: Set iteration to 1 here too?

# Set batch size to 1
dlc.auxiliaryfunctions.edit_config(config_path, {"batch_size": 1})

print("~~~~~~~~~~~~~~~~~~~~~~~")
print(f"{datetime.datetime.now().strftime('%Y-%m-%d [ %H:%M:%S ]')}")
print(" Loading annotations into project ...")
print("~~~~~~~~~~~~~~~~~~~~~~~")

# Load annotations (if desired file exists)
if use_coco:
    # Use COCO for validation
    anno_file = "coco2017_all.h5"
    # anno_file = "coco2017_val-test.h5"    # for final evaluation
else:
    # Use MacaquePose for validation
    anno_file = "macaque_val_wide.h5"
assert Path(anno_file).is_file()
assert data_dir.is_dir()
process = subprocess.Popen(f"cp {anno_file} {data_dir / 'CollectedData_basescorer.h5'}", shell=True)
process.wait()
# os.system(f"cp {str(anno_file)} {data_dir / 'CollectedData_basescorer.h5'}")

df = pd.read_hdf(data_dir / "CollectedData_basescorer.h5")

# Extract train(=0)/test information

val_indices = []
if use_coco:
    if anno_file == "coco2017_all.h5":
        # When using all data (coco2017_all.h5)

        # Filter to validation images
        is_val = [not flag for flag in df["istrain", "", ""].to_list()]
        # val_indices = [index for index, flag in enumerate(is_val) if flag]
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

if use_coco:
    file_paths = df.index.to_list()
    dataset_dir = Path("../coco/val2017")

    if anno_file == "coco2017_all.h5":
        # When using all data (coco2017_all.h5), we need to split

        with open("files-to-copy_val.txt", "w") as outfile:
            outfile.writelines(map(lambda path: str(dataset_dir / os.path.basename(path)) + "\n", file_paths))
        process = subprocess.Popen(f"cat files-to-copy_val.txt | xargs -I % cp % {data_dir}", shell=True)
        process.wait()

        # new_train_frac = int(
        #     round(len(df_train) / len(df), 2) * 100)  # not saved by DLC when using indices for train/val splitting
    else:
        # Using pre-sampled MacaquePose (validation) set

        with open("files-to-copy_test.txt", "w") as outfile:
            outfile.writelines(map(lambda path: str(dataset_dir / os.path.basename(path)) + "\n", file_paths))
        process = subprocess.Popen(f"cat files-to-copy_test.txt | xargs -I % cp % {data_dir}", shell=True)
        process.wait()
else:
    # Using pre-sampled MacaquePose (validation) set

    file_paths = df.index.to_list()
    dataset_dir = Path("../coco/v1/images")
    with open("testfiles-to-copy.txt", "w") as outfile:
        outfile.writelines(map(lambda path: str(dataset_dir / os.path.basename(path)) + "\n", file_paths))
    process = subprocess.Popen(f"cat testfiles-to-copy.txt | xargs -I % cp % {data_dir}", shell=True)
    process.wait()

# Trying to get DLC to still create .pickle file in training-datasets subdirectory
train_frac = 0
train_indices = [val_indices[-1]]
val_indices = val_indices[:-1]

dlc.check_labels(config_path)

print(f"Amount of training pictures: {len(train_indices)}")
print(f"Amount of validation pictures: {len(val_indices)}")

dlc.create_training_dataset(
    config=config_path,
    trainIndices=[train_indices,],
    testIndices=[val_indices,],
    net_type="resnet_50",   # set to the same neural network as pre-trained model
)

# NOTE: Unsure if this is the right iteration value...
pose_config_path = Path(os.getcwd()) / f"mrpbase-basescorer-{today}/dlc-models/iteration-0/" \
                                       f"mrpbase{today.strftime('%b%e').replace(' ','')}" \
                   f"-trainset95shuffle1/train/pose_cfg.yaml"
print(f"Path to pose_cfg.yaml: {pose_config_path}")
assert pose_config_path.is_file()

# Trying to fix wrong project_path in pose_cfg.yaml (here right after create_pretrained..., also on new day?)
dlc.auxiliaryfunctions.edit_config(pose_config_path, {"project_path": str(Path(os.getcwd()) / f"mrpbase-basescorer"
                                                                                              f"-{today}")})
# TODO: actually don't because we don't want retraining
# # In config.yaml, update the TrainingFraction to the true split
# train_frac_from_config = int(dlc.auxiliaryfunctions.read_plainconfig(config_path)["TrainingFraction"][0] * 100)
# dlc.auxiliaryfunctions.edit_config(config_path, {"TrainingFraction": [float(train_frac)/100]})

# # In pose_cfg.yaml of the latest iteration, change init_weights to the last snapshot of the pre-trained model
# init_weights_path = Path(
#     os.getcwd()) / f"mrpbase-basescorer-{today}/dlc-models/iteration-0/mrpbase{today.strftime('%b%e').replace(' ','')}" \
#                    f"-trainset{str(train_frac_from_config)}shuffle1/train/snapshot-1030000"
# dlc.auxiliaryfunctions.edit_config(pose_config_path, {"init_weights": str(init_weights_path)})

# TODO: actually don't because we don't want retraining
# # In pose_cfg.yaml of the latest iteration (here: 0), change dataset to the new data
# mat_path = str(f"training-datasets/iteration-0/UnaugmentedDataSet_mrpbase"
#                                   f"{today.strftime('%b%e').replace(' ','')}/mrpbase_basescorer{train_frac}shuffle1.mat")
# dlc.auxiliaryfunctions.edit_config(pose_config_path, {"dataset": str(mat_path)})

print("~~~~~~~~~~~~~~~~~~~~~~~")
print(f"{datetime.datetime.now().strftime('%Y-%m-%d [ %H:%M:%S ]')}")
print(" Evaluating model ...")
print("~~~~~~~~~~~~~~~~~~~~~~~")

# TODO: Unsure if need to set TrainingFraction to the pre-trained 95 or sth
#       but also unsure if it finds the right data to evaluate on then!
#       In this case, might need to casually edit some file names.

# Okay, try to rename files because config needed to find model
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

dlc.evaluate_network(
    config_path,
    # comparisonbodyparts="all",
    # rescale=False, # Needed?
    gputouse=0,
    # trainingsetindex=0,  # ?
)

print("~~~~~~~~~~~~~~~~~~~~~~~")
print(f"{datetime.datetime.now().strftime('%Y-%m-%d [ %H:%M:%S ]')}")
print(" All done!")
print("~~~~~~~~~~~~~~~~~~~~~~~")