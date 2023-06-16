#!/usr/bin/env python
# coding: utf-8

"""
    Run from project directory (one above the DLC project mrp-scorer-{date}.
"""


import os
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


create_new_folders = False  # Flag is True when specified through user on a new day
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--newday" , action='store_true',help = "Copy data into new folder since it is a new day!")
args = parser.parse_args()
if args.newday:
    create_new_folders = True

video_path = Path(os.getcwd() + "/dummy-video.mp4")

# Ensure video file exists
assert video_path.is_file()

print("~~~~~~~~~~~~~~~~~~~~~~~")
print(f"{datetime.datetime.now().strftime('%Y-%m-%d [ %H:%M:%S ]')}")
print("Creating pretrained project ...")
print("~~~~~~~~~~~~~~~~~~~~~~~")

config_path, train_config_path = dlc.create_pretrained_project(
    "mrp",
    "scorer",
    [video_path],
    videotype="mp4",
    model="full_macaque",
    analyzevideo=False,         # if True: a labeled video is created, else only weights downloaded
    createlabeledvideo=True,    # ? no documentation available
    copy_videos=True,           # from Colab: must leave copy_videos=True (?)
)
print(f"... done! {datetime.datetime.now().strftime('[ %H:%M:%S ]')}")
config_file = dlc.auxiliaryfunctions.read_plainconfig(config_path)
# Set iteration to value +1 from init_weights (possibly later than here?)
config_file["iteration"] = 1
dlc.auxiliaryfunctions.write_config(config_path, config_file)

today = date.today()
data_dir = Path(os.getcwd()) / f"mrp-scorer-{today}/labeled-data/dummy-video"
print(data_dir)

print("~~~~~~~~~~~~~~~~~~~~~~~")
print(f"{datetime.datetime.now().strftime('%Y-%m-%d [ %H:%M:%S ]')}")
print(" Loading COCO annotations into project ...")
print("~~~~~~~~~~~~~~~~~~~~~~~")

# Load annotations if desired file exists
anno_file = Path("coco2017_all.h5")
assert anno_file.is_file()
os.system(f"cp {str(anno_file)} {data_dir / 'CollectedData_scorer.h5'}")
# process = subprocess.Popen(f"cp {str(anno_file)} {data_dir / 'CollectedData_scorer.h5'}", shell=True)
# process.wait()

df = pd.read_hdf(data_dir / "CollectedData_scorer.h5")

# Extract train/test information

df_train = df[df["istrain", "", ""] == True]
is_train = df["istrain", "", ""].to_list()
train_indices = [index for index, flag in enumerate(is_train) if flag]

df_val = df[df["istrain", "", ""] == False]
is_val = [not flag for flag in is_train]
val_indices = [index for index, flag in enumerate(is_val) if flag]

# Drop flag column
df = df.drop(columns=["istrain"], axis=1, level=0)
df.to_hdf(data_dir / "CollectedData_scorer.h5", key="df", mode="w")

print(f"... done! {datetime.datetime.now().strftime('[ %H:%M:%S ]')}")
if create_new_folders:
    print("~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d [ %H:%M:%S ]')}")
    print(" Loading COCO images into project ...")
    print("~~~~~~~~~~~~~~~~~~~~~~~")

    # Copy images

    file_paths = df_train.index.to_list()
    dataset_dir = Path("../coco/train2017")
    with open("files-to-copy_train.txt", "w") as outfile:
        outfile.writelines(map(lambda path: str(dataset_dir / os.path.basename(path)) + "\n", file_paths))
    process = subprocess.Popen(f"cat files-to-copy_train.txt | xargs -I % cp % {data_dir}", shell=True)
    process.wait()
    # os.system('cat files-to-copy_train.txt | xargs -I % cp % {data_dir}')

    file_paths = df_val.index.to_list() # TODO: Change to df when still in test phase (!)
    dataset_dir = Path("../coco/val2017")
    with open("files-to-copy_val.txt", "w") as outfile:
        outfile.writelines(map(lambda path: str(dataset_dir / os.path.basename(path)) + "\n", file_paths))
    process = subprocess.Popen(f"cat files-to-copy_val.txt | xargs -I % cp % {data_dir}", shell=True)
    process.wait()
    # os.system('cat files-to-copy_val.txt | xargs -I % cp % {data_dir}')

    dlc.check_labels(config_path)
    dlc.create_training_dataset(
        config=config_path,
        trainIndices=[train_indices,],
        testIndices=[val_indices,],
        net_type="resnet_50",   # set to the same neural network as pre-trained model
    )
    print(f"... done! {datetime.datetime.now().strftime('[ %H:%M:%S ]')}")

print("~~~~~~~~~~~~~~~~~~~~~~~")
print(f"{datetime.datetime.now().strftime('%Y-%m-%d [ %H:%M:%S ]')}")
print(" Preparing and executing training ...")
print("~~~~~~~~~~~~~~~~~~~~~~~")

# Prepare and execute training

train_frac = int(round(len(df_train)/len(df), 2)*100)   # not saved by DLC when using indices for train/val splitting
# NOTE: Unsure if this is the right iteration value...
pose_config_path = Path(os.getcwd()) / f"mrp-scorer-{today}/dlc-models/iteration-1/mrp{today.strftime('%b%e').replace(' ','')}" \
                   f"-trainset{str(train_frac)}shuffle1/train/pose_cfg.yaml"
print(pose_config_path)
assert pose_config_path.is_file()
pose_config_file = dlc.auxiliaryfunctions.read_config(pose_config_path)

# In config.yaml, update the TrainingFraction to the true split
train_frac_from_config = int(config_file["TrainingFraction"][0] * 100)
config_file["TrainingFraction"] = [float(train_frac)/100]
dlc.auxiliaryfunctions.write_config(config_path, config_file)

# In pose_cfg.yaml of the latest iteration, change init_weights to the last snapshot of the pre-trained model
init_weights_path = Path(
    os.getcwd()) / f"mrp-scorer-{today}/dlc-models/iteration-0/mrp{today.strftime('%b%e').replace(' ','')}" \
                   f"-trainset{str(train_frac_from_config)}shuffle1/train/snapshot-1030000"
pose_config_file["init_weights"] = str(init_weights_path)
dlc.auxiliaryfunctions.write_config(pose_config_path, pose_config_file)

# In pose_cfg.yaml of the latest iteration, change dataset to the new data
pose_config_file["dataset"] = str(f"training-datasets/iteration-1/UnaugmentedDataSet_mrp"
                                  f"{today.strftime('%b%e').replace(' ','')}/mrp_scorer{train_frac}shuffle1.mat")
dlc.auxiliaryfunctions.write_config(pose_config_path, pose_config_file)

dlc.train_network(
    config_path,
    keepdeconvweights=False,
    maxiters=1000,
    # gputouse=None, # ?
)

"""dlc.evaluate_network(
    config_path,
    # comparisonbodyparts="all",
    rescale=False, # TODO: Needed?
    # gputouse=None, # ?
    trainingsetindex=0,  # TODO ?
)"""