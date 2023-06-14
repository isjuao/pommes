#!/usr/bin/env python
# coding: utf-8


import os
# os.system('pip install deeplabcut[tf,modelzoo]')
import deeplabcut as dlc
import tensorflow
import os
from pathlib import Path
from datetime import date
import pandas as pd

video_path = Path(os.getcwd() + "/dummy-video.mp4")

# Ensure video file exists
assert video_path.is_file()

print("### Creating pretrained project ...")

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
print("... done!")
config_file = dlc.auxiliaryfunctions.read_plainconfig(config_path)
# Set iteration to value +1 from init_weights (possibly later than here?)
config_file["iteration"] = 1

today = date.today()
data_dir = Path(os.getcwd() + f"/mrp-scorer-{today}/labeled-data/dummy-video")
print(data_dir)

print("### Loading COCO annotations into project ...")

# Load annotations if desired file exists
anno_file = Path("coco2017_all.h5")
assert anno_file.is_file()
os.system('cp {str(anno_file)} {data_dir / "CollectedData_scorer.h5"}')

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

print("... done!")
print("### Loading COCO images into project ...")

# Copy images

file_paths = df_train.index.to_list()
dataset_dir = Path("../coco/train2017")
with open("files-to-copy_train.txt", "w") as outfile:
    outfile.writelines(map(lambda path: str(dataset_dir / os.path.basename(path)) + "\n", file_paths))
os.system('cat files-to-copy_train.txt | xargs -I % cp % {data_dir}')

file_paths = df_val.index.to_list() # TODO: Change to df when still in test phase (!)
dataset_dir = Path("../coco/val2017")
with open("files-to-copy_val.txt", "w") as outfile:
    outfile.writelines(map(lambda path: str(dataset_dir / os.path.basename(path)) + "\n", file_paths))
os.system('cat files-to-copy_val.txt | xargs -I % cp % {data_dir}')

dlc.check_labels(config_path)
dlc.create_training_dataset(
    config=config_path,
    trainIndices=[train_indices,],
    testIndices=[val_indices,],
    net_type="resnet_50",   # set to the same neural network as pre-trained model
)

print("... done!")
print("### Preparing and executing training ...")

# Prepare and execute training

# unsure if this is the right iteration value...
pose_config_path = Path(os.getcwd()) / f"/mrp-scorer-{today}/dlc-models/iteration-1/mrp{today.strftime('%b%e').replace(' ','')}-trainset{str(int(config_file['TrainingFraction'][0] * 100))}shuffle1/train/pose_cfg.yaml"
print(pose_config_path)
assert pose_config_path.is_file()
pose_config_file = dlc.auxiliaryfunctions.read_plainconfig(pose_config_path)

# In pose_cfg.yaml of the latest iteration, change init_weights to the last snapshot of the pre-trained model
init_weights_path = Path(os.getcwd()) / f"/mrp-scorer-{today}/dlc-models/iteration-0/mrp{today.strftime('%b%e').replace(' ','')}-trainset{str(int(config_file['TrainingFraction'][0] * 100))}shuffle1/train/snapshot-1030000"
pose_config_file["init_weights"] = str(init_weights_path)

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