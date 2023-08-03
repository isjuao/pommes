#!/usr/bin/env python
# coding: utf-8

"""
    Run from project directory (one above the DLC project mrp-basetfscorer-{date}).
    Loads and generates prediction for baseline model using DLC's *T*ime lapse *F*rame function.
"""


import deeplabcut as dlc
import os
from pathlib import Path
from datetime import date
import datetime


video_name = "dummy-video"          # name of .mp4 file (can be dummy)
img_dir = "/../coco/val2017_subset" # name of folder with .jpg images
video_path = Path(os.getcwd() + f"/{video_name}.mp4")
assert video_path.is_file()

today = date.today()

print("~~~~~~~~~~~~~~~~~~~~~~~")
print(f"{datetime.datetime.now().strftime('%Y-%m-%d [ %H:%M:%S ]')}")
print(" Creating pretrained project ...")
print("~~~~~~~~~~~~~~~~~~~~~~~")

config_path, pose_config_path_it0 = dlc.create_pretrained_project(
    "mrpbasetf",
    "basetfscorer",
    [video_path],
    videotype="mp4",
    model="full_macaque",
    analyzevideo=False,
    createlabeledvideo=True,
    copy_videos=True,           # from Colab: must leave copy_videos=True
)

# Assert correct config paths and contents
print(f"Path to config.yaml: {config_path}")
assert Path(config_path).is_file()
directory = (os.getcwd() + img_dir)
assert os.path.isdir(directory)
dlc.auxiliaryfunctions.edit_config(config_path, {"batch_size": 1})

print("~~~~~~~~~~~~~~~~~~~~~~~")
print(f"{datetime.datetime.now().strftime('%Y-%m-%d [ %H:%M:%S ]')}")
print(" Running model to create predictions ...")
print("~~~~~~~~~~~~~~~~~~~~~~~")

dlc.analyze_time_lapse_frames(
    config=config_path,
    directory=directory,
    frametype=".jpg",
)

print("~~~~~~~~~~~~~~~~~~~~~~~")
print(f"{datetime.datetime.now().strftime('%Y-%m-%d [ %H:%M:%S ]')}")
print(" All done!")
print("~~~~~~~~~~~~~~~~~~~~~~~")