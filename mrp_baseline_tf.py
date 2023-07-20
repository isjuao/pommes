#!/usr/bin/env python
# coding: utf-8

"""
    Run from project directory (one above the DLC project mrp-basetfscorer-{date}).
    Loads and generates prediction for baseline model using DLC's *T*ime lapse *F*rame function.
"""


import os
# os.system('pip install deeplabcut[tf,modelzoo]')
import deeplabcut as dlc
import tensorflow as tf
import os
from pathlib import Path
from datetime import date
import pandas as pd
import subprocess
import datetime
import argparse

video_path = Path(os.getcwd() + "/dummy-video.mp4")

# Ensure video file exists
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
    analyzevideo=False,         # if True: a labeled video is created, else only weights downloaded
    createlabeledvideo=True,    # ? no documentation available
    copy_videos=True,           # from Colab: must leave copy_videos=True (?)
)
print(f"Path to config.yaml: {config_path}")
assert Path(config_path).is_file()

print("~~~~~~~~~~~~~~~~~~~~~~~")
print(f"{datetime.datetime.now().strftime('%Y-%m-%d [ %H:%M:%S ]')}")
print(" Running model to create predictions ...")
print("~~~~~~~~~~~~~~~~~~~~~~~")

directory = (os.getcwd() + "/../coco/val2017_subset")
assert os.path.isdir(directory)

dlc.auxiliaryfunctions.edit_config(config_path, {"batch_size": 1})

dlc.analyze_time_lapse_frames(
    config=config_path,
    directory=directory,
    frametype=".jpg",
)

print("~~~~~~~~~~~~~~~~~~~~~~~")
print(f"{datetime.datetime.now().strftime('%Y-%m-%d [ %H:%M:%S ]')}")
print(" All done!")
print("~~~~~~~~~~~~~~~~~~~~~~~")