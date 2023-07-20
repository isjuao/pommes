#!/usr/bin/env python
# coding: utf-8

"""
    Run from project directory (one above the DLC project).
    As of now, assumes deeplabcut.evaluate_network() was called and predictions were saved in .csv file.
"""


import numpy as np
import datetime
import pandas as pd
from pathlib import Path
import pingouin as pg
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from sklearn import metrics


print("~~~~~~~~~~~~~~~~~~~~~~~")
print(f"{datetime.datetime.now().strftime('%Y-%m-%d [ %H:%M:%S ]')}")
print(" Evaluating model ...")
print("~~~~~~~~~~~~~~~~~~~~~~~")


def get_ground_truth(path):
    """Loads ground truth data from .h5 file."""

    assert path.is_file()

    df_g = pd.read_hdf(path)
    df_ground = df_g.drop("istrain", axis=1, level=0)

    return df_ground


def stack_and_filter_dfs(df_ground: pd.DataFrame, df_pred: pd.DataFrame):
    """Stacks DataFrames into longer format and filters predictions for keypoint according to likelihood cutoff."""

    pcutoff = 0.4   # like in MacaquePose paper TODO: or change to likelihood cutoff from config.yaml

    # Stack DataFrames
    df_p_stacked = df_pred.stack(level="bodyparts").reset_index(level="bodyparts").reset_index()
    df_g_stacked = df_ground.stack(level="bodyparts").reset_index(level="bodyparts").reset_index(names=["index"])

    # Drop "unsure" predictions
    # TODO: do predictions that are sure but the keypoint does not exist in truth stay?
    df_p_stacked = df_p_stacked[df_p_stacked["likelihood"] >= pcutoff].drop("likelihood", axis=1)
    print(f"Bodyparts that are predicted with confidence and thus considered: "
          f"{df_p_stacked['bodyparts'].unique()}")

    return df_g_stacked, df_p_stacked


def rmse(df_g_stacked: pd.DataFrame, df_p_stacked: pd.DataFrame):
    """Calculates RMSE for every body part, treating x/y separately."""

    # Join overlap ("sure" predictions)
    df_both = df_p_stacked.merge(df_g_stacked, on=["index", "bodyparts"], suffixes=("_pred", "_truth"))

    # Group by bodypart
    grouped = df_both.groupby("bodyparts")

    def rmse_per_bodypart(df_group):
        # # Using sk-learn definition: Implies dealing with x/y separately
        # rmse_x = mean_squared_error(y_true=df_group["x_truth"], y_pred=df_group["x_pred"], squared=False)
        # rmse_y = mean_squared_error(y_true=df_group["y_truth"], y_pred=df_group["y_pred"], squared=False)
        # return pd.Series(data=[rmse_x, rmse_y], index=["RMSE_x", "RMSE_y"])

        # Calculate Euclidean distance for each pair of truth/prediction
        df_group["eucl"] = df_group.apply(
            lambda row: euclidean([row["x_pred"], row["y_pred"]], [row["x_truth"], row["y_truth"]]), axis=1
        )
        # Calculate RMS distance
        rmse_val = np.sqrt(np.mean(np.square(df_group["eucl"])))

        return rmse_val

    res = grouped.apply(rmse_per_bodypart)
    res = res.reset_index().rename({0: "rmse"}, axis=1)

    print("........ calculated RMSE!")
    return res


def icc(df_g_stacked: pd.DataFrame, df_p_stacked: pd.DataFrame):
    """Calculates ICC for every body part, treating x/y separately."""

    # Join overlap ("sure" predictions)
    df_both = df_p_stacked.merge(df_g_stacked, on=["index", "bodyparts"], suffixes=("_pred", "_truth"))

    # Group by bodypart
    grouped = df_both.groupby("bodyparts")

    def icc_per_bodypart(df_group):
        # Preprocess into correct format
        df_group = pd.wide_to_long(df_group, ["x", "y"], i=["index", "bodyparts"], j="rater", sep="_",
                                       suffix=r'\w+')
        df_group = df_group.reset_index()
        df_group = pd.melt(df_group, id_vars=["index", "bodyparts", "rater"], value_vars=["x", "y"],
                           var_name="dimension")

        # Calculate ICCs for dimenisons
        grouped_by_dim = df_group.groupby("dimension")
        icc_x = pg.intraclass_corr(grouped_by_dim.get_group("x"), targets="index", raters="rater", ratings="value")
        # TODO specify type to ICC2
        icc_y = pg.intraclass_corr(grouped_by_dim.get_group("y"), targets="index", raters="rater", ratings="value")

        # Concat results
        icc_x = icc_x.set_index(icc_x["Type"] + "_x").drop("Type", axis=1)
        icc_y = icc_y.set_index(icc_y["Type"] + "_y").drop("Type", axis=1)
        icc_all = pd.concat([icc_x, icc_y])

        return icc_all

    res = grouped.apply(icc_per_bodypart)

    print("........ calculated ICC!")
    return res


def conf_matrix(df_g_stacked: pd.DataFrame, df_p_stacked_lh: pd.DataFrame, pcutoff: int):
    """Calculates the confusion matrix based on the pcutoff."""

    # Define positive predictions (.set_index(["index", "bodyparts"]))
    df_p_pos = df_p_stacked_lh[df_p_stacked_lh["likelihood"] > pcutoff]
    mask_p = df_p_stacked_lh.index.isin(df_p_pos.index.to_list())
    df_p_stacked_lh["pos"] = mask_p

    # Define positive ground truth (.set_index(["index", "bodyparts"]))
    df_g_pos = df_g_stacked[(df_g_stacked["x"] != 0) | (df_g_stacked["y"] != 0)]
    mask_g = df_g_stacked.index.isin(df_g_pos.index.to_list())
    df_g_stacked["pos"] = mask_g

    df_both = pd.merge(df_p_stacked_lh[["index", "bodyparts", "pos", "likelihood"]],
                       df_g_stacked[["index", "bodyparts", "pos"]],
                         on=["index", "bodyparts"],
                         how="inner", suffixes=("_p", "_g"))

    tp = len(df_both[(df_both["pos_p"]) & (df_both["pos_g"])])
    fp = len(df_both[(df_both["pos_p"]) & (df_both["pos_g"] == False)])
    fn = len(df_both[(df_both["pos_p"] == False) & (df_both["pos_g"])])
    tn = len(df_both[(df_both["pos_p"] == False) & (df_both["pos_g"] == False)])

    cm = [tn, fp, fn, tp]
    return cm


def eval(is_coco, is_baseline):
    """
        This function assumes predictions generated through DeepLabCut (e.g. evaluate_network).
        Using ground truth and prediction data, we can define our own evaluation.
    """

    # Get predictions - hardcode for now

    if is_coco:
        # df_p = pd.read_hdf("evaluation/DLC_resnet50_mrpJun23shuffle1_10000-snapshot-10000.h5")
        pred_file_name = "DLC_resnet50_mrpbaseJul19shuffle1_1030000-snapshot-1030000_coco.h5"
    else:
        pred_file_name = "DLC_resnet50_mrpbaseJul19shuffle1_1030000-snapshot-1030000_macaque.h5"
    df_p = pd.read_hdf(f"evaluation/{pred_file_name}")
    #     # parse_grounod_truth_data_file?
    # if pred_file_name == "val2017_subsetDLC_resnet50_mrpbaseJul13shuffle1_1030000.h5":
    #     df_p.index = "labeled-data/dummy-video/" + df_p.index

    # Flatten columns and index (image file paths)
    df_p.columns = df_p.columns.droplevel(level="scorer")
    if isinstance(df_p.index, pd.MultiIndex):
        images = []
        for tup in df_p.index.to_flat_index():
            images.append("/".join(tup))
        df_p.index = images

    # Stack predictions DataFrame into longer format
    df_p_stacked = df_p.stack(level="bodyparts").reset_index(level="bodyparts").reset_index()

    # Get ground truth data

    if is_coco:
        """Handle COCO ground truth data"""

        # ground_truth_file = "coco2017_all.h5"
        ground_truth_file = "coco2017_val-test.h5"       # << TODO: use
        assert Path(ground_truth_file).is_file()
        df_g = pd.read_hdf(ground_truth_file)

        if ground_truth_file == "coco2017_all.h5":
            # Get rid of indication column and adjust column index
            df_g = df_g.drop("istrain", axis=1, level=0)
        df_g.columns = df_g.columns.droplevel(level="scorer")

        # Stack DataFrame into longer format
        df_g_stacked = df_g.stack(level="bodyparts").reset_index(level="bodyparts").reset_index(names=["index"])
    else:
        """Handle MacaquePose ground truth data"""

        ground_truth_path = Path("macaque_val.h5")
        assert ground_truth_path.is_file()
        df_g = pd.read_hdf(ground_truth_path)
        df_g["image file name"] = "labeled-data/dummy-video/" + df_g["image file name"]

        # Split positions column and rename
        df_g["x"] = df_g.apply(lambda row: row["position"][0], axis=1)
        df_g["y"] = df_g.apply(lambda row: row["position"][1], axis=1)
        df_g_stacked = df_g.drop(["index", "position"], axis=1)\
            .rename({"image file name": "index"}, axis=1)
    
    # Get rid of images that do not contain both upper limbs, only evaluate on rest
    upper_limb_missing = df_g_stacked[
        df_g_stacked["bodyparts"].isin([
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"
        ]) & (
                (df_g_stacked["x"] == 0) | (df_g_stacked["y"] == 0)
        )]["index"].unique().tolist()

    df_g_stacked = df_g_stacked[~df_g_stacked["index"].isin(upper_limb_missing)]
    df_p_stacked = df_p_stacked[~df_p_stacked["index"].isin(upper_limb_missing)]

    pcutoff = 0.1
    cm = conf_matrix(df_g_stacked, df_p_stacked, pcutoff)   # TODO: plot?

    # # Drop "unsure" predictions according to likelihood cutoff
    # # (= 0.6, from pre-trained config.yaml, or 0.4, from MacaquePose paper)
    # df_p_stacked = df_p_stacked[df_p_stacked["likelihood"] >= pcutoff]
    # print(f"Bodyparts that are predicted with confidence -> are considered: {df_p_stacked['bodyparts'].unique()}")
    df_p_stacked = df_p_stacked.drop("likelihood", axis=1)

    # Calculate Metrics

    # RMSE, for x and y coordinates separately, for each bodypart
    # (DLC says MAE, but can probably infer RMSE for every bodypart using function in evaluate.py)
    # TODO: waiting for reply by MacaquePose group
    res_rmse = rmse(df_g_stacked, df_p_stacked)

    # # ICC, for x and y coordinates separately, for each bodypart
    # res_icc = icc(df_g_stacked, df_p_stacked)
    # TODO: use R code for this

    # MAP: use DLC's evaluate_assembly() from inferenceutils.py
    print("TBD")

    # PCK: calculation is just a low-pass filter with arbitrary distance threshold?
    print("TBD")

    # load model?
    # evaluate: use y_true and y_pred
    # tf.keras.metrics.RootMeanSquaredError
    # model.compile(optimizer="sgd",
    #               loss="binary_crossentropy",
    #               metrics=[km.binary_true_positive()])


# Set ground truth
is_coco = True
# Set prediction scorer name
is_baseline = True

eval(is_coco, is_baseline)

print("~~~~~~~~~~~~~~~~~~~~~~~")
print(f"{datetime.datetime.now().strftime('%Y-%m-%d [ %H:%M:%S ]')}")
print(" All done!")
print("~~~~~~~~~~~~~~~~~~~~~~~")