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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from mrp_plotting import plot_conf_matrix, plot_single_rmse


print("~~~~~~~~~~~~~~~~~~~~~~~")
print(f"{datetime.datetime.now().strftime('%Y-%m-%d [ %H:%M:%S ]')}")
print(" Evaluating model ...")
print("~~~~~~~~~~~~~~~~~~~~~~~")


def get_ground_truth(path):
    """
        CURRENTLY NOT IN USE - needs review before usage
        Loads ground truth data from .h5 file.
    """

    assert path.is_file()
    df_g = pd.read_hdf(path)
    df_ground = df_g.drop("istrain", axis=1, level=0)

    return df_ground


def stack_and_filter_dfs(df_ground: pd.DataFrame, df_pred: pd.DataFrame):
    """
        CURRENTLY NOT IN USE - needs review before usage
        Stacks DataFrames into longer format and filters predictions for keypoint according to likelihood cutoff.
    """

    # Stack DataFrames
    df_p_stacked = df_pred.stack(level="bodyparts").reset_index(level="bodyparts").reset_index()
    df_g_stacked = df_ground.stack(level="bodyparts").reset_index(level="bodyparts").reset_index(names=["index"])

    # Drop "unsure" predictions
    df_p_stacked = df_p_stacked[df_p_stacked["likelihood"] >= pcutoff].drop("likelihood", axis=1)
    print(f"Bodyparts that are predicted with confidence and thus considered: "
          f"{df_p_stacked['bodyparts'].unique()}")

    return df_g_stacked, df_p_stacked


def icc(df_g_stacked: pd.DataFrame, df_p_stacked: pd.DataFrame):
    """
        CURRENTLY NOT IN USE - needs review before usage
        Calculates ICC for every body part, treating x/y separately.
    """

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

        # Calculate ICCs for dimenisons, TODO: specify type to ICC2
        grouped_by_dim = df_group.groupby("dimension")
        icc_x = pg.intraclass_corr(grouped_by_dim.get_group("x"), targets="index", raters="rater", ratings="value")
        icc_y = pg.intraclass_corr(grouped_by_dim.get_group("y"), targets="index", raters="rater", ratings="value")

        # Concat results
        icc_x = icc_x.set_index(icc_x["Type"] + "_x").drop("Type", axis=1)
        icc_y = icc_y.set_index(icc_y["Type"] + "_y").drop("Type", axis=1)
        icc_all = pd.concat([icc_x, icc_y])

        return icc_all

    res = grouped.apply(icc_per_bodypart)

    print("........ calculated ICC!")
    return res


def rmse(df_g_stacked: pd.DataFrame, df_p_stacked: pd.DataFrame, scaling_factors: pd.Series):
    """Calculates RMSE for every body part, treating x/y as co-dependent values."""

    # Join overlap: "sure" predictions from df_p_stacked, upper limb containing images from df_g_stacked
    df_both = df_p_stacked.merge(df_g_stacked, on=["index", "bodyparts"], suffixes=("_pred", "_truth"))
    # Group by bodypart
    grouped = df_both.groupby("bodyparts")

    def rmse_per_bodypart(df_group):
        """Calculates RMSE for one body part, treating x/y as co-dependent values."""

        # Calculate Euclidean distance for each pair of ground truth/prediction
        df_group["eucl"] = df_group.apply(
            lambda row: euclidean(
                [row["x_pred"], row["y_pred"]], [row["x_truth"], row["y_truth"]]
            )/scaling_factors.loc[row["index"]],
            axis=1
        )
        # Calculate RMS distance
        rmse_val = np.sqrt(np.mean(np.square(df_group["eucl"])))

        return rmse_val

    res = grouped.apply(rmse_per_bodypart)
    res = res.reset_index().rename({0: "rmse"}, axis=1)
    print(res)

    print("........ calculated RMSE!")
    return res


def classif_perf(df_g_stacked: pd.DataFrame, df_p_stacked_lh: pd.DataFrame):
    """Calculates the confusion matrix based on the confidence level cutoff."""

    # Define positive predictions (TP, FP)
    df_p_pos = df_p_stacked_lh[df_p_stacked_lh["likelihood"] > pcutoff]
    mask_p = df_p_stacked_lh.index.isin(df_p_pos.index.to_list())
    df_p_stacked_lh["pos"] = mask_p

    # Define positive ground truth (TP, FN)
    df_g_pos = df_g_stacked[(df_g_stacked["x"] != 0) | (df_g_stacked["y"] != 0)]
    mask_g = df_g_stacked.index.isin(df_g_pos.index.to_list())
    df_g_stacked["pos"] = mask_g

    # Merge data for unique body part / image combination
    df_both = pd.merge(df_p_stacked_lh[["index", "bodyparts", "pos", "likelihood"]],
                       df_g_stacked[["index", "bodyparts", "pos"]],
                       on=["index", "bodyparts"],
                       how="inner", suffixes=("_p", "_g"))
    # Get classification prediction/truth values
    y_true = df_both["pos_g"].apply(lambda x: int(x))
    y_pred = df_both["pos_p"].apply(lambda x: int(x))

    # Performance metrics, TODO: add CIs and adjust rounding
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"A: {round(acc, 4)}, P: {round(prec, 4)}, R: {round(rec, 4)}, F1: {round(f1, 4)}")

    # Confusion matrix
    tp = len(df_both[(df_both["pos_p"]) & (df_both["pos_g"])])
    fp = len(df_both[(df_both["pos_p"]) & (df_both["pos_g"] == False)])
    fn = len(df_both[(df_both["pos_p"] == False) & (df_both["pos_g"])])
    tn = len(df_both[(df_both["pos_p"] == False) & (df_both["pos_g"] == False)])

    cm = [tn, fp, fn, tp]
    return cm, acc, prec, rec, f1


def eval():
    """
        This function assumes predictions generated through DeepLabCut (e.g. evaluate_network).
        Using ground truth and prediction data, we can define our own evaluation.
    """

    # GET PREDICTIONS (hardcode for now)

    if is_coco:
        # # Predictions of baseline model on COCO test set
        # pred_file_name = "DLC_resnet50_mrpbaseJul24shuffle1_1030000-snapshot-1030000_coco-test.h5"
        # # Predictions of baseline model on COCO validation set
        pred_file_name = "DLC_resnet50_mrpbaseJul20shuffle1_1030000-snapshot-1030000_coco-val.h5"
    else:
        # # Predictions of baseline model on MacaquePose
        pred_file_name = "DLC_resnet50_mrpbaseJul22shuffle1_1030000-snapshot-1030000_macaque.h5"
    if not is_baseline:
        assert is_coco
        # # Predictions of TL model on COCO validation set
        pred_file_name = "DLC_resnet50_mrpJun23shuffle1_10000-snapshot-10000_coco-val.h5"

    df_p = pd.read_hdf(f"evaluation/{pred_file_name}")

    # Flatten columns and index (image file paths) from multilevel index into strings
    df_p.columns = df_p.columns.droplevel(level="scorer")
    if isinstance(df_p.index, pd.MultiIndex):
        images = []
        for tup in df_p.index.to_flat_index():
            images.append("/".join(tup))
        df_p.index = images
    # Stack predictions DataFrame into longer format
    df_p_stacked = df_p.stack(level="bodyparts").reset_index(level="bodyparts").reset_index()

    # GET GROUND TRUTH DATA

    if is_coco:
        ground_truth_file = "coco2017_all.h5"
        if is_coco_test:
            ground_truth_file = "coco2017_val-test.h5"
        assert Path(ground_truth_file).is_file()
        df_g = pd.read_hdf(ground_truth_file)

        if not is_coco_test:
            # Filter to validation split
            df_g = df_g[df_g[("istrain", "", "")] == False]
            # Get rid of indication column and adjust column index
            df_g = df_g.drop("istrain", axis=1, level=0)

        df_g.columns = df_g.columns.droplevel(level="scorer")
        # Stack DataFrame into longer format
        df_g_stacked = df_g.stack(level="bodyparts").reset_index(level="bodyparts").reset_index(names=["index"])
    else:
        ground_truth_path = Path("macaque_val.h5")
        assert ground_truth_path.is_file()
        df_g = pd.read_hdf(ground_truth_path)

        # Add full path to image file names
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

    # KEYPOINT CLASSIFICATION EVALUATION

    # Generate the confusion matrix based on the likelihood cutoff
    cm, acc, prec, rec, f1 = classif_perf(df_g_stacked, df_p_stacked)

    # # TODO: (optionally) drop "unsure" predictions according to likelihood cutoff
    # # (= 0.6, from pre-trained config.yaml, or 0.4, from MacaquePose paper, or 0.1, from DeepLabCut paper)
    # df_p_stacked = df_p_stacked[df_p_stacked["likelihood"] >= pcutoff]
    # print(f"Bodyparts that are predicted with confidence, thus considered: {df_p_stacked['bodyparts'].unique()}")

    df_p_stacked = df_p_stacked.drop("likelihood", axis=1)

    # COORDINATE PREDICTION EVALUATION

    # 1) RMSE

    # Load scaling factors
    if is_coco:
        scaling_factors = pd.read_hdf("scaling_coco_val.h5")
        if is_coco_test:
            scaling_factors = pd.read_hdf("scaling_coco_test.h5")
    else:
        scaling_factors = pd.read_hdf("scaling_macaque.h5")
    res_rmse = rmse(df_g_stacked, df_p_stacked, scaling_factors)
    # # Save results for post-processing
    # res_rmse.to_hdf(f"evaluation/rmse_{pred_name}_{dataset_name}.h5", key="df", mode="w")

    # 2) ICC (for x and y coordinates together, for each bodypart): would use R code for this
    # 3) MAP: use DLC's evaluate_assembly() from inferenceutils.py
    # 4) PCK: calculation is just a low-pass filter with arbitrary distance threshold?

    # PLOTTING

    plot_conf_matrix(cm, dataset_name, pred_name)
    plot_single_rmse(res_rmse, dataset_name)

    print("~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d [ %H:%M:%S ]')}")
    print(" All done!")
    print("~~~~~~~~~~~~~~~~~~~~~~~")


# SETTINGS

# Set ground truth
is_coco = False
is_coco_test = False
# Set prediction scorer name
is_baseline = True
# Set confidence level cutoff, TODO: optimise
pcutoff = 0.4

# Get dataset names
if is_coco:
    dataset_name = "COCO"
    if is_coco_test:
        dataset_name = "COCO-TEST"
else:
    dataset_name = "MacaquePose"
if is_baseline:
    pred_name = "base"
else:
    pred_name = "tl"

# RUN EVALUATION

eval()
