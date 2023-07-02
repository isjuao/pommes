#!/usr/bin/env python
# coding: utf-8

"""
    Run from project directory (one above the DLC project).
"""


import deeplabcut as dlc

import tensorflow as tf
import datetime
import argparse
import pandas as pd
from pathlib import Path
import pingouin as pg
from sklearn.metrics import mean_squared_error


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--project", help="Specify name of DLC project", required=True)
args = parser.parse_args()

print("~~~~~~~~~~~~~~~~~~~~~~~")
print(f"{datetime.datetime.now().strftime('%Y-%m-%d [ %H:%M:%S ]')}")
print(" Evaluating model ...")
print("~~~~~~~~~~~~~~~~~~~~~~~")

# dlc.evaluate_network(
#     args.project + "/config.yaml",
#     # comparisonbodyparts="all",
#     # rescale=False,
#     gputouse=0,
#     # trainingsetindex=0,
# )

# def pred():
#     print("TBD")
#     # TODO: set up "own" prediction
#     # model_folder = "/path/to/model/folder"
#     # net_type = "resnet_50"
#     # sess, inputs, outputs = dlc.pose_estimation_tensorflow.nnets.net.load_model(model_folder, net_type)
#     # tf.compat.v1.reset_default_graph?
#
#     sess = tf.compat.v1.Session()
#     saver = tf.compat.v1.train.import_meta_graph("TBD.meta")
#     saver.restore(sess, tf.train.latest_checkpoint('./'))
#
#     # DLC: MAE; which is proportional to the average root mean square error?
#     # PCK calculation is just a low-pass filter with arbitrary distance threshold


def get_ground_truth(path):
    """Loads ground truth data from .h5 file."""

    assert path.is_file()

    df_g = pd.read_hdf(path)
    df_ground = df_g.drop("istrain", axis=1, level=0)

    return df_ground


def stack_and_filter_dfs(df_ground: pd.DataFrame, df_pred: pd.DataFrame):
    """Stacks DataFrames into longer format and filters predictions for keypoint according to likelihood cutoff."""

    pcutoff = 0.6   # TODO: likelihood cutoff from config.yaml

    # Stack DataFrames
    df_p_stacked = df_pred.stack(level="bodyparts").reset_index(level="bodyparts").reset_index()
    df_g_stacked = df_ground.stack(level="bodyparts").reset_index(level="bodyparts").reset_index(names=["index"])

    # Drop "unsure" predictions
    # TODO: do predictions that are sure but the keypoint does not exist in truth stay?
    df_p_stacked = df_p_stacked[df_p_stacked["likelihood"] >= pcutoff].drop("likelihood", axis=1)
    print(f"Bodyparts that are predicted with confidence and thus considered for ICC: "
          f"{df_p_stacked['bodyparts'].unique()}")

    return df_g_stacked, df_p_stacked


def rmse(df_ground: pd.DataFrame, df_pred: pd.DataFrame):
    """Calculates RMSE for every body part, treating x/y separately."""

    df_g_stacked, df_p_stacked = stack_and_filter_dfs(df_ground, df_pred)

    # Join overlap ("sure" predictions)
    df_both = df_p_stacked.merge(df_g_stacked, on=["index", "bodyparts"], suffixes=("_pred", "_truth"))

    # Group by bodypart
    grouped = df_both.groupby("bodyparts")

    def rmse_per_bodypart(df_group):
        rmse_x = mean_squared_error(y_true=df_group["x_truth"], y_pred=df_group["x_pred"], squared=False)
        rmse_y = mean_squared_error(y_true=df_group["y_truth"], y_pred=df_group["y_pred"], squared=False)

        return pd.Series(data=[rmse_x, rmse_y], index=["RMSE_x", "RMSE_y"])

    res = grouped.apply(rmse_per_bodypart)

    print("........ calculated RMSE!")
    return res


def icc(df_ground: pd.DataFrame, df_pred: pd.DataFrame):
    """Calculates ICC for every body part, treating x/y separately."""

    df_g_stacked, df_p_stacked = stack_and_filter_dfs(df_ground, df_pred)

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
        icc_y = pg.intraclass_corr(grouped_by_dim.get_group("y"), targets="index", raters="rater", ratings="value")

        # Concat results
        icc_x = icc_x.set_index(icc_x["Type"] + "_x").drop("Type", axis=1)
        icc_y = icc_y.set_index(icc_y["Type"] + "_y").drop("Type", axis=1)
        icc_all = pd.concat([icc_x, icc_y])

        return icc_all

    res = grouped.apply(icc_per_bodypart)

    print("........ calculated ICC!")
    return res


def eval(ground_truth_path):
    """
        This function assumes we have the predictions. #TODO: either using dlc or tf
        Using ground truth and prediction data, we can define our own evaluation.
    """

    # Get ground truth data
    df_g = get_ground_truth(ground_truth_path)
    df_g.columns =df_g.columns.droplevel(level="scorer")

    # Get predictions
    df_p = pd.read_hdf("evaluation/DLC_resnet50_mrpJun23shuffle1_10000-snapshot-10000.h5")  # TODO: hardcode for now
        # parse_ground_truth_data_file?
    df_p.columns = df_p.columns.droplevel(level="scorer")
    # Flatten index (image file paths)
    images = []
    for tup in df_p.index.to_flat_index():
        images.append("/".join(tup))
    df_p.index = images

    # RMSE, for x and y coordinates separately, for each bodypart
    # (DLC says MAE, but can probably infer RMSE for every bodypart using function in evaluate.py)
    # TODO: waiting for reply by MacaquePose group
    res_rmse = rmse(df_g, df_p)


    # ICC, for x and y coordinates separately, for each bodypart
    res_icc = icc(df_g, df_p)

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


anno_file = Path("coco2017_all.h5")
eval(anno_file)

print("~~~~~~~~~~~~~~~~~~~~~~~")
print(f"{datetime.datetime.now().strftime('%Y-%m-%d [ %H:%M:%S ]')}")
print(" All done!")
print("~~~~~~~~~~~~~~~~~~~~~~~")