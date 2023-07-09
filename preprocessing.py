"""
COCO data preprocessing in order to be used as pre-labeled training data for DeepLabCut models.
Large parts of the code were adapted from: https://github.com/robertklee/COCO-Human-Pose.git
"""
import json
import numpy as np
from pycocotools.coco import COCO
import pandas as pd


def get_meta(coco):
    """Generator to load image and person meta data from JSON annotations."""

    ids = list(coco.imgs.keys())
    for i, img_id in enumerate(ids):
        img_meta = coco.imgs[img_id]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        img_file_name = img_meta["file_name"]
        w = img_meta["width"]
        h = img_meta["height"]
        url = img_meta["coco_url"]

        yield [img_id, img_file_name, w, h, url, anns]


def convert_to_df(coco):
    """Collects images data and person meta data."""

    images_data = []
    persons_data = []

    for img_id, img_fname, w, h, url, meta in get_meta(coco):
        images_data.append({
            "image_id": int(img_id),
            "src_set_image_id": int(img_id), # repeat id to reference after join
            "coco_url": url,
            "path": "labeled-data/dummy-video/" + img_fname,
            "width": int(w),
            "height": int(h)
        })
        for m in meta:
            # Assert a human is depicted
            assert m["category_id"] == 1
            
            persons_data.append({
                "ann_id": m["id"],
                "image_id": m["image_id"],
                "is_crowd": m["iscrowd"],
                "bbox": m["bbox"],
                "bbox_area" : m["bbox"][2] * m["bbox"][3],
                "area": m["area"],
                "num_keypoints": m["num_keypoints"],
                "keypoints": m["keypoints"],
                "segmentation": m["segmentation"]
            })

    # Build Dataframes from lists of dicts

    images_df = pd.DataFrame(images_data)
    images_df.set_index("image_id", inplace=True)

    persons_df = pd.DataFrame(persons_data)
    persons_df.set_index("image_id", inplace=True)

    return images_df, persons_df


def get_df(path_to_anns, dataset_name):
    """Returns the Dataframe of a given annotation dataset."""

    coco_anns = COCO(path_to_anns) # load annotations
    images_df, persons_df = convert_to_df(coco_anns)
    coco_df = pd.merge(images_df, persons_df, right_index=True, left_index=True)
    coco_df["source"] = dataset_name
    coco_df.head()

    return coco_df


def load_coco():
    """Returns the two Dataframes for the 2017 keypoints COCO train set and validation set."""

    df_train = get_df(path_to_anns="annotations_trainval2017/annotations/person_keypoints_train2017.json",
                      dataset_name="train2017")
    df_val = get_df(path_to_anns="annotations_trainval2017/annotations/person_keypoints_val2017.json",
                    dataset_name="val2017")
    return df_train, df_val


def filter_df_coco(df_org):
    """Filters the annotation Dataframe according to set requirements."""

    # Single-human images: for now, we disregard multi-person images instead of cropping them
    dups = df_org["src_set_image_id"].duplicated(keep=False)
    df_filtered = df_org[~dups]

    # At least 5 keypoints
    df_filtered = df_filtered[df_filtered["num_keypoints"] > 4]

    return df_filtered


def rearrange_df(coco_df):
    """Restricts and rearranges the remaining Dataframe into the format required by DeepLabCut."""

    # Extract keypoint names
    keypoint_names = ""
    with open("annotations_trainval2017/annotations/person_keypoints_val2017.json", "r") as f:
        data = json.load(f)
        keypoint_names = data["categories"][0]["keypoints"]

    # Explode keypoints into same row as columns
    coco_df = coco_df.merge(coco_df["keypoints"].apply(pd.Series), left_index=True, right_index=True)

    # Create multi-index
    x_columns = list(zip(["scorer" for _ in range(0, 17)], keypoint_names, ["x" for _ in range(0, 17)]))
    y_columns = list(zip(["scorer" for _ in range(0, 17)], keypoint_names, ["y" for _ in range(0, 17)]))
    all_columns = [single for pair in zip(x_columns, y_columns) for single in pair]
    final_columns = pd.MultiIndex.from_tuples(all_columns, names=["scorer", "bodyparts", "coords"])

    # coco_df = coco_df.drop(["src_set_image_id", "coco_url", "width", "height", "ann_id", "is_crowd", "bbox",
    # "bbox_area", "area", "num_keypoints", "keypoints", "segmentation", "source"], axis=1)

    # Set row index
    coco_df = coco_df.set_index("path")

    # Drop visibility column and other information
    coco_df = coco_df[sorted(list(range(0, 51, 3)) + list(range(1, 51, 3)))]

    # Set multi-index as column names
    coco_df.columns = final_columns

    return coco_df


def prepare_coco(df_train, df_val):
    """Step-by-step preprocessing after data loading."""

    # Filtering
    df_train = filter_df_coco(df_train)
    df_val = filter_df_coco(df_val)

    # (Re-)arranging
    df_train = rearrange_df(df_train)
    df_val = rearrange_df(df_val)

    return df_train, df_val


def save_coco(df_train, df_val, split_val):
    """Saves the annotation DataFrames as separate .h5 files."""

    df_train.to_hdf("coco2017_train.h5", key="df", mode="w")

    if split_val:
        """Request to split validation file, e.g. to hold out one half for downstream evaluation."""

        df_val0 = df_val.sample(frac=0.5)
        df_val1 = df_val.drop(df_val0.index)

        df_val0.to_hdf("coco2017_val.h5", key="df", mode="w")
        df_val1.to_hdf("coco2017_val-test.h5", key="df", mode="w")

        return df_train, df_val0
    else:
        df_val.to_hdf("coco2017_val.h5", key="df", mode="w")

        return df_train, df_val


def merge_coco_sets(df_train, df_val):
    """
        Takes pre-created annotation DataFrames and merges them, adding an attribute to specify if an image belongs to
        the train or validation set, before saving it as a new .h5 file.
    """

    df_train["istrain"] = [True for _ in range(len(df_train))]
    df_val["istrain"] = [False for _ in range(len(df_val))]

    # Merge and sort DataFrame
    df_all = pd.concat([df_train, df_val]).sort_index()

    df_all.to_hdf("coco2017_all.h5", key="df", mode="w")


def filter_df_mac(df):
    """TBD"""

    df["kp"] = df.apply(lambda row: json.loads(row["keypoints"]), axis=1)

    # Single-animal only
    def is_multi_animal(row):
        if len(row["kp"]) > 1:
            return 1
        else:
            return 0

    multi = list(df.apply(is_multi_animal, axis=1))
    mask = [bool(not val) for val in multi]

    df = df[mask]

    # # At least 5 keypoints ?
    # df_filtered = df_filtered[df_filtered["num_keypoints"] > 4]

    return df


def prepare_macaquepose_valset():
    # TODO: Implement routine to get validation subset from MacaquePose annotations

    # Load MacaquePose annotations and fix naming error
    df_all = pd.read_csv("annotations.csv")\
        .rename({"keypoinbts": "keypoints"}, axis=1)\
        .drop(columns="segmentation")

    # Apply filter
    df_single = filter_df_mac(df_all)

    # Stretch to get keypoints
    df = df_single.explode(column="kp")\
        .explode(column="kp")\
        .reset_index()  # double explode because of nested list
    df = df.join(pd.json_normalize(df.pop("kp")))
    df["bodyparts"] = df.apply(lambda row: row["name"].replace(" ", "_"), axis=1)
    df = df.drop(["name"], axis=1)

    # Coordinates into tuples
    df["position"] = df.apply(lambda row: tuple(row["position"]) if row["position"] is not None else (0, 0), axis=1)

    # TODO: How to deal with non-present keypoints? (Right now make 0, but how evaluation?)

    return df
    print("TBD")


def save_val_macaque(df):
    """Save subset of MacaquePose to serve as validation set."""

    # Create subset
    np.random.seed(42)
    subset = [np.random.randint(0, max(df["index"]) + 1) for _ in range(0, 779)]
    df_val = df[df["index"].isin(subset)]

    # Save to .h5 file
    df_val.to_hdf("macaque_val.h5", key="df", mode="w")


if __name__ == "__main__":
    dataset = "macaque"

    if dataset == "coco":
        df_train, df_val = load_coco()
        df_train, df_val = prepare_coco(df_train, df_val)
        df_train, df_val = save_coco(df_train, df_val, True)
        merge_coco_sets(df_train, df_val)
    elif dataset == "macaque":
        df = prepare_macaquepose_valset()
        save_val_macaque(df)

    print("Preprocessing complete.")