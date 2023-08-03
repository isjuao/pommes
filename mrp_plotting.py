import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def plot_conf_matrix(cm: list, dataset_name: str, pred_name:str):
    """Plots the normalised confusion matrix, assuming an input order of ["TN", "FP", "FN", "TP"]."""

    # Calculate normalised values
    pred0 = [cm[0]/sum(cm), cm[2]/sum(cm)]
    pred1 = [cm[1]/sum(cm), cm[3]/sum(cm)]

    data = pd.DataFrame({0: pred0, 1: pred1}, index=[0, 1])

    fig, ax1 = plt.subplots()
    p = sns.heatmap(
        data,
        annot=True,
        linewidth=0.5,
        annot_kws={"size": 18},
        vmin=0,
        vmax=1,
        cmap="viridis_r",
        ax=ax1,
    )

    ax1.set_xlabel(f"Prediction ({pred_name})", fontsize=18)
    ax1.set_xticklabels([0, 1], fontsize=16)
    ax1.set_ylabel("Ground Truth", fontsize=18)
    ax1.set_yticklabels([0, 1], fontsize=16)
    cbar = p.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    ax1.set_title(f"Normalised CM ({dataset_name})", fontsize=18)

    out_path = "evaluation/plots/cm_norm_macaque-base_c.pdf"

    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.show()
    # plt.savefig(out_path, dpi=200, bbox_inches="tight")


def order_bodyparts(res_rmse):
    """Bring bodyparts into preferred order."""

    bps = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
           "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee",
           "left_ankle", "right_ankle"]
    res_rmse["bodyparts"] = pd.Categorical(res_rmse["bodyparts"], categories=bps, ordered=True)
    res_rmse = res_rmse.sort_values(by=["bodyparts"], axis=0)

    return res_rmse

def plot_single_rmse(res_rmse: pd.DataFrame, dataset_name: str):
    """Plots a barchart for the RMSE values of different bodyparts."""

    sns.set_theme(style="whitegrid")

    res_rmse = order_bodyparts(res_rmse)

    fig, ax1 = plt.subplots()
    p = sns.barplot(
        data=res_rmse,
        x="bodyparts",
        y="rmse",
        ax=ax1,
        facecolor=(0.2, 0.4, 0.6, 0.8),
    )

    # Adjust labels
    for item in p.get_xticklabels():
        item.set_rotation(45)
        item.set_fontsize(8)
    ax1.set_ylabel("RMSE Per Bodypart")
    ax1.set_xlabel("Bodypart")
    ax1.set_title(f"Scaled RMSE Per Bodypart ({dataset_name})")

    plt.tight_layout()
    plt.show()


def plot_double_rmse():
    """Plots a barchart for the RMSE values of different bodyparts for two sets of predictions."""

    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots()

    # Hardcode filenames for now

    filename1 = "evaluation/rmse_base_MacaquePose.h5"
    color1 = (0.2, 0.4, 0.6, 0.8)   # blue
    # filename1 = "evaluation/rmse_tl_COCO.h5"
    # color1 = (0.2, 0.6, 0.4, 0.8)   # green
    filename2 = "evaluation/rmse_base_COCO.h5"
    color2 = (1.0, 0.6, 0.4, 0.8)   # red

    rmse1 = order_bodyparts(pd.read_hdf(filename1))
    rmse1["category"] = f"{filename1.split('.')[0].split('/')[1]}"
    rmse2 = order_bodyparts(pd.read_hdf(filename2))
    rmse2["category"] = f"{filename2.split('.')[0].split('/')[1]}"

    rmse = pd.DataFrame(pd.concat([rmse1, rmse2], axis=0)).rename(columns={0: "rmse"})

    mean1 = rmse1["rmse"].mean()
    ax1.axhline(y=mean1, color=color1, linestyle="dashed", linewidth=1)
    mean2 = rmse2["rmse"].mean()
    ax1.axhline(y=mean2, color=color2, linestyle="dashed", linewidth=1)

    p = sns.barplot(
        data=rmse,
        x="bodyparts",
        y="rmse",
        hue="category",
        palette=[color1, color2],
        ax=ax1,
    )

    # Adjust labels
    for item in p.get_xticklabels():
        item.set_rotation(45)
        item.set_fontsize(11)
    ax1.set_ylabel("RMSE Per Bodypart")
    ax1.set_xlabel("Bodypart")
    ax1.set_title(f"Scaled RMSE Per Bodypart")
    plt.legend(fontsize=11)

    plt.gca().set_aspect("equal")
    plt.tight_layout()
    out_path = "evaluation/plots/rmse_barplot_baselinecomp.pdf"

    plt.show()
    # plt.savefig(out_path, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    # Running functions independently

    plot_double_rmse()

