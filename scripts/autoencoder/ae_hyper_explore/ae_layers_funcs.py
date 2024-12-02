"""Function .py file for ae_layers Notebook"""

import pandas as pd
import matplotlib.pyplot as plt


def loss_df(model):
    """
    Creates the loss df from the training / validation losses.txt file

    Parameters:
    - model (str): model folder name to obtain loss files

    Returns:
    - training_loss df, validation_loss df of format: epoch loss
    """
    df = pd.read_csv(
        f"../../../ae_models/{model}/training_losses.txt", header=None, names=["loss"]
    )
    df["epoch"] = df.index

    df2 = pd.read_csv(
        f"../../../ae_models/{model}/validation_losses.txt", header=None, names=["loss"]
    )
    df2["epoch"] = df2.index
    return df[["epoch", "loss"]], df2[["epoch", "loss"]]


def plot_LR(e3, e4, e5, ver, title):
    """
    Plots the losses for differing learning rates

    Parameters:
    - eN (df): df input for loss with LR 4e-N
    - ver (str): version of loss (validation or training) for axes labels
    - title (str): title for plot
    """
    plt.plot(e3["epoch"], e3["loss"], label="4e-3")
    plt.plot(e4["epoch"], e4["loss"], label="4e-4")
    plt.plot(e5["epoch"], e5["loss"], label="4e-5")
    plt.xlabel("Epoch")
    plt.ylabel(f"{ver} Loss")
    plt.title(title)
    plt.yscale("log")
    plt.legend()
    plt.show()


def plot_loss(m1, names, compress, ver, dataset, compare=None):
    """
    Plots the losses for differing layer sizes

    Parameters:
    - m1 (df): base df for losses to plot
    - names (lst[str]): list of strings for legend
    - compress (lst[str]): list of compression factors for legend
    - dataset (str): dataset name for plot details
    - compare (lst[df]): list of additional dfs to plot for comparison
    """
    plt.plot(m1["epoch"], m1["loss"], label=f"{compress[0]}: {names[0]}")
    if compare is not None:
        for i, m in enumerate(compare):
            plt.plot(m["epoch"], m["loss"], label=f"{compress[i + 1]}: {names[i + 1]}")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel(f"{ver} Loss")
    plt.title(f"{dataset} AE Layer {ver} Loss Comparison")
    plt.yscale("log")
    plt.show()
