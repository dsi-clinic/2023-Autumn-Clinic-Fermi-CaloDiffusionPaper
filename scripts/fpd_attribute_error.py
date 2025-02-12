import argparse
import torch
import h5py
import numpy as np
import jetnet
import json
from autoencoder.CaloEnco import CaloEnco
from utils import NNConverter, LoadJson, DataLoader, EarlyStopper, nn


import torch.utils.data as torchdata
import os
from typing import Tuple, Optional, Dict


# Setup args
def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FPD Metric Evaluation")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/home/kwallace2/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/scripts/ae_models/dataset1_phot_AE_64_64_64_64_0.0004",
        help="Path to the model directory",
    )
    # parser.add_argument(
    #     "--checkpoint_path",
    #     type=str,
    #     default="/home/kwallace2/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/scripts/ae_models/dataset1_phot_AE_64_64_64_64_0.0004/best_val.pth",
    #     help="Path to the checkpoint file",
    # )
    # parser.add_argument(
    #     "--config_path",
    #     type=str,
    #     default="/home/kwallace2/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/scripts/ae_models/dataset1_phot_AE_64_64_64_64_0.0004/config_dataset1_photon.json",
    #     help="Path to the config file",
    # )
    parser.add_argument(
        "--reference_hdf5",
        type=str,
        default="/net/projects/fermi-1/data/dataset_1/dataset_1_photons_1.hdf5",
        help="Path to the reference HDF5 file",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/net/projects/fermi-1/data/dataset_1",
        help="Directory where the dataset is stored.",
    )
    return parser.parse_args()


def setup_dataset(config: Dict, 
    data_folder: str, 
    take_subset: bool = False,
    val_frac: float=0.2, # TODO: Figure out correct value for default val_frac
) -> torchdata.DataLoader:
    """
    Given a data folder and config dictionary, this function loads the validation
    dataset and returns a torchdata.DataLoader object.
    """
    orig_shape = "orig" in config.get("SHOWER_EMBED", "")
    shape_pad = config["SHAPE_PAD"]
    data, energies = [], []
    for i, dataset in enumerate(config["FILES"]):
        data_, e_ = DataLoader(
            os.path.join(data_folder, dataset),
            shape_pad,
            emax=config["EMAX"],
            emin=config["EMIN"],
            nevts=args.nevts,
            max_deposit=config[
                "MAXDEP"
            ],  # Noise can generate more deposited energy than generated
            logE=config["logE"],
            showerMap=config["SHOWERMAP"],
            nholdout=(
                config.get("HOLDOUT", 0)
                if (i == len(config["FILES"]) - 1)
                else 0
            ),
            dataset_num=config.get("DATASET_NUM", 2),
            orig_shape=orig_shape,
        )

        data = data_ if i == 0 else np.concatenate((data, data_))
        energies = e_ if i == 0 else np.concatenate((energies, e_))

    # Reshape data based on the SHOWER_EMBED config value
    energies = np.reshape(energies, (-1))
    if not orig_shape:
        data = np.reshape(data, shape_pad)
    else:
        data = np.reshape(data, (len(data), -1))

    print("DATA SHAPE BEFORE", data.shape, energies.shape)
    if take_subset:
        data, energies = data[:500, :], energies[:500]
        print("DATA SHAPE AFTER", data.shape, energies.shape)

    torch_data_tensor = torch.from_numpy(data)
    torch_E_tensor = torch.from_numpy(energies)

    torch_dataset = torchdata.TensorDataset(torch_E_tensor, torch_data_tensor)

    # Split into training and validation sets
    num_data = data.shape[0]
    nTrain = int(round(self.args.frac * num_data))
    nVal = num_data - nTrain
    train_dataset, val_dataset = torch.utils.data.random_split(
        torch_dataset, [nTrain, nVal]
    )

    loader_train = torchdata.DataLoader(
        train_dataset, batch_size=config["BATCH"], shuffle=True
    )
    loader_val = torchdata.DataLoader(
        val_dataset, batch_size=config["BATCH"], shuffle=True
    )

    del data, torch_data_tensor, torch_E_tensor, train_dataset, val_dataset

    return loader_val


def main(args: argparse.Namespace) -> None:
    # Set up paths relative to model_dir
    model_dir = args.model_dir
    checkpoint_path = f"{model_dir}/best_val.pth"
    config_path = f"{model_dir}/config_dataset1_photon.json"

    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    shape = (368,)  # match # of features per shower
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CaloEnco(
        shape,
        config=config,
        training_obj="mean_pred",
        nsteps=config.get("NSTEPS", 1000),
        layer_sizes=config.get("LAYER_SIZES", [64, 64, 64, 64]),
    ).to(device)

    # Load trained model weights
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    filtered_checkpoint = {k: v for k, v in checkpoint.items() if "NN_embed" not in k}

    print("Autoencoder model loaded successfully.")
    model.eval()

    # Load the Validation Dataset
    dataset = setup_dataset(config=config, data_folder=args.dataset_dir)

    # Generate Autoencoder Outputs
    model_preds = None # TODO: Init a torch tensor with the correct shape
    with torch.no_grad():
        for E, data in loader:
            # TODO: what is data? what is E? Which one do we want the model to reconstruct?
            data = data.to(device=self.device)
            E = E.to(device=self.device)

            # Evaluate the autoencoder on this batch 
            # TODO: Is this the correct way to call model.pred() ?
            model_out = model.pred(data, E)


            # TODO: write some code to concatenate the model_out's into a single 
            # pytorch tensor called model_preds

    # TODO: Is it OK to call this once per batch, or does it need to 
    # be called on all of the dataset at once?
    fpd_val, fpd_err = jetnet.evaluation.fpd(
        real_features=None, # TODO: Is this the correct thing to put here?
        gen_features=model_out
    )

    # TDOD: write some code
    results_path = f"{model_dir}/fpd_results.txt"
    with open(results_path, "w") as f:
        f.write(f"FPD Metric: {fpd_val:.6f} ± {fpd_err:.6f}\n")

    print(f"FPD evaluation completed. Results saved to {results_path}.")


"""
This code raises: 
Traceback (most recent call last):
  File "/home/kwallace2/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/CaloChallenge/code/fpd_attribute_error.py", line 59, in <module>
    reconstructed_showers = model(input_tensors)
  File "/home/kwallace2/miniconda3/envs/fermi/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/kwallace2/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/scripts/autoencoder/CaloEnco.py", line 391, in forward
    x = self.convTrans(x)  # Process normally
  File "/home/kwallace2/miniconda3/envs/fermi/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/kwallace2/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/scripts/autoencoder/ae_models.py", line 100, in forward
    x = F.pad(x, pad=(0, 0, circ_pad, circ_pad, 0, 0), mode="circular")
RuntimeError: Padding length too large.
"""


if __name__ == "__main__":
    args = setup_args()
    main(args)
