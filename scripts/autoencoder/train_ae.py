import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse

# import h5py as h5
import torch
import torch.optim as optim
import torch.utils.data as torchdata
import sys
import os

# from ae_models import *
from CaloEnco import CaloEnco
import tqdm
from scripts.utils import NNConverter, LoadJson, DataLoader, EarlyStopper, nn


def trim_file_path(cwd: str, num_back: int):
    """ """
    split_path = cwd.split("/")
    trimmed_split_path = split_path[:-num_back]
    trimmed_path = "/".join(trimmed_split_path)

    return trimmed_path


if __name__ == "__main__":
    print("TRAIN AUTOENCODER")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_folder",
        default="/wclustre/cms_mlsim/denoise/CaloChallenge/",
        help="Folder containing data and MC files",
    )
    parser.add_argument("--model", default="AE", help="AE model to train")
    parser.add_argument(
        "-c",
        "--config",
        default="configs/test.json",
        help="Config file with training parameters",
    )
    parser.add_argument(
        "--nevts", type=int, default=-1, help="Number of events to load"
    )
    parser.add_argument(
        "--frac",
        type=float,
        default=0.85,
        help="Fraction of total events used for training",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        default=False,
        help="Load pretrained weights to continue the training",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Pytorch seed")
    parser.add_argument(
        "--reset_training", action="store_true", default=False, help="Retrain"
    )
    parser.add_argument("--binning_file", type=str, default=None)
    parser.add_argument(
        "--patience", type=int, default=25, help="Patience for early stopper"
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=1e-5,
        help="Minimum loss change range for early stopper",
    )
    parser.add_argument(
        "--save_folder_append",
        type=str,
        default=None,
        help="Optional text to append to training folder to separate outputs of training runs with the same config file",
    )
    parser.add_argument(
        "--save_folder_absolute",
        type=str,
        default=None,
        help="Optional path to use for training folder instead of default ..",
    )
    parser.add_argument("--resnet_set", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--layer_sizes",
        type=int,
        nargs="+",
        default=None,
        help="Manual layer sizes input instead of from config file",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        nargs="+",
        default=None,
        help="Manual learning rate input instead of from config file",
    )
    parser.add_argument(
        "--no_early_stop",
        action="store_true",
        help="Turns off early stop functionality and defailts to max epochs",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="Manually assign a maximum number of epochs",
    )
    parser.add_argument(
        "--vae_zdim",
        type=int,
        default=32,
        help="Latent dimension for VAE if model is 'VAE'",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Weight for KL divergence term in VAE loss",
    )
    parser.add_argument(
        "--kl_warmup_steps",
        type=int,
        default=1000,
        help="Number of steps for KL warmup",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="mse",
        choices=["mse", "l1", "huber"],
        help="Type of reconstruction loss to use",
    )
    flags = parser.parse_args()

    cwd = __file__
    calo_challenge_dir = trim_file_path(cwd=cwd, num_back=3)
    sys.path.append(calo_challenge_dir)
    print(calo_challenge_dir)
    from scripts.utils import *
    from CaloChallenge.code.XMLHandler import *

    dataset_config = LoadJson(flags.config)

    print("TRAINING OPTIONS")
    print(dataset_config, flush=True)

    torch.manual_seed(flags.seed)

    nholdout = dataset_config.get("HOLDOUT", 0)

    batch_size = dataset_config["BATCH"]

    if flags.no_early_stop:
        if flags.max_epochs is None:
            num_epochs = dataset_config["MAXEPOCH"]
        else:
            num_epochs = flags.max_epochs
    else:
        num_epochs = dataset_config["MAXEPOCH"]

    early_stop = dataset_config["EARLYSTOP"]
    training_obj = "mean_pred"
    loss_type = dataset_config.get("LOSS_TYPE", "l2")
    dataset_num = dataset_config.get("DATASET_NUM", 2)
    shower_embed = dataset_config.get("SHOWER_EMBED", "")
    orig_shape = "orig" in shower_embed
    energy_loss_scale = dataset_config.get("ENERGY_LOSS_SCALE", 0.0)

    data = []
    energies = []

    for i, dataset in enumerate(dataset_config["FILES"]):
        data_, e_ = DataLoader(
            os.path.join(flags.data_folder, dataset),
            dataset_config["SHAPE_PAD"],
            emax=dataset_config["EMAX"],
            emin=dataset_config["EMIN"],
            nevts=flags.nevts,
            max_deposit=dataset_config[
                "MAXDEP"
            ],  # Noise can generate more deposited energy than generated
            logE=dataset_config["logE"],
            showerMap=dataset_config["SHOWERMAP"],
            nholdout=nholdout if (i == len(dataset_config["FILES"]) - 1) else 0,
            dataset_num=dataset_num,
            orig_shape=orig_shape,
        )

        if i == 0:
            data = data_
            energies = e_
        else:
            data = np.concatenate((data, data_))
            energies = np.concatenate((energies, e_))

    NN_embed = None
    if "NN" in shower_embed:
        if dataset_num == 1:
            if flags.binning_file is None:
                flags.binning_file = (
                    "../CaloChallenge/code/binning_dataset_1_photons.xml"
                )
            bins = XMLHandler("photon", flags.binning_file)
        else:
            if flags.binning_file is None:
                flags.binning_file = (
                    "../CaloChallenge/code/binning_dataset_1_pions.xml"
                )
            bins = XMLHandler("pion", flags.binning_file)

        NN_embed = NNConverter(bins=bins).to(device=device)

    dshape = dataset_config["SHAPE_PAD"]
    energies = np.reshape(energies, (-1))
    if not orig_shape:
        data = np.reshape(data, dshape)
    else:
        data = np.reshape(data, (len(data), -1))

    num_data = data.shape[0]
    print("Data Shape " + str(data.shape))
    data_size = data.shape[0]
    torch_data_tensor = torch.from_numpy(data)
    torch_E_tensor = torch.from_numpy(energies)
    del data

    torch_dataset = torchdata.TensorDataset(torch_E_tensor, torch_data_tensor)
    nTrain = int(round(flags.frac * num_data))
    nVal = num_data - nTrain
    train_dataset, val_dataset = torch.utils.data.random_split(
        torch_dataset, [nTrain, nVal]
    )

    loader_train = torchdata.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    loader_val = torchdata.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    del torch_data_tensor, torch_E_tensor, train_dataset, val_dataset
    if flags.learning_rate is None:
        learning_rate = float(dataset_config["LR"])
    else:
        learning_rate = float(flags.learning_rate[0])
    if flags.layer_sizes is None:
        checkpoint_folder = "../ae_models/{}_{}_default_{}".format(
            dataset_config["CHECKPOINT_NAME"], flags.model, learning_rate
        )
    else:
        layer_sizes_str = "_".join(map(str, flags.layer_sizes))
        checkpoint_folder = "../ae_models/{}_{}_{}_{}".format(
            dataset_config["CHECKPOINT_NAME"], flags.model, layer_sizes_str, learning_rate
        )
    if (
        flags.save_folder_absolute is not None
    ):  # Optionally replace this folder with whatever
        checkpoint_folder = (
            f"{flags.save_folder_absolute}{checkpoint_folder[2:]}"
        )
    if (
        flags.save_folder_append is not None
    ):  # Optionally append another folder title
        checkpoint_folder = f"{checkpoint_folder}{flags.save_folder_append}/"

    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
        print(f"Checkpoint folder created at: {checkpoint_folder}")

    checkpoint = dict()
    checkpoint_path = os.path.join(checkpoint_folder, "checkpoint.pth")

    if flags.load and os.path.exists(checkpoint_path):
        print(
            "Loading training checkpoint from %s" % checkpoint_path, flush=True
        )
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(checkpoint.keys())

    if flags.model == "AE":
        shape = (
            dataset_config["SHAPE_PAD"][1:]
            if (not orig_shape)
            else dataset_config["SHAPE_ORIG"][1:]
        )
        model = CaloEnco(
            shape,
            config=dataset_config,
            training_obj=training_obj,
            NN_embed=NN_embed,
            nsteps=dataset_config["NSTEPS"],
            cold_diffu=False,
            avg_showers=None,
            std_showers=None,
            E_bins=None,
            resnet_set=flags.resnet_set,
            layer_sizes=flags.layer_sizes,
        ).to(device=device)

        # Sometimes save only weights, sometimes save other info
        if "model_state_dict" in checkpoint.keys():
            model.load_state_dict(checkpoint["model_state_dict"])
        elif len(checkpoint.keys()) > 1:
            model.load_state_dict(checkpoint)
    elif flags.model == "VAE":
        shape = (
            dataset_config["SHAPE_PAD"][1:]
            if (not orig_shape)
            else dataset_config["SHAPE_ORIG"][1:]
        )
        print("Shape being passed to CaloVAE:", shape)
        from ae_models import CaloVAE
        model = CaloVAE(
            shape=shape,
            config=dataset_config,
            training_obj=training_obj,
            NN_embed=NN_embed,
            nsteps=dataset_config["NSTEPS"],
            zdim=flags.vae_zdim,
            layer_sizes=flags.layer_sizes,
            beta=flags.beta,
            kl_warmup_steps=flags.kl_warmup_steps,
        ).to(device=device)
    else:
        print("Model %s not supported!" % flags.model)
        exit(1)

    os.system(
        "cp autoencoder/ae_models.py {}".format(checkpoint_folder)
    )  # bkp of model def
    os.system(
        "cp {} {}".format(flags.config, checkpoint_folder)
    )  # bkp of config file

    early_stopper = EarlyStopper(
        patience=flags.patience,
        mode="min",
        min_delta=flags.min_delta
    )
    if "early_stop_dict" in checkpoint.keys() and not flags.reset_training:
        early_stopper.__dict__ = checkpoint["early_stop_dict"]
    print(early_stopper.__dict__)

    criterion = nn.MSELoss().to(device=device)

    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, factor=0.1, patience=15, verbose=True
    )
    if "optimizer_state_dict" in checkpoint.keys() and not flags.reset_training:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint.keys() and not flags.reset_training:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    training_losses = np.array([])
    val_losses = np.array([])
    start_epoch = 0
    min_validation_loss = 99999.0
    if "train_loss_hist" in checkpoint.keys() and not flags.reset_training:
        train_hist = checkpoint["train_loss_hist"]
        training_losses = checkpoint["train_loss_hist"]
        val_losses = checkpoint["val_loss_hist"]
        start_epoch = checkpoint["epoch"] + 1

    for epoch in range(start_epoch, num_epochs):
        print("Beginning epoch %i" % epoch, flush=True)
        for i, param in enumerate(model.parameters()):
            break
        train_loss = 0

        model.train()
        for i, (E, data) in tqdm(
            enumerate(loader_train, 0), unit="batch", total=len(loader_train)
        ):
            model.zero_grad()
            optimizer.zero_grad()

            data = data.to(device=device)
            E = E.to(device=device)

            t = torch.randint(
                0, model.nsteps, (data.size()[0],), device=device
            ).long()

            batch_loss = model.compute_loss(
                data,
                E,
                t=t,
                loss_type=flags.loss_type,
                energy_loss_scale=energy_loss_scale,
            )
            batch_loss.backward()

            optimizer.step()
            train_loss += batch_loss.item()

            del data, E, batch_loss

        train_loss = train_loss / len(loader_train)
        training_losses = np.append(training_losses, train_loss)
        print("loss: " + str(train_loss))

        if loader_val is not None:
            val_loss = 0
            model.eval()
            for i, (vE, vdata) in tqdm(
                enumerate(loader_val, 0), unit="batch", total=len(loader_val)
            ):
                vdata = vdata.to(device=device)
                vE = vE.to(device=device)

                t = torch.randint(
                    0, model.nsteps, (vdata.size()[0],), device=device
                ).long()

                batch_loss = model.compute_loss(
                    vdata,
                    vE,
                    t=t,
                    loss_type=flags.loss_type,
                    energy_loss_scale=energy_loss_scale,
                )

                val_loss += batch_loss.item()
                del vdata, vE, batch_loss

            val_loss = val_loss / len(loader_val)
            val_losses = np.append(val_losses, val_loss)
            print("val_loss: " + str(val_loss), flush=True)

            scheduler.step(torch.tensor([train_loss]))

            if val_loss < min_validation_loss:
                torch.save(
                    model.state_dict(),
                    os.path.join(checkpoint_folder, "best_val.pth"),
                )
                min_validation_loss = val_loss

            if (
                not flags.no_early_stop
            ):  # Only use early stopper if it has not been turned off by the flag
                if early_stopper.early_stop(val_loss):
                    print("Early stopping!")
                    break

            model.eval()
            print("SAVING")

            # Save full training state so can be resumed
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss_hist": training_losses,
                    "val_loss_hist": val_losses,
                    "early_stop_dict": early_stopper.__dict__,
                },
                checkpoint_path,
            )

            with open(checkpoint_folder + "/training_losses.txt", "w") as tfileout:
                tfileout.write(
                    "\n".join("{}".format(tl) for tl in training_losses) + "\n"
                )
            with open(
                checkpoint_folder + "/validation_losses.txt", "w"
            ) as vfileout:
                vfileout.write(
                    "\n".join("{}".format(vl) for vl in val_losses) + "\n"
                )
        else:
            val_loss = train_loss
            print("(No validation set)")

    print("Saving to %s" % checkpoint_folder, flush=True)
    torch.save(model.state_dict(), os.path.join(checkpoint_folder, "final.pth"))

    with open(checkpoint_folder + "/training_losses.txt", "w") as tfileout:
        tfileout.write(
            "\n".join("{}".format(tl) for tl in training_losses) + "\n"
        )
    with open(checkpoint_folder + "/validation_losses.txt", "w") as vfileout:
        vfileout.write("\n".join("{}".format(vl) for vl in val_losses) + "\n")
