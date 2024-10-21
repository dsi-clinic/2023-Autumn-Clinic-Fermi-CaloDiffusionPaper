import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import h5py as h5
import torch.optim as optim
import torch.utils.data as torchdata
from tqdm import tqdm

from CaloDiffu import CaloDiffu

# from models import

import utils
import sys
import torch


from autoencoder.CaloEnco import CaloEnco

if __name__ == "__main__":
    print("TRAIN DIFFU")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Device: cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: mps")
    else:
        device = torch.device("cpu")
        print("Device: cpu")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_folder",
        default="/wclustre/cms_mlsim/denoise/CaloChallenge/",
        help="Folder containing data and MC files",
    )
    parser.add_argument(
        "--model", default="Diffu", help="Diffusion model to train"
    )
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
    parser.add_argument(
        "--layer_sizes",
        type=int,
        nargs="+",
        default=None,
        help="Manual layer sizes input instead of from config file",
    )
    parser.add_argument("--binning_file", type=str, default=None)
    parser.add_argument(
        "--save_folder_append",
        type=str,
        default=None,
        help="Optional text to append to training folder to separate outputs of training runs with the same config file",
    )
    parser.add_argument("--model_loc", default="test", help="Location of model")

    flags = parser.parse_args()

    dataset_config = utils.LoadJson(flags.config)

    def trim_file_path(cwd: str, num_back: int):
        """ """
        split_path = cwd.split("/")
        trimmed_split_path = split_path[:-num_back]
        trimmed_path = "/".join(trimmed_split_path)

        return trimmed_path

    cwd = __file__
    calo_challenge_dir = trim_file_path(cwd=cwd, num_back=3)
    sys.path.append(calo_challenge_dir)
    from scripts.utils import NNConverter, EarlyStopper, nn
    from CaloChallenge.code import XMLHandler

    print("TRAINING OPTIONS")
    print(dataset_config, flush=True)

    torch.manual_seed(flags.seed)

    cold_diffu = dataset_config.get("COLD_DIFFU", False)
    cold_noise_scale = dataset_config.get("COLD_NOISE", 1.0)

    nholdout = dataset_config.get("HOLDOUT", 0)

    batch_size = dataset_config["BATCH"]
    num_epochs = dataset_config["MAXEPOCH"]
    early_stop = dataset_config["EARLYSTOP"]
    training_obj = dataset_config.get("TRAINING_OBJ", "noise_pred")
    loss_type = dataset_config.get("LOSS_TYPE", "l2")
    dataset_num = dataset_config.get("DATASET_NUM", 2)
    shower_embed = dataset_config.get("SHOWER_EMBED", "")
    orig_shape = "orig" in shower_embed
    energy_loss_scale = dataset_config.get("ENERGY_LOSS_SCALE", 0.0)

    data = []
    energies = []

    for i, dataset in enumerate(dataset_config["FILES"]):
        data_, e_ = utils.DataLoader(
            os.path.join(flags.data_folder, dataset),
            dataset_config["SHAPE_PAD"],
            emax=dataset_config["EMAX"],
            emin=dataset_config["EMIN"],
            nevts=flags.nevts,
            max_deposit=dataset_config[
                "MAXDEP"
            ],  # noise can generate more deposited energy than generated
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

    avg_showers = std_showers = E_bins = None
    if cold_diffu:
        f_avg_shower = h5.File(dataset_config["AVG_SHOWER_LOC"])
        # Already pre-processed
        avg_showers = torch.from_numpy(
            f_avg_shower["avg_showers"][()].astype(np.float32)
        ).to(device=device)
        std_showers = torch.from_numpy(
            f_avg_shower["std_showers"][()].astype(np.float32)
        ).to(device=device)
        E_bins = torch.from_numpy(
            f_avg_shower["E_bins"][()].astype(np.float32)
        ).to(device=device)

    NN_embed = None
    if "NN" in shower_embed:
        if dataset_num == 1:
            if flags.binning_file is None:
                flags.binning_file = (
                    "../CaloChallenge/code/binning_dataset_1_photons.xml"
                )
            bins = XMLHandler.XMLHandler("photon", flags.binning_file)
        else:
            if flags.binning_file is None:
                flags.binning_file = (
                    "../CaloChallenge/code/binning_dataset_1_pions.xml"
                )
            bins = XMLHandler.XMLHandler("pion", flags.binning_file)

        NN_embed = NNConverter(bins=bins).to(device=device)

    dshape = dataset_config["SHAPE_PAD"]
    energies = np.reshape(energies, (-1))
    if not orig_shape:
        data = np.reshape(data, dshape)
    else:
        data = np.reshape(data, (len(data), -1))

    num_data = data.shape[0]

    torch_data_tensor = torch.from_numpy(data).to(
        device=device
    )  # creating a new tensor and stores it on the same memory from the original instantiated variables
    torch_E_tensor = torch.from_numpy(energies).to(device=device)
    del data

    torch_dataset = torchdata.TensorDataset(torch_E_tensor, torch_data_tensor)

    if flags.model == "Latent_Diffu":
        loader_encode = torchdata.DataLoader(
            torch_dataset, batch_size=batch_size, shuffle=False
        )
    elif flags.model == "Diffu":
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

    del torch_data_tensor
    if flags.model == "Diffu":
        del torch_E_tensor, train_dataset, val_dataset

    checkpoint_folder = "../models/{}_{}/".format(
        dataset_config["CHECKPOINT_NAME"], flags.model
    )
    if (
        flags.save_folder_append is not None
    ):  # optionally append another folder title
        checkpoint_folder = f"{checkpoint_folder}{flags.save_folder_append}/"
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    checkpoint = dict()
    checkpoint_path = os.path.join(checkpoint_folder, "checkpoint.pth")
    if flags.load and os.path.exists(checkpoint_path):
        print(
            "Loading training checkpoint from %s" % checkpoint_path, flush=True
        )
        checkpoint = torch.load(checkpoint_path, map_location=device)

    shape = (
        dataset_config["SHAPE_PAD"][1:]
        if (not orig_shape)
        else dataset_config["SHAPE_ORIG"][1:]
    )

    if flags.model == "Diffu":
        model = CaloDiffu(
            shape,
            config=dataset_config,
            training_obj=training_obj,
            NN_embed=NN_embed,
            nsteps=dataset_config["NSTEPS"],
            cold_diffu=cold_diffu,
            avg_showers=avg_showers,
            std_showers=std_showers,
            E_bins=E_bins,
        ).to(device=device)

    # Added a conditional statement, based off of flags.model, that runs the latent diffusion training pipeline to do the following three things:
    # Encode data into a latent space with autoencoder
    # Perform latent diffusion pipeline with encoded data
    # Store trained latent diffusion model into directory

    elif flags.model == "Latent_Diffu":
        ae_checkpoint = torch.load(flags.model_loc, map_location=device)

        # Instantiating autoencoder
        AE = CaloEnco(
            shape,
            config=dataset_config,
            training_obj="mean_pred",
            NN_embed=NN_embed,
            nsteps=dataset_config["NSTEPS"],
            cold_diffu=False,
            avg_showers=None,
            std_showers=None,
            E_bins=None,
            layer_sizes=flags.layer_sizes,
        ).to(device=device)

        if "model_state_dict" in ae_checkpoint.keys():
            AE.load_state_dict(ae_checkpoint["model_state_dict"])
        elif len(ae_checkpoint.keys()) > 1:
            AE.load_state_dict(ae_checkpoint)

        print("Encoding Data...")
        encoded_data = []
        for i, (E, data) in tqdm(
            enumerate(loader_encode, 0), unit="batch", total=len(loader_encode)
        ):
            E = E.to(device=device)
            data = data.to(device=device)
            enc = AE.encode(data, E).detach().cpu().numpy()
            if i == 0:
                encoded_data = enc
            else:
                encoded_data = np.concatenate((encoded_data, enc))
            del E, data

        encoded_data = torch.tensor(encoded_data).to(device=device)
        print("Data successfully encoded with shape:", encoded_data.shape)

        # Standardizing encoded data to have mean = 0 and std = 1
        encoded_mean = torch.mean(encoded_data).item()
        encoded_std = torch.std(encoded_data).item()
        encoded_data = (encoded_data - torch.mean(encoded_data)) / torch.std(
            encoded_data
        )
        with open(checkpoint_folder + "encoded_mean_std.txt", "w") as f:
            f.write(f"encoded_mean={encoded_mean}\nencoded_std={encoded_std}")

        max_downsample = np.array(encoded_data.shape)[-3:].min() // 2

        nTrain = int(round(flags.frac * num_data))
        nVal = num_data - nTrain
        energies_encoded_data = torchdata.TensorDataset(
            torch_E_tensor, encoded_data
        )
        train_dataset, val_dataset = torch.utils.data.random_split(
            energies_encoded_data, [nTrain, nVal]
        )

        loader_train = torchdata.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        loader_val = torchdata.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True
        )

        del torch_E_tensor, train_dataset, val_dataset

        shape = encoded_data.shape[1:]

        # Instantiating difussion model
        model = CaloDiffu(
            shape,
            config=dataset_config,
            training_obj=training_obj,
            NN_embed=None,
            nsteps=dataset_config["NSTEPS"],
            cold_diffu=cold_diffu,
            avg_showers=avg_showers,
            std_showers=std_showers,
            E_bins=E_bins,
            max_downsample=max_downsample,
            is_latent=True,
        ).to(device=device)

    else:
        print("Model %s not supported!" % flags.model)
        exit(1)

    if "model_state_dict" in checkpoint.keys():
        model.load_state_dict(checkpoint["model_state_dict"])
    elif len(checkpoint.keys()) > 1:
        model.load_state_dict(checkpoint)

    os.system(
        "cp CaloDiffu.py {}".format(checkpoint_folder)
    )  # bkp of model def
    os.system("cp models.py {}".format(checkpoint_folder))  # bkp of model def
    os.system(
        "cp {} {}".format(flags.config, checkpoint_folder)
    )  # bkp of config file

    early_stopper = EarlyStopper(
        patience=dataset_config["EARLYSTOP"], mode="diff", min_delta=1e-5
    )
    if "early_stop_dict" in checkpoint.keys() and not flags.reset_training:
        early_stopper.__dict__ = checkpoint["early_stop_dict"]
    print(early_stopper.__dict__)

    criterion = nn.MSELoss().to(device=device)

    optimizer = optim.Adam(model.parameters(), lr=float(dataset_config["LR"]))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, factor=0.1, patience=15, verbose=True
    )
    if "optimizer_state_dict" in checkpoint.keys() and not flags.reset_training:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint.keys() and not flags.reset_training:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    training_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    start_epoch = 0
    min_validation_loss = 99999.0
    if "train_loss_hist" in checkpoint.keys() and not flags.reset_training:
        training_losses = checkpoint["train_loss_hist"]
        val_losses = checkpoint["val_loss_hist"]
        start_epoch = checkpoint["epoch"] + 1

    # Training loop performing diffusion modeling with encoded data (latent diffusion)
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
            noise = torch.randn_like(data)

            if (
                cold_diffu
            ):  # cold diffusion interpolates from avg showers instead of pure noise
                noise = model.gen_cold_image(E, cold_noise_scale, noise)
            batch_loss = model.compute_loss(
                data,
                E,
                noise=noise,
                t=t,
                loss_type=loss_type,
                energy_loss_scale=energy_loss_scale,
            )
            batch_loss.backward()

            optimizer.step()
            train_loss += batch_loss.item()

            del data, E, noise, batch_loss

        train_loss = train_loss / len(loader_train)
        training_losses[epoch] = train_loss
        print("loss: " + str(train_loss))

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
            noise = torch.randn_like(vdata)
            if cold_diffu:
                noise = model.gen_cold_image(vE, cold_noise_scale, noise)

            batch_loss = model.compute_loss(
                vdata,
                vE,
                noise=noise,
                t=t,
                loss_type=loss_type,
                energy_loss_scale=energy_loss_scale,
            )

            val_loss += batch_loss.item()
            del vdata, vE, noise, batch_loss

        val_loss = val_loss / len(loader_val)
        val_losses[epoch] = val_loss
        print("val_loss: " + str(val_loss), flush=True)

        scheduler.step(torch.tensor([train_loss]))

        if val_loss < min_validation_loss:
            torch.save(
                model.state_dict(),
                os.path.join(checkpoint_folder, "best_val.pth"),
            )
            min_validation_loss = val_loss

        if early_stopper.early_stop(val_loss - train_loss):
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

    print("Saving to %s" % checkpoint_folder, flush=True)
    torch.save(model.state_dict(), os.path.join(checkpoint_folder, "final.pth"))

    with open(checkpoint_folder + "/training_losses.txt", "w") as tfileout:
        tfileout.write(
            "\n".join("{}".format(tl) for tl in training_losses) + "\n"
        )
    with open(checkpoint_folder + "/validation_losses.txt", "w") as vfileout:
        vfileout.write("\n".join("{}".format(vl) for vl in val_losses) + "\n")
