"""This script contains a refactored version of train_ae.py.

To run this python script to train an autoencoder, modify train_ae_4x64.sh or 
train_ae_4x32_ds3.shto use train_ae_class.py instead of train_ae.py. The model training loss,
validation losses, and the Frechet physics distance will be saved within
a sub-subfolder inside the ae_models folder, which can be found
outside of the 2023-Autumn-Clinic...folder.
For testing purposes during development, set the the take_subset parameter
inside the main block to True when initializing the AutoencoderTrainer.
This will run the training script with only 500 rows of data.
"""

import os
from typing import Optional

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import os
from pathlib import Path

import jetnet
import torch
import torch.optim as optim
import torch.utils.data as torchdata
from tqdm import tqdm

from autoencoder.CaloEnco import CaloEnco
from CaloChallenge.code.XMLHandler import XMLHandler
from scripts.utils import DataLoader, EarlyStopper, LoadJson, NNConverter


class AutoencoderTrainer:
    """A trainer class for Autoencoder models.

    The train() method handles data loading, model setup, training loop
    execution, and checkpoint management.
    """

    def __init__(self, set_seed: bool = False, take_subset: bool = False) -> None:
        """Initialize the autoencoder trainer.
        
        First loads the config file
        and optional CLI arguments, which inform the setup for training. By
        default, does not set a random seed
        """
        self.device = self._set_device()
        self.args = self._parse_cli_arguments()
        self.config = LoadJson(self.args.config)
        self._set_random_seed(set_seed)
        self.train_loader, self.val_loader = self._prepare_datasets(take_subset)
        self.checkpoint_folder = self._create_checkpoint_folder(set_seed, take_subset)
        self.nn_embed = self._setup_nn_embedding()
        self.model = self._initialize_model()
        self.optimizer, self.scheduler, self.early_stopper = (
            self._setup_training_components()
        )

    def _set_device(self) -> torch.device:
        """Determines and sets up the computing device, using the GPU when available."""
        if torch.cuda.is_available():
            print("Using CUDA")
            return torch.device("cuda")
        print("Using CPU")
        return torch.device("cpu")

    def _parse_cli_arguments(self) -> argparse.Namespace:
        """Parse command line arguments for training configuration."""
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
            help="Optional text to append to training folder to \
            separate outputs of training runs with the same config file",
        )
        parser.add_argument(
            "--save_folder_absolute",
            type=str,
            default=None,
            help="Optional path to use for training folder instead of \
            default ..",
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
            help="Turns off early stop functionality and defailts to \
            max epochs",
        )
        parser.add_argument(
            "--max_epochs",
            type=int,
            default=None,
            help="Manually assign a maximum number of epochs",
        )

        return parser.parse_args()

    def _set_random_seed(self, set_seed: bool) -> None:
        """Sets the pytorch and numpy random seeds."""
        torch.manual_seed(self.args.seed)
        if set_seed:
            np.random.seed(self.args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.args.seed)
                torch.cuda.manual_seed_all(self.args.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    def _prepare_datasets(
        self, take_subset: bool
    ) -> tuple[torchdata.DataLoader, torchdata.DataLoader]:
        """Reshape the data as needed and split data into training and validation datasets."""
        orig_shape = "orig" in self.config.get("SHOWER_EMBED", "")
        shape_pad = self.config["SHAPE_PAD"]
        data, energies = [], []
        for i, dataset in enumerate(self.config["FILES"]):
            data_, e_ = DataLoader(
                Path(self.args.data_folder) / dataset,
                shape_pad,
                emax=self.config["EMAX"],
                emin=self.config["EMIN"],
                nevts=self.args.nevts,
                max_deposit=self.config[
                    "MAXDEP"
                ],  # Noise can generate more deposited energy than generated
                logE=self.config["logE"],
                showerMap=self.config["SHOWERMAP"],
                nholdout=self.config.get("HOLDOUT", 0)
                if (i == len(self.config["FILES"]) - 1)
                else 0,
                dataset_num=self.config.get("DATASET_NUM", 2),
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
            train_dataset, batch_size=self.config["BATCH"], shuffle=True
        )
        loader_val = torchdata.DataLoader(
            val_dataset, batch_size=self.config["BATCH"], shuffle=True
        )

        del data, torch_data_tensor, torch_E_tensor, train_dataset, val_dataset

        return loader_train, loader_val

    def _create_checkpoint_folder(self, set_seed: bool, take_subset: bool) -> str:
        """Create and prepare checkpoint folder file path, which includes info regarding learning rate and layer sizes."""
        learning_rate = float(
            self.args.learning_rate[0] if self.args.learning_rate else self.config["LR"]
        )
        subset = "subset" if take_subset else "full_data"
        random = "deterministic" if set_seed else "random"
        checkpoint_folder = (
            f"../ae_models/{subset}-{random}/"
            f"{self.config['CHECKPOINT_NAME']}_{self.args.model}_"
            f"{'-'.join(map(str, self.args.layer_sizes or []))}_"
            f"{learning_rate}/"
        )
        print("Checkpoint folder is at: ", checkpoint_folder)

        # By default these arguments are None, so these blocks won't execute
        if self.args.save_folder_absolute:
            checkpoint_folder = (
                f"{self.args.save_folder_absolute}{checkpoint_folder[2:]}"
            )
        if self.args.save_folder_append:
            checkpoint_folder = f"{checkpoint_folder}{self.args.save_folder_append}/"

        checkpoint_folder_path = Path(checkpoint_folder)
        if not checkpoint_folder_path.exists():
            checkpoint_folder_path.mkdir(parents=True, exist_ok = True)

        print(f"Checkpoint folder created at: {checkpoint_folder}")

        checkpoint = {}
        checkpoint_path = Path(checkpoint_folder) / "checkpoint.pth"

        if self.args.load and Path.exists(checkpoint_path):
            print(f"Loading training checkpoint from {checkpoint_path}", flush=True)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            print(checkpoint.keys())

        self.checkpoint = checkpoint
        self.checkpoint_path = checkpoint_path

        return checkpoint_folder

    def _setup_nn_embedding(self) -> Optional[NNConverter]:
        """Set up neural network embedding if required based on the SHOWER_EMBED specified in the config file.
        
        The binning file is set to the CLI argument if provided, otherwise a default .xml for dataset 1 is used.
        """
        if "NN" not in self.config.get("SHOWER_EMBED", ""):
            return None

        dataset_num = self.config.get("DATASET_NUM", 2)
        if dataset_num == 1:
            if self.args.binning_file is None:
                self.args.binning_file = (
                    "../CaloChallenge/code/binning_dataset_1_photons.xml"
                )
            bins = XMLHandler("photon", self.args.binning_file)
        else:
            if self.args.binning_file is None:
                self.args.binning_file = (
                    "../CaloChallenge/code/binning_dataset_1_pions.xml"
                )
            bins = XMLHandler("pion", self.args.binning_file)

        NN_embed = NNConverter(bins=bins).to(device=self.device)

        return NN_embed

    def _initialize_model(self) -> CaloEnco:
        """Initialize the autoencoder model based on the SHOWER_EMBED, SHAPE_PAD, SHAPE_ORIG, and NSTEPS in the config file.

        Loads the model from checkpoints if possible.
        """
        if self.args.model != "AE":
            raise ValueError(f"Model {self.args.model} not supported!")

        orig_shape = "orig" in self.config.get("SHOWER_EMBED", "")
        shape = (
            self.config["SHAPE_PAD"][1:]
            if not orig_shape
            else self.config["SHAPE_ORIG"][1:]
        )

        model = CaloEnco(
            shape,
            config=self.config,
            training_obj="mean_pred",
            NN_embed=self.nn_embed,
            nsteps=self.config["NSTEPS"],
            cold_diffu=False,
            avg_showers=None,
            std_showers=None,
            E_bins=None,
            resnet_set=self.args.resnet_set,
            layer_sizes=self.args.layer_sizes,
        ).to(device=self.device)

        # Load checkpoint if exists
        if "model_state_dict" in self.checkpoint:
            model.load_state_dict(self.checkpoint["model_state_dict"])
        elif len(self.checkpoint) > 1:
            model.load_state_dict(self.checkpoint)

        return model

    def _setup_training_components(
        self,
    ) -> tuple[
        optim.Optimizer, optim.lr_scheduler._LRScheduler, Optional[EarlyStopper]
    ]:
        """Set up optimizer, scheduler, and early stopping components for training."""
        learning_rate = float(
            self.args.learning_rate[0] if self.args.learning_rate else self.config["LR"]
        )
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=15, verbose=True
        )

        # Load saved states if they exist
        if "optimizer_state_dict" in self.checkpoint and not self.args.reset_training:
            optimizer.load_state_dict(self.checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in self.checkpoint and not self.args.reset_training:
            scheduler.load_state_dict(self.checkpoint["scheduler_state_dict"])

        # Setup early stopping if enabled
        early_stopper = None
        if not self.args.no_early_stop:
            early_stopper = EarlyStopper(
                patience=self.args.patience, mode="diff", min_delta=self.args.min_delta
            )
            if "early_stop_dict" in self.checkpoint and not self.args.reset_training:
                early_stopper.__dict__ = self.checkpoint["early_stop_dict"]

        return optimizer, scheduler, early_stopper

    def _get_num_epochs(self) -> int:
        """Returns the number of epochs specified in the config file.
        
        Exception if early stopping is not enabled and there is a max_epochs set through the CLI args.
        """
        if self.args.no_early_stop and self.args.max_epochs:
            return self.args.max_epochs

        return self.config["MAXEPOCH"]

    def _run_epoch(self, loader: DataLoader, training: bool = True) -> float:
        """Run one epoch of training or validation."""
        self.model.train() if training else self.model.eval()
        total_loss = 0

        for E, data in tqdm(loader, unit="batch"):
            if training:
                self.model.zero_grad()
                self.optimizer.zero_grad()

            data = data.to(device=self.device)
            E = E.to(device=self.device)

            t = torch.randint(
                0, self.model.nsteps, (data.size()[0],), device=self.device
            ).long()

            batch_loss = self.model.compute_loss(
                data,
                E,
                t=t,
                loss_type="mse",
                energy_loss_scale=self.config.get("ENERGY_LOSS_SCALE", 0.0),
            )

            if training:
                batch_loss.backward()
                self.optimizer.step()

            total_loss += batch_loss.item()
            del data, E, batch_loss

        return total_loss / len(loader)

    def _train_epoch(self) -> float:
        """Train for one epoch and return average loss."""
        return self._run_epoch(self.train_loader, training=True)

    def _validate_epoch(self) -> tuple[float, float]:
        """Validate the model, return average loss and FPD score."""
        self.model.eval()
        total_loss = 0
        all_real_features = []
        all_gen_features = []
        with torch.no_grad():
            for E, data in tqdm(self.val_loader, unit="batch"):
                data = data.to(device=self.device)
                E = E.to(device=self.device)

                # Generate time step embedding
                t = torch.zeros((data.shape[0],), dtype=torch.long, device=self.device)
                t_emb = self.model.do_time_embed(t)

                # Reconstruct data with the autoencoder
                model_out = self.model.pred(data, E, t_emb)

                # Compute loss
                batch_loss = self.model.compute_loss(
                    data,
                    E,
                    t=t,
                    loss_type="mse",
                    energy_loss_scale=self.config.get("ENERGY_LOSS_SCALE", 0.0),
                )
                total_loss += batch_loss.item()

                # Store features for FPD computation
                all_real_features.append(data.cpu())
                all_gen_features.append(model_out.cpu())

        # Concatenate all features across batches
        real_features = torch.cat(all_real_features, dim=0)
        gen_features = torch.cat(all_gen_features, dim=0)

        # Compute FPD
        fpd_val, fpd_err = jetnet.evaluation.fpd(
            real_features=real_features.numpy(), gen_features=gen_features.numpy()
        )

        return total_loss / len(self.val_loader), fpd_val

    def _save_model(self, filename: str) -> None:
        """Save model state."""
        print(f"Saving to {self.checkpoint_folder}", flush=True)
        torch.save(
            self.model.state_dict(), Path(self.checkpoint_folder)/ filename)

    def _save_checkpoint(
        self,
        epoch: int,
        training_losses: np.ndarray,
        val_losses: np.ndarray,
        fpd_scores: np.ndarray,
    ) -> None:
        """Save full training state, including FPD scores."""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "train_loss_hist": training_losses,
                "val_loss_hist": val_losses,
                "fpd_scores": fpd_scores,
                "early_stop_dict": self.early_stopper.__dict__,
            },
            self.checkpoint_path,
        )

    def _write_losses(self, loss_array: np.ndarray, filename: str) -> None:
        """Writes loss values & FPD scores into a .txt file."""
        path = Path(self.checkpoint_folder) / filename
        with path.open("w") as f:
            for loss in loss_array:
                f.write(f"{loss}\n")

    def train(self) -> None:
        """Train the model for specified number of epochs.

        Logs validation losses and FPD scores.
        """
        # Compute initial validation loss & FPD
        with torch.no_grad():
            initial_loss, initial_fpd = self._validate_epoch()

        training_losses, val_losses, fpd_scores = (
            np.array([]),
            np.array([initial_loss]),
            np.array([initial_fpd]),
        )
        start_epoch = 0
        min_validation_loss = 99999.0
        num_epochs = self._get_num_epochs()

        # If training history exists, start from that epoch
        if "train_loss_hist" in self.checkpoint.keys() and not self.args.reset_training:
            training_losses = self.checkpoint["train_loss_hist"]
            val_losses = self.checkpoint["val_loss_hist"]
            fpd_scores = self.checkpoint.get("fpd_scores", fpd_scores)
            start_epoch = self.checkpoint["epoch"] + 1

        for epoch in range(start_epoch, num_epochs):
            print(f"Beginning epoch {epoch}", flush=True)

            train_loss = self._train_epoch()
            val_loss, fpd_val = self._validate_epoch()
            print(f"loss: {train_loss}")
            print(f"val_loss: {val_loss}")
            print(f"FPD: {fpd_val:.6f}", flush=True)

            training_losses = np.append(training_losses, train_loss)
            val_losses = np.append(val_losses, val_loss)
            fpd_scores = np.append(fpd_scores, fpd_val)

            self.scheduler.step(torch.tensor([train_loss]))

            if val_loss < min_validation_loss:
                self._save_model("best_val.pth")
                min_validation_loss = val_loss

            if not self.args.no_early_stop and self.early_stopper.early_stop(
                val_loss - train_loss
            ):
                print("Early stopping!")
                break

            # Save checkpoint
            self._save_checkpoint(epoch, training_losses, val_losses, fpd_scores)
            self._write_losses(training_losses, "training_losses")
            self._write_losses(val_losses, "validation_losses")
            self._write_losses(fpd_scores, "fpd_scores")  # Log FPD scores

        self._save_model("final.pth")


if __name__ == "__main__":
    trainer = AutoencoderTrainer(set_seed=True, take_subset=False)
    trainer.train()
