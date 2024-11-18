import numpy as np
import os
import argparse
import h5py as h5
import torch
import torch.optim as optim
import torch.utils.data as torchdata
from tqdm import tqdm
import sys
from torchinfo import torchinfo

from CaloDiffu import CaloDiffu
from autoencoder.CaloEnco import CaloEnco
import utils

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class DiffusionTrainer:
    def __init__(self):
        self.device = self.set_device()
        self.flags = self.parse_arguments()
        self.dataset_config = utils.LoadJson(self.flags.config)
        self.setup_paths()
        self.setup_training_params()
        self.setup_data()
        self.setup_model()
        self.setup_training_components()

    def set_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Device: cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Device: mps")
        else:
            device = torch.device("cpu")
            print("Device: cpu")
        return device

    def parse_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_folder", default="/wclustre/cms_mlsim/denoise/CaloChallenge/", help="Folder containing data and MC files")
        parser.add_argument("--model", default="Diffu", help="Diffusion model to train")
        parser.add_argument("-c", "--config", default="configs/test.json", help="Config file with training parameters")
        parser.add_argument("--nevts", type=int, default=-1, help="Number of events to load")
        parser.add_argument("--frac", type=float, default=0.85, help="Fraction of total events used for training")
        parser.add_argument("--load", action="store_true", default=False, help="Load pretrained weights to continue the training")
        parser.add_argument("--seed", type=int, default=1234, help="Pytorch seed")
        parser.add_argument("--reset_training", action="store_true", default=False, help="Retrain")
        parser.add_argument("--layer_sizes", type=int, nargs="+", default=None, help="Manual layer sizes input instead of from config file")
        parser.add_argument("--binning_file", type=str, default=None)
        parser.add_argument("--save_folder_append", type=str, default=None, help="Optional text to append to training folder to separate outputs of training runs with the same config file")
        parser.add_argument("--model_loc", default="test", help="Location of model")
        return parser.parse_args()

    def setup_paths(self):
        def trim_file_path(cwd: str, num_back: int):
            split_path = cwd.split("/")
            trimmed_split_path = split_path[:-num_back]
            return "/".join(trimmed_split_path)

        cwd = __file__
        self.calo_challenge_dir = trim_file_path(cwd=cwd, num_back=3)
        sys.path.append(self.calo_challenge_dir)
        from scripts.utils import NNConverter, EarlyStopper, nn
        from CaloChallenge.code import XMLHandler
        self.NNConverter = NNConverter
        self.EarlyStopper = EarlyStopper
        self.nn = nn
        self.XMLHandler = XMLHandler

    def setup_training_params(self):
        torch.manual_seed(self.flags.seed)
        self.cold_diffu = self.dataset_config.get("COLD_DIFFU", False)
        self.cold_noise_scale = self.dataset_config.get("COLD_NOISE", 1.0)
        self.nholdout = self.dataset_config.get("HOLDOUT", 0)
        self.batch_size = self.dataset_config["BATCH"]
        self.num_epochs = self.dataset_config["MAXEPOCH"]
        self.early_stop = self.dataset_config["EARLYSTOP"]
        self.training_obj = self.dataset_config.get("TRAINING_OBJ", "noise_pred")
        self.loss_type = self.dataset_config.get("LOSS_TYPE", "l2")
        self.dataset_num = self.dataset_config.get("DATASET_NUM", 2)
        self.shower_embed = self.dataset_config.get("SHOWER_EMBED", "")
        self.orig_shape = "orig" in self.shower_embed
        self.energy_loss_scale = self.dataset_config.get("ENERGY_LOSS_SCALE", 0.0)

    def setup_data(self):
        self.data = []
        self.energies = []

        for i, dataset in enumerate(self.dataset_config["FILES"]):
            data_, e_ = utils.DataLoader(
                os.path.join(self.flags.data_folder, dataset),
                self.dataset_config["SHAPE_PAD"],
                emax=self.dataset_config["EMAX"],
                emin=self.dataset_config["EMIN"],
                nevts=self.flags.nevts,
                max_deposit=self.dataset_config["MAXDEP"],
                logE=self.dataset_config["logE"],
                showerMap=self.dataset_config["SHOWERMAP"],
                nholdout=self.nholdout if (i == len(self.dataset_config["FILES"]) - 1) else 0,
                dataset_num=self.dataset_num,
                orig_shape=self.orig_shape,
            )

            if i == 0:
                self.data = data_
                self.energies = e_
            else:
                self.data = np.concatenate((self.data, data_))
                self.energies = np.concatenate((self.energies, e_))

        self.setup_cold_diffusion()
        self.setup_nn_embed()
        self.prepare_data_tensors()

    def setup_cold_diffusion(self):
        self.avg_showers = self.std_showers = self.E_bins = None
        if self.cold_diffu:
            f_avg_shower = h5.File(self.dataset_config["AVG_SHOWER_LOC"])
            self.avg_showers = torch.from_numpy(f_avg_shower["avg_showers"][()].astype(np.float32)).to(device=self.device)
            self.std_showers = torch.from_numpy(f_avg_shower["std_showers"][()].astype(np.float32)).to(device=self.device)
            self.E_bins = torch.from_numpy(f_avg_shower["E_bins"][()].astype(np.float32)).to(device=self.device)

    def setup_nn_embed(self):
        self.NN_embed = None
        if "NN" in self.shower_embed:
            if self.dataset_num == 1:
                if self.flags.binning_file is None:
                    self.flags.binning_file = "/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/CaloChallenge/code/binning_dataset_1_photons.xml"
                bins = self.XMLHandler.XMLHandler("photon", self.flags.binning_file)
            else:
                if self.flags.binning_file is None:
                    self.flags.binning_file = "/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/CaloChallenge/code/binning_dataset_1_pions.xml"
                bins = self.XMLHandler.XMLHandler("pion", self.flags.binning_file)

            self.NN_embed = self.NNConverter(bins=bins).to(device=self.device)

    def prepare_data_tensors(self):
        dshape = self.dataset_config["SHAPE_PAD"]
        self.energies = np.reshape(self.energies, (-1))
        if not self.orig_shape:
            self.data = np.reshape(self.data, dshape)
        else:
            self.data = np.reshape(self.data, (len(self.data), -1))

        self.num_data = self.data.shape[0]

        self.torch_data_tensor = torch.from_numpy(self.data).to(device=self.device)
        self.torch_E_tensor = torch.from_numpy(self.energies).to(device=self.device)
        del self.data

        self.torch_dataset = torchdata.TensorDataset(self.torch_E_tensor, self.torch_data_tensor)

        if self.flags.model == "Latent_Diffu":
            self.loader_encode = torchdata.DataLoader(self.torch_dataset, batch_size=self.batch_size, shuffle=False)
        elif self.flags.model == "Diffu":
            nTrain = int(round(self.flags.frac * self.num_data))
            nVal = self.num_data - nTrain
            train_dataset, val_dataset = torch.utils.data.random_split(self.torch_dataset, [nTrain, nVal])
            self.loader_train = torchdata.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            self.loader_val = torchdata.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        del self.torch_data_tensor
        if self.flags.model == "Diffu":
            del self.torch_E_tensor, train_dataset, val_dataset

    def setup_model(self):
        self.checkpoint_folder = "../models/{}_{}/".format(self.dataset_config["CHECKPOINT_NAME"], self.flags.model)
        if self.flags.save_folder_append is not None:
            self.checkpoint_folder = f"{self.checkpoint_folder}{self.flags.save_folder_append}/"
        if not os.path.exists(self.checkpoint_folder):
            os.makedirs(self.checkpoint_folder)

        self.checkpoint = dict()
        self.checkpoint_path = os.path.join(self.checkpoint_folder, "checkpoint.pth")
        if self.flags.load and os.path.exists(self.checkpoint_path):
            print("Loading training checkpoint from %s" % self.checkpoint_path, flush=True)
            self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        self.shape = self.dataset_config["SHAPE_PAD"][1:] if (not self.orig_shape) else self.dataset_config["SHAPE_ORIG"][1:]

        if self.flags.model == "Diffu":
            self.setup_diffusion_model()
        elif self.flags.model == "Latent_Diffu":
            self.setup_latent_diffusion_model()
        else:
            print(f"Model {self.flags.model} not supported!")
            exit(1)

    def setup_diffusion_model(self):
        self.model = CaloDiffu(
            self.shape,
            config=self.dataset_config,
            training_obj=self.training_obj,
            NN_embed=self.NN_embed,
            nsteps=self.dataset_config["NSTEPS"],
            cold_diffu=self.cold_diffu,
            avg_showers=self.avg_showers,
            std_showers=self.std_showers,
            E_bins=self.E_bins,
        ).to(device=self.device)

        print("\nDiffusion Model Summary:")
        print(torchinfo.summary(self.model))

    def setup_latent_diffusion_model(self):
        ae_checkpoint = torch.load(self.flags.model_loc, map_location=self.device)

        self.AE = CaloEnco(
            self.shape,
            config=self.dataset_config,
            training_obj="mean_pred",
            NN_embed=self.NN_embed,
            nsteps=self.dataset_config["NSTEPS"],
            cold_diffu=False,
            avg_showers=None,
            std_showers=None,
            E_bins=None,
            layer_sizes=self.flags.layer_sizes,
        ).to(device=self.device)

        print("\nAutoEncoder Model Summary:")
        print(torchinfo.summary(self.AE))

        if "model_state_dict" in ae_checkpoint.keys():
            self.AE.load_state_dict(ae_checkpoint["model_state_dict"])
        elif len(ae_checkpoint.keys()) > 1:
            self.AE.load_state_dict(ae_checkpoint)

        print("Encoding Data...")
        encoded_data = []
        for i, (E, data) in tqdm(enumerate(self.loader_encode, 0), unit="batch", total=len(self.loader_encode)):
            E = E.to(device=self.device)
            data = data.to(device=self.device)
            enc = self.AE.encode(data, E).detach().cpu().numpy()
            if i == 0:
                encoded_data = enc
            else:
                encoded_data = np.concatenate((encoded_data, enc))
            del E, data

        encoded_data = torch.tensor(encoded_data).to(device=self.device)
        print("Data successfully encoded with shape:", encoded_data.shape)

        encoded_mean = torch.mean(encoded_data).item()
        encoded_std = torch.std(encoded_data).item()
        encoded_data = (encoded_data - torch.mean(encoded_data)) / torch.std(encoded_data)
        with open(self.checkpoint_folder + "encoded_mean_std.txt", "w") as f:
            f.write(f"encoded_mean={encoded_mean}\nencoded_std={encoded_std}")

        max_downsample = np.array(encoded_data.shape)[-3:].min() // 2

        nTrain = int(round(self.flags.frac * self.num_data))
        nVal = self.num_data - nTrain
        energies_encoded_data = torchdata.TensorDataset(self.torch_E_tensor, encoded_data)
        train_dataset, val_dataset = torch.utils.data.random_split(energies_encoded_data, [nTrain, nVal])

        self.loader_train = torchdata.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.loader_val = torchdata.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        del self.torch_E_tensor, train_dataset, val_dataset

        self.shape = encoded_data.shape[1:]

        self.model = CaloDiffu(
            self.shape,
            config=self.dataset_config,
            training_obj=self.training_obj,
            NN_embed=None,
            nsteps=self.dataset_config["NSTEPS"],
            cold_diffu=self.cold_diffu,
            avg_showers=self.avg_showers,
            std_showers=self.std_showers,
            E_bins=self.E_bins,
            max_downsample=max_downsample,
            is_latent=True,
        ).to(device=self.device)

        print("\nLatent Diffusion Model Summary:")
        print(torchinfo.summary(self.model))

    def setup_training_components(self):
        if "model_state_dict" in self.checkpoint.keys():
            self.model.load_state_dict(self.checkpoint["model_state_dict"])
        elif len(self.checkpoint.keys()) > 1:
            self.model.load_state_dict(self.checkpoint)

        os.system(f"cp CaloDiffu.py {self.checkpoint_folder}")
        os.system(f"cp models.py {self.checkpoint_folder}")
        os.system(f"cp {self.flags.config} {self.checkpoint_folder}")

        self.early_stopper = self.EarlyStopper(patience=self.dataset_config["EARLYSTOP"], mode="diff", min_delta=1e-5)
        if "early_stop_dict" in self.checkpoint.keys() and not self.flags.reset_training:
            self.early_stopper.__dict__ = self.checkpoint["early_stop_dict"]
        print(self.early_stopper.__dict__)

        self.criterion = self.nn.MSELoss().to(device=self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=float(self.dataset_config["LR"]))
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, factor=0.1, patience=15, verbose=True)
        if "optimizer_state_dict" in self.checkpoint.keys() and not self.flags.reset_training:
            self.optimizer.load_state_dict(self.checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in self.checkpoint.keys() and not self.flags.reset_training:
            self.scheduler.load_state_dict(self.checkpoint["scheduler_state_dict"])

        self.training_losses = np.zeros(self.num_epochs)
        self.val_losses = np.zeros(self.num_epochs)
        self.start_epoch = 0
        self.min_validation_loss = 99999.0
        if "train_loss_hist" in self.checkpoint.keys() and not self.flags.reset_training:
            self.training_losses = self.checkpoint["train_loss_hist"]
            self.val_losses = self.checkpoint["val_loss_hist"]
            self.start_epoch = self.checkpoint["epoch"] + 1

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            print(f"Beginning epoch {epoch}", flush=True)
            self.train_epoch(epoch)
            self.validate_epoch(epoch)
            self.update_scheduler(epoch)
            self.save_checkpoint(epoch)
            if self.early_stopper.early_stop(self.val_losses[epoch] - self.training_losses[epoch]):
                print("Early stopping!")
                break

        self.save_final_model()

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0

        for i, (E, data) in tqdm(enumerate(self.loader_train, 0), unit="batch", total=len(self.loader_train)):
            self.model.zero_grad()
            self.optimizer.zero_grad()

            data = data.to(device=self.device)
            E = E.to(device=self.device)

            t = torch.randint(0, self.model.nsteps, (data.size()[0],), device=self.device).long()
            noise = torch.randn_like(data)

            if self.cold_diffu:
                noise = self.model.gen_cold_image(E, self.cold_noise_scale, noise)
            batch_loss = self.model.compute_loss(data, E, noise=noise, t=t, loss_type=self.loss_type, energy_loss_scale=self.energy_loss_scale)
            batch_loss.backward()

            self.optimizer.step()
            train_loss += batch_loss.item()

            del data, E, noise, batch_loss

        train_loss = train_loss / len(self.loader_train)
        self.training_losses[epoch] = train_loss
        print(f"loss: {train_loss}")

    def validate_epoch(self, epoch):
        self.model.eval()
        val_loss = 0

        for i, (vE, vdata) in tqdm(enumerate(self.loader_val, 0), unit="batch", total=len(self.loader_val)):
            vdata = vdata.to(device=self.device)
            vE = vE.to(device=self.device)

            t = torch.randint(0, self.model.nsteps, (vdata.size()[0],), device=self.device).long()
            noise = torch.randn_like(vdata)
            if self.cold_diffu:
                noise = self.model.gen_cold_image(vE, self.cold_noise_scale, noise)

            batch_loss = self.model.compute_loss(vdata, vE, noise=noise, t=t, loss_type=self.loss_type, energy_loss_scale=self.energy_loss_scale)

            val_loss += batch_loss.item()
            del vdata, vE, noise, batch_loss

        val_loss = val_loss / len(self.loader_val)
        self.val_losses[epoch] = val_loss
        print(f"val_loss: {val_loss}", flush=True)

        if val_loss < self.min_validation_loss:
            torch.save(self.model.state_dict(), os.path.join(self.checkpoint_folder, "best_val.pth"))
            self.min_validation_loss = val_loss

    def update_scheduler(self, epoch):
        self.scheduler.step(torch.tensor([self.training_losses[epoch]]))

    def save_checkpoint(self, epoch):
        self.model.eval()
        print("SAVING")

        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_loss_hist": self.training_losses,
            "val_loss_hist": self.val_losses,
            "early_stop_dict": self.early_stopper.__dict__,
        }, self.checkpoint_path)

        with open(self.checkpoint_folder + "/training_losses.txt", "w") as tfileout:
            tfileout.write("\n".join("{}".format(tl) for tl in self.training_losses) + "\n")
        with open(self.checkpoint_folder + "/validation_losses.txt", "w") as vfileout:
            vfileout.write("\n".join("{}".format(vl) for vl in self.val_losses) + "\n")

    def save_final_model(self):
        print(f"Saving to {self.checkpoint_folder}", flush=True)
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_folder, "final.pth"))

        with open(self.checkpoint_folder + "/training_losses.txt", "w") as tfileout:
            tfileout.write("\n".join("{}".format(tl) for tl in self.training_losses) + "\n")
        with open(self.checkpoint_folder + "/validation_losses.txt", "w") as vfileout:
            vfileout.write("\n".join("{}".format(vl) for vl in self.val_losses) + "\n")

if __name__ == "__main__":
    trainer = DiffusionTrainer()
    trainer.train()
