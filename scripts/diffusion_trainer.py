import numpy as np
import os
import torch
import torch.optim as optim
import torch.utils.data as torchdata
import h5py as h5
import argparse
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime

from CaloDiffu import CaloDiffu
from autoencoder.CaloEnco import CaloEnco
from scripts.utils import NNConverter, EarlyStopper
from CaloChallenge.code import XMLHandler
import utils

logger = logging.getLogger(__name__)

class DiffusionTrainer:
    """Class to handle training of diffusion models, including both regular and latent diffusion"""
    def __init__(self, args, device=None):
        """Initialize the trainer with command line arguments"""
        self.args = args
        self.device = self._setup_device(device)
        
        # Training state
        self.model = None
        self.autoencoder = None
        self.optimizer = None
        self.scheduler = None
        self.early_stopper = None
        self.epoch = 1
        self.best_epoch = 0
        self.best_metrics = {'loss': float('inf')}
        
        # Data handling
        self.train_loader = None
        self.val_loader = None
        self.data_shape = None
        self.encoded_stats = None

        # Setup training directory
        self.checkpoint_folder = self._setup_checkpoint_folder()
        Path(self.checkpoint_folder).mkdir(parents=True, exist_ok=True)

    def _setup_device(self, device):
        """Setup the computation device"""
        if device is not None:
            return device
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _setup_checkpoint_folder(self):
        """Create checkpoint folder path"""
        folder = f"../models/{self.args.config['CHECKPOINT_NAME']}_{self.args.model}/"
        if self.args.save_folder_append:
            folder = f"{folder}{self.args.save_folder_append}/"
        return folder

    def load_data(self):
        """Load and prepare training data"""
        data = []
        energies = []
        
        # Load data from all specified files
        for i, dataset in enumerate(self.args.config["FILES"]):
            data_, e_ = utils.DataLoader(
                os.path.join(self.args.data_folder, dataset),
                self.args.config["SHAPE_PAD"],
                emax=self.args.config["EMAX"],
                emin=self.args.config["EMIN"],
                nevts=self.args.nevts,
                max_deposit=self.args.config["MAXDEP"],
                logE=self.args.config["logE"],
                showerMap=self.args.config["SHOWERMAP"],
                nholdout=self.args.config.get("HOLDOUT", 0) if i == len(self.args.config["FILES"]) - 1 else 0,
                dataset_num=self.args.config.get("DATASET_NUM", 2),
                orig_shape="orig" in self.args.config.get("SHOWER_EMBED", "")
            )
            
            if i == 0:
                data, energies = data_, e_
            else:
                data = np.concatenate((data, data_))
                energies = np.concatenate((energies, e_))

        # Reshape data
        dshape = self.args.config["SHAPE_PAD"]
        energies = np.reshape(energies, (-1))
        if not "orig" in self.args.config.get("SHOWER_EMBED", ""):
            data = np.reshape(data, dshape)
        else:
            data = np.reshape(data, (len(data), -1))

        # Convert to torch tensors
        data_tensor = torch.from_numpy(data).to(self.device)
        energy_tensor = torch.from_numpy(energies).to(self.device)
        
        # Create dataset and split into train/val
        dataset = torchdata.TensorDataset(energy_tensor, data_tensor)
        
        if self.args.model == "Latent_Diffu":
            self.train_loader = torchdata.DataLoader(dataset, batch_size=self.args.config["BATCH"], shuffle=False)
        else:
            train_size = int(round(self.args.frac * len(dataset)))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            self.train_loader = torchdata.DataLoader(train_dataset, batch_size=self.args.config["BATCH"], shuffle=True)
            self.val_loader = torchdata.DataLoader(val_dataset, batch_size=self.args.config["BATCH"], shuffle=True)

        self.data_shape = data.shape[1:] if not "orig" in self.args.config.get("SHOWER_EMBED", "") else self.args.config["SHAPE_ORIG"][1:]

    def setup_model(self):
        """Setup the model architecture"""
        if self.args.model == "Diffu":
            self.model = self._setup_diffusion_model()
        elif self.args.model == "Latent_Diffu":
            self.model = self._setup_latent_diffusion_model()
        else:
            raise ValueError(f"Model {self.args.model} not supported!")

    def _setup_diffusion_model(self):
        """Setup regular diffusion model"""
        cold_diffu = self.args.config.get("COLD_DIFFU", False)
        avg_showers = std_showers = E_bins = None
        
        if cold_diffu:
            with h5.File(self.args.config["AVG_SHOWER_LOC"]) as f:
                avg_showers = torch.from_numpy(f["avg_showers"][()].astype(np.float32)).to(self.device)
                std_showers = torch.from_numpy(f["std_showers"][()].astype(np.float32)).to(self.device)
                E_bins = torch.from_numpy(f["E_bins"][()].astype(np.float32)).to(self.device)
                
        NN_embed = self._setup_nn_embed()
                
        return CaloDiffu(
            self.data_shape,
            config=self.args.config,
            training_obj=self.args.config.get("TRAINING_OBJ", "noise_pred"),
            NN_embed=NN_embed,
            nsteps=self.args.config["NSTEPS"],
            cold_diffu=cold_diffu,
            avg_showers=avg_showers,
            std_showers=std_showers,
            E_bins=E_bins
        ).to(self.device)

    def _setup_latent_diffusion_model(self):
        """Setup latent diffusion model including autoencoder"""
        # Load and setup autoencoder
        ae_checkpoint = torch.load(self.args.model_loc, map_location=self.device)
        self.autoencoder = CaloEnco(
            self.data_shape,
            config=self.args.config,
            training_obj="mean_pred",
            NN_embed=self._setup_nn_embed(),
            nsteps=self.args.config["NSTEPS"],
            cold_diffu=False,
            layer_sizes=self.args.layer_sizes
        ).to(self.device)
        
        if "model_state_dict" in ae_checkpoint:
            self.autoencoder.load_state_dict(ae_checkpoint["model_state_dict"])
        else:
            self.autoencoder.load_state_dict(ae_checkpoint)
            
        # Encode training data
        encoded_data = self._encode_training_data()
        
        # Calculate and save encoding statistics
        self.encoded_stats = {
            'mean': torch.mean(encoded_data).item(),
            'std': torch.std(encoded_data).item()
        }
        self._save_encoded_stats()
        
        # Normalize encoded data
        encoded_data = (encoded_data - self.encoded_stats['mean']) / self.encoded_stats['std']
        
        # Setup data loaders with encoded data
        encoded_dataset = torchdata.TensorDataset(torch_E_tensor, encoded_data)
        train_size = int(round(self.args.frac * len(encoded_dataset)))
        val_size = len(encoded_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(encoded_dataset, [train_size, val_size])
        
        self.train_loader = torchdata.DataLoader(train_dataset, batch_size=self.args.config["BATCH"], shuffle=True)
        self.val_loader = torchdata.DataLoader(val_dataset, batch_size=self.args.config["BATCH"], shuffle=True)
        
        # Setup latent diffusion model
        max_downsample = np.array(encoded_data.shape)[-3:].min() // 2
        
        return CaloDiffu(
            encoded_data.shape[1:],
            config=self.args.config,
            training_obj=self.args.config.get("TRAINING_OBJ", "noise_pred"),
            nsteps=self.args.config["NSTEPS"],
            max_downsample=max_downsample,
            is_latent=True
        ).to(self.device)

    def _encode_training_data(self):
        """Encode training data using autoencoder"""
        encoded_data = []
        logger.info("Encoding training data...")
        
        with torch.no_grad():
            for i, (E, data) in tqdm(enumerate(self.train_loader), unit="batch", total=len(self.train_loader)):
                E = E.to(self.device)
                data = data.to(self.device)
                enc = self.autoencoder.encode(data, E).detach().cpu().numpy()
                
                if i == 0:
                    encoded_data = enc
                else:
                    encoded_data = np.concatenate((encoded_data, enc))
                    
        return torch.tensor(encoded_data).to(self.device)

    def _save_encoded_stats(self):
        """Save encoding statistics to file"""
        with open(os.path.join(self.checkpoint_folder, "encoded_mean_std.txt"), "w") as f:
            f.write(f"encoded_mean={self.encoded_stats['mean']}\nencoded_std={self.encoded_stats['std']}")

    def _setup_nn_embed(self):
        """Setup neural network embeddings if needed"""
        if "NN" not in self.args.config.get("SHOWER_EMBED", ""):
            return None
            
        if self.args.config.get("DATASET_NUM", 2) == 1:
            binning_file = self.args.binning_file or "../CaloChallenge/code/binning_dataset_1_photons.xml"
            bins = XMLHandler.XMLHandler("photon", binning_file)
        else:
            binning_file = self.args.binning_file or "../CaloChallenge/code/binning_dataset_1_pions.xml"
            bins = XMLHandler.XMLHandler("pion", binning_file)
            
        return NNConverter(bins=bins).to(self.device)

    def setup_training(self):
        """Setup optimizer, scheduler and early stopping"""
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(self.args.config["LR"]))
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            factor=0.1,
            patience=15,
            verbose=True
        )
        self.early_stopper = EarlyStopper(
            patience=self.args.config["EARLYSTOP"],
            mode="diff",
            min_delta=1e-5
        )

    def save_checkpoint(self, val_metrics=None):
        """Save model checkpoint"""
        save_dict = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_epoch': self.best_epoch,
            'best_metrics': self.best_metrics,
            'early_stop_dict': self.early_stopper.__dict__
        }
        
        # Save latest checkpoint
        torch.save(save_dict, os.path.join(self.checkpoint_folder, "checkpoint.pth"))
        
        # Save best model if this is the best validation loss
        if val_metrics and val_metrics['loss'] < self.best_metrics['loss']:
            self.best_epoch = self.epoch
            self.best_metrics = val_metrics
            torch.save(save_dict, os.path.join(self.checkpoint_folder, "best_val.pth"))
            logger.info(f'New best validation loss! Saving model at epoch {self.epoch}')

    def load_checkpoint(self):
        """Load existing checkpoint"""
        checkpoint_path = os.path.join(self.checkpoint_folder, "checkpoint.pth")
        if not os.path.exists(checkpoint_path):
            return
            
        logger.info('Loading previous training checkpoint')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_epoch = checkpoint['best_epoch']
        self.best_metrics = checkpoint['best_metrics']
        self.early_stopper.__dict__ = checkpoint['early_stop_dict']

    def train(self):
        """Main training loop"""
        training_losses = np.zeros(self.args.config["MAXEPOCH"])
        val_losses = np.zeros(self.args.config["MAXEPOCH"])
        
        logger.info("Starting training...")
        
        for epoch in range(self.epoch, self.args.config["MAXEPOCH"] + 1):
            self.epoch = epoch
            logger.info(f"Beginning epoch {epoch}")
            
            # Training phase
            train_loss = self._train_epoch()
            training_losses[epoch-1] = train_loss
            logger.info(f"Training loss: {train_loss:.6f}")
            
            # Validation phase
            val_loss = self._validate_epoch()
            val_losses[epoch-1] = val_loss
            logger.info(f"Validation loss: {val_loss:.6f}")
            
            # Update learning rate and save checkpoint
            self.scheduler.step(torch.tensor([train_loss]))
            self.save_checkpoint({'loss': val_loss})
            
            # Save loss histories
            self._save_loss_histories(training_losses, val_losses)
            
            # Early stopping check
            if self.early_stopper.early_stop(val_loss - train_loss):
                logger.info("Early stopping triggered!")
                break
                
        logger.info("Training complete!")
        torch.save(self.model.state_dict(), os.path.join(self))