import numpy as np
import copy

# import time
import torch
import torch.nn as nn

# from torch.autograd import Variable
# from torchinfo import summary
import sys

# import os

try:
    from ae_models import cosine_beta_schedule, FCN, CondAE
except:
    from autoencoder.ae_models import cosine_beta_schedule, FCN, CondAE


def trim_file_path(cwd: str, num_back: int):
    """ """
    split_path = cwd.split("/")
    trimmed_split_path = split_path[:-num_back]
    trimmed_path = "/".join(trimmed_split_path)

    return trimmed_path


cwd = __file__
scripts_dir = trim_file_path(cwd=cwd, num_back=3)
sys.path.append(scripts_dir)
from scripts.utils import create_phi_image, create_R_Z_image


class CaloEnco(nn.Module):
    """
    Class that contains functions necessary for training autoencoder including computing loss, encoding, and decoding.

    Parameters:
    - data_shape (torch.tensor): shape of data to be encoded
    - config (dict): parsed config file
    - training_obj (str): shorthand for training objective, should be 'mean_pred' for autoencoder training
    - nsteps (int): number of time steps used in training the diffusion model to be used
    - cold_diffu (bool): indicator for whether generating using average showers
    - E_bins (torch.tensor): contains information about binning of sensors, should remain None for typical autoencoder training
    - avg_showers (torch.tensor): contains average showers energy over time for cold diffusion, should be None for typical autoencoder training
    - std_showers (torch.tensors): standardized showers to be indexed in average shower lookup, should be None for typical autoencoder training
    - NN_embed (PyTorch model): NN_embed model to convert irregular dataset 1 binning to regular binning
    - resnet_set (list): alternate method to control downsampling, allows removal of resnet + downsample block combinations from UNet architecture
    - layer_sizes (list): primary method to control downsampling, list of input channel dimensions which is zipped for UNet ResNet blocks

    """

    def __init__(
        self,
        data_shape,
        config=None,
        training_obj="mean_pred",
        nsteps=400,
        cold_diffu=False,
        E_bins=None,
        avg_showers=None,
        std_showers=None,
        NN_embed=None,
        resnet_set=[0, 1, 2],
        layer_sizes=None,
    ):
        super(CaloEnco, self).__init__()
        self._data_shape = data_shape
        self.nvoxels = np.prod(self._data_shape)
        self.config = config
        self._num_embed = self.config["EMBED"]
        self.num_heads = 1
        self.nsteps = nsteps
        self.cold_diffu = cold_diffu
        self.E_bins = E_bins
        self.avg_showers = avg_showers
        self.std_showers = std_showers
        self.training_obj = training_obj
        self.shower_embed = self.config.get("SHOWER_EMBED", "")
        self.fully_connected = "FCN" in self.shower_embed
        self.NN_embed = NN_embed
        self.resnet_set = resnet_set

        supported = ["mean_pred"]
        is_obj = [s in self.training_obj for s in supported]
        if not any(is_obj):
            print("Training objective %s not supported!" % self.training_obj)
            exit(1)

        if config is None:
            raise ValueError("Config file not given")

        self.verbose = 1

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # Linear schedule
        schedd = config.get("NOISE_SCHED", "linear")
        self.discrete_time = True

        if "linear" in schedd:
            self.betas = torch.linspace(
                self.beta_start, self.beta_end, self.nsteps
            )
        elif "cosine" in schedd:
            self.betas = cosine_beta_schedule(self.nsteps)
        elif "log" in schedd:
            self.discrete_time = False
            self.P_mean = -1.5
            self.P_std = 1.5
        else:
            print("Invalid NOISE_SCHEDD param %s" % schedd)
            exit(1)

        if self.discrete_time:
            # Precompute useful quantities for training
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

            # Shift all elements over by inserting unit value in first place
            alphas_cumprod_prev = torch.nn.functional.pad(
                self.alphas_cumprod[:-1], (1, 0), value=1.0
            )

            self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
            self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
            self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
                1.0 - self.alphas_cumprod
            )

            self.posterior_variance = (
                self.betas
                * (1.0 - alphas_cumprod_prev)
                / (1.0 - self.alphas_cumprod)
            )

        self.time_embed = config.get("TIME_EMBED", "sin")
        self.E_embed = config.get("COND_EMBED", "sin")
        cond_dim = config["COND_SIZE_UNET"]
        if layer_sizes is None:
            layer_sizes = config["LAYER_SIZE_UNET"]
        block_attn = config.get("BLOCK_ATTN", False)
        mid_attn = config.get("MID_ATTN", False)
        compress_Z = config.get("COMPRESS_Z", False)

        if self.fully_connected:
            # Fully connected network architecture
            self.model = FCN(
                cond_dim=cond_dim,
                dim_in=config["SHAPE_ORIG"][1],
                num_layers=config["NUM_LAYERS_LINEAR"],
                cond_embed=(self.E_embed == "sin"),
                time_embed=(self.time_embed == "sin"),
            )

            self.R_Z_inputs = False

            # summary_shape = [[1, config["SHAPE_ORIG"][1]], [1], [1]]

        else:
            RZ_shape = config["SHAPE_PAD"][1:]

            self.R_Z_inputs = config.get("R_Z_INPUT", False)
            self.phi_inputs = config.get("PHI_INPUT", False)

            in_channels = 1

            self.R_image, self.Z_image = create_R_Z_image(
                device, scaled=True, shape=RZ_shape
            )
            self.phi_image = create_phi_image(device, shape=RZ_shape)

            if self.R_Z_inputs:
                in_channels = 3

            if self.phi_inputs:
                in_channels += 1

            calo_summary_shape = list(copy.copy(RZ_shape))
            calo_summary_shape.insert(0, 1)
            calo_summary_shape[1] = in_channels

            calo_summary_shape[0] = 1
            # summary_shape = [calo_summary_shape, [1], [1]]

            self.model = CondAE(
                cond_dim=cond_dim,
                out_dim=1,
                channels=in_channels,
                layer_sizes=layer_sizes,
                block_attn=block_attn,
                mid_attn=mid_attn,
                cylindrical=config.get("CYLINDRICAL", False),
                compress_Z=compress_Z,
                data_shape=calo_summary_shape,
                cond_embed=(self.E_embed == "sin"),
                time_embed=False,
                resnet_set=self.resnet_set,
                compress=config.get("COMPRESS_FACTOR", 2),
            )  # Removed time embeddings to match sizes

    # Wrapper for backwards compatability
    def load_state_dict(self, d):
        return super().load_state_dict(d)

    def add_RZPhi(self, x):
        cats = [x]
        if self.R_Z_inputs:

            batch_R_image = self.R_image.repeat([x.shape[0], 1, 1, 1, 1]).to(
                device=x.device
            )
            batch_Z_image = self.Z_image.repeat([x.shape[0], 1, 1, 1, 1]).to(
                device=x.device
            )

            cats += [batch_R_image, batch_Z_image]
        if self.phi_inputs:
            batch_phi_image = self.phi_image.repeat(
                [x.shape[0], 1, 1, 1, 1]
            ).to(device=x.device)

            cats += [batch_phi_image]

        if len(cats) > 1:
            return torch.cat(cats, axis=1)
        else:
            return x

    def lookup_avg_std_shower(self, inputEs):
        idxs = (
            torch.bucketize(inputEs, self.E_bins) - 1
        )  # NP indexes bins starting at 1
        return self.avg_showers[idxs], self.std_showers[idxs]

    def compute_loss(
        self,
        data,
        energy,
        t=None,
        loss_type="mse",
        rnd_normal=None,
        energy_loss_scale=1e-2,
    ):

        if self.discrete_time:
            if t is None:
                t = torch.randint(
                    0, self.nsteps, (data.size()[0],), device=data.device
                ).long()
            sigma = None
            # sigma2 = (
            #     extract(self.sqrt_one_minus_alphas_cumprod, t, data.shape) ** 2
            # )
        else:
            if rnd_normal is None:
                rnd_normal = torch.randn((data.size()[0],), device=data.device)
            sigma = (rnd_normal * self.P_std + self.P_mean).exp()
            # sigma2 = sigma**2

        t_emb = self.do_time_embed(t, self.time_embed, sigma)

        pred = self.pred(data, energy, t_emb)

        # weight = 1.0

        if "mean_pred" in self.training_obj:
            target = data
            # weight = 1.0 / sigma2
            x0_pred = pred

        if loss_type == "mse":
            loss = torch.nn.functional.mse_loss(target, pred)
        else:
            raise NotImplementedError()

        if "energy" in self.training_obj:
            # Sum total energy
            dims = [i for i in range(1, len(data.shape))]
            tot_energy_pred = torch.sum(x0_pred, dim=dims)
            tot_energy_data = torch.sum(data, dim=dims)
            loss_en = (
                energy_loss_scale
                * torch.nn.functional.mse_loss(tot_energy_data, tot_energy_pred)
                / self.nvoxels
            )
            loss += loss_en

        return loss

    def do_time_embed(
        self,
        t=None,
        embed_type="identity",
        sigma=None,
    ):
        if self.discrete_time:
            if sigma is None:
                # Identify tensor device so we can match index tensor t
                cumprod_device = self.sqrt_one_minus_alphas_cumprod.device
                # Move index tensor t to indexed tensor device before operation
                sigma = self.sqrt_one_minus_alphas_cumprod[t.to(cumprod_device)]
            if embed_type == "identity" or embed_type == "sin":
                return t
            if embed_type == "scaled":
                return t / self.nsteps
            if embed_type == "sigma":
                return sigma.to(t.device)
            if embed_type == "log":
                return 0.5 * torch.log(sigma).to(t.device)
        else:
            if embed_type == "log":
                return 0.5 * torch.log(sigma).to(t.device)
            else:
                return sigma

    def pred(self, x, E, t_emb):

        if self.NN_embed is not None:
            x = self.NN_embed.enc(x).to(x.device)
        out = self.model(self.add_RZPhi(x), E, t_emb)
        if self.NN_embed is not None:
            out = self.NN_embed.dec(out).to(x.device)
        return out

    def encode(self, x, E):
        """
        Class function that performs only the encoding step in the conditional u-net to transform original data into a lower
        dimensional space to generated a latent space

        Parameters:
        - x (torch.tensor): data to be encoded
        - E (torch.tensor): energies

        Returns:
        - original data that's dimensionally reduced into encoded shape (latent space)
        """

        if self.NN_embed is not None:
            x = self.NN_embed.enc(x).to(x.device)
        out = self.model.encode(self.add_RZPhi(x), E)

        return out

    def decode(self, x, E):
        """
        Class function that performs only the decoding step in the conditional u-net to transform encoded data into its original dimension size
        or otherwise transforming latent space shape into original shape

        Parameters:
        - x (torch.tensor): data to be encoded
        - E (torch.tensor): energies

        Returns:
        - encoded data transformed back into its original shape
        """

        out = self.model.decode(x, E)
        if self.NN_embed is not None:
            out = self.NN_embed.dec(out).to(x.device)

        return out
