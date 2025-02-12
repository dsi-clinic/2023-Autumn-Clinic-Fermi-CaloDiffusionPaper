import torch
import h5py
import numpy as np
import jetnet
import json
from CaloEnco import CaloEnco

# Paths
model_dir = "/home/kwallace2/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/scripts/ae_models/dataset1_phot_AE_64_64_64_64_0.0004"
checkpoint_path = f"{model_dir}/best_val.pth"
config_path = f"{model_dir}/config_dataset1_photon.json"
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

# Xavier initialization
with torch.no_grad():
    if hasattr(model.convTrans, "convTrans"):
        if model.convTrans.convTrans.weight is not None:
            torch.nn.init.xavier_uniform_(model.convTrans.convTrans.weight)
        if model.convTrans.convTrans.bias is not None:
            model.convTrans.convTrans.bias.zero_()

print("Autoencoder model loaded successfully.")
model.eval()

# Load the Reference Data 
reference_hdf5 = "/net/projects/fermi-1/data/dataset_1/dataset_1_photons_1.hdf5"

with h5py.File(reference_hdf5, "r") as f:
    reference_showers = f["showers"][:]  # Shape: (121000, 368)

print(f"Reference dataset loaded. Shape: {reference_showers.shape}")

# Generate Autoencoder Outputs 
# Convert reference showers to PyTorch tensor and send to device
input_tensors = torch.tensor(reference_showers, dtype=torch.float32).to(device)
print(f"Input tensor shape: {input_tensors.shape}")

with torch.no_grad():
    reconstructed_showers = model(input_tensors)

reconstructed_showers = reconstructed_showers.cpu().numpy()

print(f"Autoencoder-generated showers shape: {reconstructed_showers.shape}")

# Compute FPD Metric Using JetNet 

fpd_val, fpd_err = jetnet.evaluation.fpd(
    real_features=reference_showers,  # Original dataset
    gen_features=reconstructed_showers  # Autoencoder output
)

print(f"FPD: {fpd_val:.4f} ± {fpd_err:.4f}")

results_path = f"{model_dir}/fpd_results.txt"
with open(results_path, "w") as f:
    f.write(f"FPD Metric: {fpd_val:.6f} ± {fpd_err:.6f}\n")

print(f"FPD evaluation completed. Results saved to {results_path}.")

'''
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
'''
