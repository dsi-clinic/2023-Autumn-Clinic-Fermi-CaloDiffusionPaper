import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
import h5py as h5
import torch
import torch.optim as optim
import torch.utils.data as torchdata
import sys
import os

from ae_models import *
from CaloEnco import *

from scripts.utils import *
from CaloChallenge.code.XMLHandler import *

####not currently changing binning file

def trim_file_path(cwd:str, num_back:int):
    '''
    '''
    split_path = cwd.split("/")
    trimmed_split_path = split_path[:-num_back]
    trimmed_path = "/".join(trimmed_split_path)

    return trimmed_path

if(torch.cuda.is_available()): device = torch.device('cuda')
else: device = torch.device('cpu')

parser = argparse.ArgumentParser()

parser.add_argument('--data_folder', default='/net/projects/fermi-1/doug/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/data/dataset_2/', help='Folder containing data and MC files')
parser.add_argument('-c', '--config', default='/net/projects/fermi-1/doug/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/configs/config_dataset2.json', help='Config file with training parameters')
parser.add_argument('--nevts', type=int,default=-1, help='Number of events to load')
parser.add_argument('--binning_file', type=str, default='/net/projects/fermi-1/doug/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/CaloChallenge/code/binning_dataset_2.xml')
parser.add_argument('--model_loc', default='/net/projects/fermi-1/doug/ae_models/dataset2_AE/downsample_2_8_hrs/final.pth', help='Location of model')
# parser.add_argument('--layer_sizes', type=str, default=None, help="Manual layer sizes input instead of from config file")

flags = parser.parse_args()

cwd = __file__
calo_challenge_dir = trim_file_path(cwd=cwd, num_back=3)
sys.path.append(calo_challenge_dir)
# print(calo_challenge_dir)

dataset_config = LoadJson(flags.config)

nholdout  = dataset_config.get('HOLDOUT', 0)
dataset_num = dataset_config.get('DATASET_NUM', 2)
shower_embed = dataset_config.get('SHOWER_EMBED', '')
orig_shape = ('orig' in shower_embed)
batch_size = dataset_config['BATCH']

##############################################################
def find_layer_sizes(model_loc):

    layer_sizes_info = ""
    model_loc = model_loc.split('/')
    for i,val in enumerate(model_loc):
        if '.pth' in val:
            layer_sizes_info = model_loc[i-1]
            break
    
    if layer_sizes_info.startswith('static'):
        layer_sizes = layer_sizes_info[len('static')+1:len('static')+1+11]
    else:
        layer_sizes = 'default'

    layer_sizes_enum_dict = {'default': 'default', 
                        '32_32_32_32': '4x32', 
                        '16_16_16_16': '4x16', 
                        '16_16_32_32': '2x16_2x32'}
    

    layer_sizes_layers_dict = {'default': {'raw_batch': (4500000, -1), 'encoded': (307200000, -1)}, 
                        '32_32_32_32': {'raw_batch': (44528000, -1), 'encoded': (108416000, -1)}, 
                        '16_16_16_16': {'raw_batch': (4500000, -1), 'encoded': (307200000, -1)}, 
                        '16_16_32_32': {'raw_batch': (4500000, -1), 'encoded': (307200000, -1)}}

    dict_info = [layer_sizes_enum_dict[layer_sizes], layer_sizes_layers_dict[layer_sizes]]
    if layer_sizes == 'default':
        return dict_info + [None] 
    return dict_info + [list(map(int, layer_sizes.split('_')))]

layer_sizes_name, layer_sizes_layers, layer_sizes = find_layer_sizes(flags.model_loc)
print(f'layer sizes: {layer_sizes}')
##############################################################



data = []
energies = []

# LOAD IN DATA
for i, dataset in enumerate(dataset_config['FILES']):
    data_,e_ = DataLoader(
        os.path.join(flags.data_folder,dataset),
        dataset_config['SHAPE_PAD'],
        emax = dataset_config['EMAX'],emin = dataset_config['EMIN'],
        nevts = flags.nevts,
        max_deposit=dataset_config['MAXDEP'], #noise can generate more deposited energy than generated
        logE=dataset_config['logE'],
        showerMap = dataset_config['SHOWERMAP'],

        nholdout = nholdout if (i == len(dataset_config['FILES']) -1 ) else 0,
        dataset_num  = dataset_num,
        orig_shape = orig_shape,
    )
    

    if(i ==0): 
        data = data_
        energies = e_
    else:
        data = np.concatenate((data, data_))
        energies = np.concatenate((energies, e_))

# DATA RE-PROCESSING
NN_embed = None
if('NN' in shower_embed):
    if(dataset_num == 1):
        if flags.binning_file is None:
            flags.binning_file = "../CaloChallenge/code/binning_dataset_1_photons.xml"
        bins = XMLHandler("photon", flags.binning_file)
    else: 
        if flags.binning_file is None:
            flags.binning_file = "../CaloChallenge/code/binning_dataset_1_pions.xml"
        bins = XMLHandler("pion", flags.binning_file)

    NN_embed = NNConverter(bins = bins).to(device = device)
    

dshape = dataset_config['SHAPE_PAD']
energies = np.reshape(energies,(-1))    
if(not orig_shape): data = np.reshape(data,dshape)
else: data = np.reshape(data, (len(data), -1))

num_data = data.shape[0]
print("Data Shape " + str(data.shape))
data_size = data.shape[0]
#print("Pre-processed shower mean %.2f std dev %.2f" % (np.mean(data), np.std(data)))
torch_data_tensor = torch.from_numpy(data)
torch_E_tensor = torch.from_numpy(energies)

# SAVE DATA AND E as npz


cwd = os.getcwd()
# os.mkdir(cwd+'/data_for_latent')
np.savez(cwd+f'/data_for_latent/raw_dataset_{layer_sizes_name}.npz', data=torch_data_tensor, E=torch_E_tensor) #Save File 

torch_dataset  = torchdata.TensorDataset(torch_E_tensor, torch_data_tensor)
loader_encode = torchdata.DataLoader(torch_dataset, batch_size = batch_size, shuffle = False)

del data

shape = dataset_config['SHAPE_PAD'][1:] if (not orig_shape) else dataset_config['SHAPE_ORIG'][1:]
checkpoint = torch.load(flags.model_loc, map_location = device)

AE = CaloEnco(shape, config=dataset_config, training_obj='mean_pred', NN_embed=NN_embed, 
                        nsteps=dataset_config['NSTEPS'], cold_diffu=False, avg_showers=None, 
                        std_showers=None, E_bins=None, layer_sizes=layer_sizes).to(device = device)



if('model_state_dict' in checkpoint.keys()): AE.load_state_dict(checkpoint['model_state_dict'])
elif(len(checkpoint.keys()) > 1): AE.load_state_dict(checkpoint)

## ^^ THIS LINE IS THE PROBLEM

with torch.no_grad():
    print("Encoding Data...")
    encoded_data = [] #empty torch tensor
    for i, (E,data) in tqdm(enumerate(loader_encode, 0), unit="batch", total=len(loader_encode)):
        data = data.to(device = device)
        E = E.to(device = device)
        encoded_batch = AE.encode(data, E).detach().to(device = 'cpu')
        encoded_data.append(encoded_batch) # Appends torch tensors to a list

print("Data Successfully Encoded")
encoded_data = torch.cat(encoded_data)
print(f"Encoded Data Shape: {encoded_data.shape}")

np.savez(cwd+f'/data_for_latent/encoded_dataset_{layer_sizes_name}.npz', encoded_data) #Save File to 


