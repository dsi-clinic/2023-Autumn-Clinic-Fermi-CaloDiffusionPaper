# pylint: disable=invalid-name
""" Main script to evaluate contributions to the Fast Calorimeter Challenge 2022

    input:
        - set of events in .hdf5 file format (same shape as training data)
    output:
        - metrics for evaluation (plots, classifier scores, etc.)

    usage:
        -i --input_file: Name and path of the input file to be evaluated.
        -r --reference_file: Name and path of the reference .hdf5 file. A .pkl file will be
                             created at the same location for faster subsequent evaluations.
        -m --mode: Which metric to look at. Choices are
                   'all': does all of the below (with low-level classifier).
                   'avg': plots the average shower of the whole dataset.
                   'avg-E': plots the average showers at different energy (ranges).
                   'hist-p': plots histograms of high-level features.
                   'hist-chi': computes the chi2 difference of the histograms.
                   'hist': plots histograms and computes chi2.
                   'cls-low': trains a classifier on low-level features (voxels).
                   'cls-low-normed': trains a classifier on normalized voxels.
                   'cls-high': trains a classifier on high-level features (same as histograms).
        -d --dataset: Which dataset the evaluation is for. Choices are
                      '1-photons', '1-pions', '2', '3'
           --output_dir: Folder in which the evaluation results (plots, scores) are saved.
           --save_mem: If included, data is moved to the GPU batch by batch instead of once.
                       This reduced the memory footprint a lot, especially for datasets 2 and 3.

           --no_cuda: if added, code will not run on GPU, even if available.
           --which_cuda: Which GPU to use if multiple are available.

    additional options for the classifier start with --cls_ and can be found below.
"""

import argparse
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.decomposition import PCA

import jetnet

import HighLevelFeatures as HLF

from evaluate_plotting_helper import *

plt_ext = "png"

torch.set_default_dtype(torch.float64)

#plt.rc('text', usetex=True)
#plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb}')
#plt.rc('font', family='serif')
#
#
#
#plt.rcParams.update({
#    "text.usetex": True,
#    "font.family": "sans-serif",
#    "font.sans-serif": ["Helvetica"]})
#

########## Parser Setup ##########

parser = argparse.ArgumentParser(description=('Evaluate calorimeter showers of the '+\
                                              'Fast Calorimeter Challenge 2022.'))

parser.add_argument('--input_file', '-i', help='Name of the input file to be evaluated.', 
                    choices = ["/net/projects/fermi-1/data/dataset_1/dataset_1_pions_1.hdf5",
"/net/projects/fermi-1/data/dataset_1/dataset_1_photons_1.hdf5",
"/net/projects/fermi-1/data/dataset_1/dataset_1_pions_2.hdf5",
"/net/projects/fermi-1/data/dataset_1/dataset_1_photons_2.hdf5",
"/net/projects/fermi-1/data/dataset_2/dataset_2_1.hdf5",
"/net/projects/fermi-1/data/dataset_2/dataset_2_2.hdf5",
"/net/projects/fermi-1/data/dataset_3/3.1/dataset_3_1.hdf5",
"/net/projects/fermi-1/data/dataset_3/3.2/dataset_3_2.hdf5",
"/net/projects/fermi-1/data/dataset_3/3.4/dataset_3_4.hdf5",
"/net/projects/fermi-1/data/dataset_3/3.3/dataset_3_3.hdf5"])
# parser.add_argument('--reference_file', '-r',
#                     help='Name and path of the .hdf5 file to be used as reference. '+\
#                     'A .pkl file is created at the same location '+\
#                     'in the first run for faster runtime in subsequent runs.')
parser.add_argument('--mode', '-m', default='all',
                    choices=['all', 'avg', 'avg-E', 'hist-p', 'hist-chi', 'hist',
                             'cls-low', 'cls-low-normed', 'cls-high', 'fpd', 'kpd'],
                    help=("What metric to evaluate: " +\
                          "'avg' plots the shower average;" +\
                          "'avg-E' plots the shower average for energy ranges;" +\
                          "'hist-p' plots the histograms;" +\
                          "'hist-chi' evaluates a chi2 of the histograms;" +\
                          "'fpd' measures the Frechet physics distance on the high-level features;" +\
                          "'kpd' measures the Kernel physics distance on the high-level features;" +\
                          "'hist' evaluates a chi2 of the histograms and plots them;" +\
                          "'cls-low' trains a classifier on the low-level feautures;" +\
                          "'cls-low-normed' trains a classifier on the low-level feautures" +\
                          " with calorimeter layers normalized to 1;" +\
                          "'cls-high' trains a classifier on the high-level features;" +\
                          "'all' does the full evaluation, ie all of the above" +\
                          " with low-level classifier."))
parser.add_argument('--dataset', '-d', choices=['1-photons', '1-pions', '2', '3'],
                    help='Which dataset is evaluated.')
parser.add_argument('--output_dir', default='evaluation_results/',
                    help='Where to store evaluation output files (plots and scores).')
parser.add_argument('--ratio', default=False, action = 'store_true',
                    help='Add ratio panel to plots')
#parser.add_argument('--source_dir', default='source/',
#                    help='Folder that contains (soft links to) files required for'+\
#                    ' comparative evaluations (high level features stored in .pkl or '+\
#                   'datasets prepared for classifier runs.).')


# classifier options

# not possible since train/test/val split is done differently each time
# to-do: save random-seed to file/read prior to split
#parser.add_argument('--cls_load', action='store_true', default=False,
#                    help='Whether or not load classifier from --output_dir')

parser.add_argument('--cls_n_layer', type=int, default=2,
                    help='Number of hidden layers in the classifier, default is 2.')

parser.add_argument('--cls_n_iters', type=int, default=1,
                    help='Repeat n times')
parser.add_argument('--cls_n_hidden', type=int, default='2048',
                    help='Hidden nodes per layer of the classifier, default is 64')
parser.add_argument('--cls_dropout_probability', type=float, default=0.2,
                    help='Dropout probability of the classifier, default is 0.2')

parser.add_argument('--cls_batch_size', type=int, default=200,
                    help='Classifier batch size, default is 200.')
parser.add_argument('--cls_n_epochs', type=int, default=50,
                    help='Number of epochs to train classifier, default is 100.')
parser.add_argument('--cls_lr', type=float, default=2e-4,
                    help='Learning rate of the classifier, default is 2e-4.')

# CUDA parameters
parser.add_argument('--no_cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--which_cuda', default=0, type=int,
                    help='Which cuda device to use')

parser.add_argument('--save_mem', action='store_true',
                    help='Data is moved to GPU batch by batch instead of once in total.')

########## Functions and Classes ##########


def check_file(given_file, arg, which=None):
    """ checks if the provided file has the expected structure based on the dataset """
    print("Checking if {} file has the correct form ...".format(
        which if which is not None else 'provided'))
    num_features = {'1-photons': 368, '1-pions': 533, '2': 6480, '3': 40500}[arg.dataset]
    num_events = given_file['incident_energies'].shape[0]
    assert given_file['showers'].shape[0] == num_events, \
        ("Number of energies provided does not match number of showers, {} != {}".format(
            num_events, given_file['showers'].shape[0]))
    assert given_file['showers'].shape[1] == num_features, \
        ("Showers have wrong shape, expected {}, got {}".format(
            num_features, given_file['showers'].shape[1]))

    print("Found {} events in the file.".format(num_events))
    print("Checking if {} file has the correct form: DONE \n".format(
        which if which is not None else 'provided'))

def extract_shower_and_energy(given_file, which, max_evt = None):
    """ reads .hdf5 file and returns samples and their energy """
    shower = given_file['showers'][:max_evt]
    energy = given_file['incident_energies'][:max_evt]
    return shower, energy

def load_reference(filename):
    """ Load existing pickle with high-level features for reference in plots """
    print("Loading file with high-level features.")
    with open(filename, 'rb') as file:
        hlf_ref = pickle.load(file)
    return hlf_ref

def save_reference(ref_hlf, fname):
    """ Saves high-level features class to file """
    print("Saving file with high-level features.")
    #filename = os.path.splitext(os.path.basename(ref_name))[0] + '.pkl'
    with open(fname, 'wb') as file:
        pickle.dump(ref_hlf, file)
    print("Saving file with high-level features DONE.")

def plot_histograms(hlf_class, reference_class, arg):
    """ plots histograms based with reference file as comparison """
    SetStyle()
    plot_Etot_Einc(hlf_class, reference_class, arg, ratio = arg.ratio)
    plot_E_layers(hlf_class, reference_class, arg, ratio = arg.ratio)
    plot_ECEtas(hlf_class, reference_class, arg, ratio = arg.ratio)
    plot_ECPhis(hlf_class, reference_class, arg, ratio = arg.ratio)
    plot_ECWidthEtas(hlf_class, reference_class, arg, ratio = arg.ratio)
    plot_ECWidthPhis(hlf_class, reference_class, arg, ratio = arg.ratio)
    if arg.dataset[0] == '1':
        plot_Etot_Einc_discrete(hlf_class, reference_class, arg)

########## Main ##########

if __name__ == '__main__':

    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    particle = {'1-photons': 'photon', '1-pions': 'pion',
                '2': 'electron', '3': 'electron'}[args.dataset]
    # minimal readout per voxel, ds1: from Michele, ds2/3: 0.5 keV / 0.033 scaling factor
    args.min_energy = {'1-photons': 10, '1-pions': 10,
                       '2': 0.5e-3/0.033, '3': 0.5e-3/0.033}[args.dataset]
    hlf = HLF.HighLevelFeatures(particle, filename='binning_dataset_{}.xml'.format(args.dataset.replace('-', '_')))
    source_file = h5py.File(args.input_file, 'r')
    shower, energy = extract_shower_and_energy(source_file, which='input')

    print(f'shape: {shower.shape}')
    print(shower, end="\n\n")


    components = 5
    old_shower = shower
    pca = PCA(n_components=components)
    shower = pca.fit_transform(shower)
    shower = pca.inverse_transform(shower)
    reconstruction_error = np.mean((shower - old_shower) ** 2)

    print(f'reconstruction error: {reconstruction_error}')
    # shower = pca.fit_transform(shower)

    # dummy = []
    # for val in range(len(shower)):
    #     d = [1]*shower[0]
    #     dummy.append(d)
    # dummy = np.array(dummy)

    title = f"CaloDiffusion {args.dataset}"
    print(args.dataset)
    hlf.DrawAverageShower(shower, filename=os.path.join(args.output_dir,
            'average_shower_{}_pca_n_comp_{}.{}'.format(args.dataset, components, plt_ext)), title=title)



