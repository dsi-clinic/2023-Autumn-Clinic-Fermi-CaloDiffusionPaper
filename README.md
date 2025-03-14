# Code Cleanup Grading
Due to the complex nature of our project and git repo not coming implemented with `ruff`, we've received an exemption
from Tim on the final technical cleanup grading. Please select from the following files to grade on code readability and
formatting, these are the files we worked on specifically in this quarter and should all pass `ruff check` and `ruff
format`.  
1. `/scripts/autoencoder/ae_models.py`  
2. `/scripts/train_diffu_class.py`
3. `/scripts/autoencoder/train_ae_class.py`
4. `/scripts/ae_eval_fpd.py`

# DSI Clinic Initial Week

You will be meeting with the mentor during the 2nd week. Before meeting with them please do the following:

1. Watch the video from [last year](https://drive.google.com/file/d/1eSQwKvNqEGQ-Ux75Su4xD01Tdht1RYxG/view) to become familiar with the project. 
2. If you are not familiar with PyTorch, please begin going through this basic tutorial. Make sure to get through _at least_ the sections entitled (1) Tensors (2) Datasets & DataLoaders and (3) Transforms [here](https://pytorch.org/tutorials/beginner/basics/intro.html).
3. Watch the recording here (look for the recording link on _Applications: Generative Models I_): [video link here](https://indico.physics.lbl.gov/event/2850/timetable/?view=standard#10-applications-generative-mod)


After the first week some good places to start are:
1. This notebook has a good tutorial: https://github.com/ml4fp/2024-lbnl/blob/main/generative/Gen_Tutorial_PyTorch.ipynb
2. Then they could try this tutorial about diffusion models, which is the main idea we are working with : https://huggingface.co/blog/annotated-diffusion
3. Then our paper can give a decent idea of this application : https://arxiv.org/abs/2308.03876


# CaloDiffusion official repository

Repo for diffusion based calorimeter generation. Now includes pipelines to train and evaluate autoencoders and latent diffusion models for calorimeter generation.
  
Pytorch v1.9 was used to implement all models.


# Data

Results are presented using the [Fast Calorimeter Data Challenge dataset](https://calochallenge.github.io/homepage/) and are available for download on zenodo:

*  [Dataset 1](https://zenodo.org/record/6368338)

*  [Dataset 2](https://zenodo.org/record/6366271)

*  [Dataset 3](https://zenodo.org/record/6366324)

# Train an autoencoder with

```bash
python3 scripts/autoencoder/train_ae.py \
	--data_folder [PATH TO FOLDER CONTAINING DATASETS] \
	--config [PATH TO CONFIG FILE FOR DATASET AND PARTICLE] \
	--binning_file [PATH TO RELEVANT BINNING FILE IN CaloChallenge DIRECTORY] \
	--layer_sizes [LIST OF LAYER SIZES WHICH CONTROL AUTOENCODER COMPRESSION]
```

* Additional flags exist to control training parameters (learning rate, scheduler, early stop conditions, etc.)
* If you need to continue pick up training, can use the `--load` flag to load in checkpoint and resume training
* Autoencoders compress input data by a *compression factor* denoted as $c$. $c$ = original dimensionality / compressed dimensionality.
* Higher compression means faster latent diffusion models but there is an accuracy tradeoff
* Understand the compression rates and how they modify data size exactly [here](https://github.com/dsi-clinic/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/blob/main/scripts/autoencoder/ae_hyper_explore/compression_rates.ipynb)

Compression factor $c$ is controlled by the `--layer_sizes` flag, which is a list of the number of input channels into the autoencoder ResNet blocks. The formula to determine compression factor from a given layer sizes input is seen below. 

**Compression Factor from Layer Sizes Formula**\
$$\frac{2^{3*(|\ell|-2)}}{\ell_{-1}/4}$$

* $\ell$ refers to the list passed in the `--layer_sizes` flag
* $|\ell|$ is the length of layer sizes list
* $\ell_{-1}$ is the last item in the layer sizes list
  
# Evaluate an autoencoder  

```bash
python3 scripts/autoencoder/ae_eval.py \
	--data_folder [PATH TO FOLDER CONTAINING DATASETS] \
	--config [PATH TO CONFIG FILE FOR DATASET AND PARTICLE] \
	--binning_file [PATH TO RELEVANT BINNING FILE IN CaloChallenge DIRECTORY]
	--model_loc [PATH TO AUTOENCODER SAVED WEIGHTS] \
	--layer_sizes [LIST OF LAYER SIZES FOR TRAINED AUTOENCODER] \
	--sample
python3 scripts/autoencoder/ae_eval.py \
	--data_folder [PATH TO FOLDER CONTAINING DATASETS] \
	--config [PATH TO CONFIG FILE FOR DATASET AND PARTICLE] \
	--binning_file [PATH TO RELEVANT BINNING FILE IN CaloChallenge DIRECTORY]
	--model_loc [PATH TO AUTOENCODER SAVED WEIGHTS] \
	--layer_sizes [LIST OF LAYER SIZES FOR TRAINED AUTOENCODER] \
	--sample
```
  * The first call encodes and reconstructs using the autoencoder then saves the output dataset
  * The second call compares the reconstructed dataset to the original and outputs comparison plots
  * Can also use `evaluate.py`, directions for which are below

There is also an additional evaluation method that will create a plot comparing the performance of many auto encoders at once. The resulting scatterplot shows the compression factor on the x-axis and FPD on the y-axis. 

 ```bash
python3 scripts/autoencoder/
```

To perform PCA as a form of dimensionality analysis on the encoded data and compare it to the raw data, there are various scripts that can be run:

1) scripts/autoencoder/evaluate_latent.py contains the script that can be run to extract the raw data and encoded data relative to a particular autoencoder. Specify the autoencoder model file path with the model_loc flag. NOTE: After running this file once, you will have the full raw dataset and can comment out the dataset line in order to not re-load all the raw data. This data will go into the autoencoder/data_for_latent folder

```bash
python3 scripts/autoencoder/evaluate_latent.py \
	--data_folder [PATH TO FOLDER CONTAINING RAW DATASET] \
	--config [PATH TO CONFIG FILE FOR RAW DATASET AND PARTICLE] \
	--binning_file [PATH TO RELEVANT BINNING FILE IN CaloChallenge DIRECTORY]
	--model_loc [PATH TO AUTOENCODER SAVED WEIGHTS] \

```

2) You can then use these files to run the PCA script immediately without having to specify new file paths or data, it'll all be in the correct file after running the above evaluate_latent.py script. 
 
 ```bash
python3 scripts/autoencoder/evaluate_latent.py
```

3) Visualizations will be created in the autoencoder/pca_plots folder. 


To compare the evalutations of multiple autoencoder models with different layer sizes and/or learning rates, you can run scripts/autoencoder/hyper_compare_eval.py:

```bash
python3 scripts/autoencoder/hyper_compare_eval.py \
	--model_eval_folder [PATH TO CaloChalenge/code/evaluate.py OUTPUT RESULTS FOR ALL AE MODELS] \
	--plot_output_folder [PATH TO PUT OUTPUT (WILL CREATE IF NONEXISTENT)]
```

This will create several plots and a file with models sorted by their respective Frechet Physics Distance (FPD) in the passed plot_output_folder. Note that this requires evaluation results for all and only models to be compared to be stored in a single folder (passed in model_eval_folder). Each model's eval result folder must also have the format static_<layer_sizes>e<epoch>lr<learning_rate> with layer size numbers separated by underscores, e.g. static_16_16_16_16_32e20lr0.004 in order for the program to identify the model hyperparameters.


# Run the training scripts
 
### To train classic diffusion model
```bash
cd scripts
python train_diffu.py --data_folder DIRECTORY_OF_INPUTS --config CONFIG
```

* Example configs used to produce the results in the paper are in `configs` folder

* Trained models used to produce the results in the paper are in the `trained_models` folder

### To train latent diffusion model with pre-trained autoencoder
```bash
python3 scripts/train_diffu.py \
	--data_folder [PATH TO FOLDER CONTAINING DATASETS] \
	--config [PATH TO CONFIG FILE FOR DATASET AND PARTICLE] \
	--binning_file [PATH TO RELEVANT BINNING FILE IN CaloChallenge DIRECTORY]
	--model_loc [PATH TO AUTOENCODER SAVED WEIGHTS] \
	--layer_sizes [LIST OF LAYER SIZES FOR TRAINED AUTOENCODER] \
	--model Latent_Diffu
```
# Sampling with the learned model

The `plot.py` both samples from the models (if the `--sample` flag is included) and creates the summary plots.

```bash
python plot.py --data_folder DIRECTORY_OF_INPUTS \
	-g GENERATED_SHOWERS.h5 \
	--nevts N --sample --config CONFIG \
	--sample_steps N_DIFFUSION_STEPS \
	--sample_offset N_OFFSET
```

To sample from a latent diffusion model use the below call. Use the `--sample` flag to generate and save new showers.
```bash
python3 scripts/plot.py \
	--data_folder [PATH TO FOLDER CONTAINING DATASETS] \
	--config [PATH TO CONFIG FILE FOR DATASET AND PARTICLE] \
	--binning_file [PATH TO RELEVANT BINNING FILE IN CaloChallenge DIRECTORY] \
	--ae_loc [PATH TO TRAINED AUTOENCODER WEIGHTS] \
	--model_loc [PATH TO TRAINED LATENT DIFFUSION WEIGHTS] \
	--model Latent_Diffu \
	--layer_sizes [LIST OF LAYER SIZES FOR TRAINED AUTOENCODER] \
	--encoded_mean_std_loc [LOCATION OF encoded_mean_std.txt GENERATED BY train_diffu.py]
	--sample
```
There are additional options in `plot.py`.

400 diffusion steps were used for the baseline results in the paper.

A sampling offset of 2 was used for datasets 1 and 2 (necessary to prevent instabilities).

An example commands to generate showers for dataset 1 photons would be as follows:

```bash

python plot.py --config ../configs/config_dataset1_photon.json --model_loc ../trained_models/dataset1_photon.pth --sample --sample_steps 400 --sample_offset 2 --nevts 1000 -g test_ds1.h5

```

# Evaluate diffusion models and generate plots shown in paper

```bash
cd CaloChallenge/code
python3 evaluate.py \
	--input_file [PATH TO GENERATED SHOWER] \
	--reference_file [PATH TO GEANT4 SHOWERS] \
	--dataset [DATASET FROM WHICH SHOWERS ARE GENERATE]
```
* `--input_file` is the file generated by running `plot.py` with the `--sample` flag
* `--reference_file` is the evaluation dataset being pull from the config file passed into `plot.py` 
* One of either: '1-photons', '1-pions', '2', '3'

Some of the quantitative metrics and plotting for dataset 1 are done based on the [CaloChallenge evaluation code](https://github.com/OzAmram/CaloChallenge)

The evaluation for the CaloChallenge code proceeds can be performed as:

```
python evaluate.py -i GENERATED_DATASET.h5 -r REFERENCE_DATASET.h5 -d DATASET_NUMBER --output_dir DIRECTORY_FOR_OUTPUTS/ -m all --ratio

```

The `-m all` flag runs all the evaluations of the CaloChallenge (plots, chi2 metric, classifier metric, KPD/FPD metric).

See `evaluate.py` for more options

# Generating AutoEncoder Evaluation Graphs:
Use the notebooks found in `/scripts/autoencoder/ae_hyper_explore/` to generate graphs of the autoencoder results.  
`LR_analysis.ipynb` compares autoencoder models with different learning rates, hard coded to 4e-3, -4, and -5.  
`dataset1_analysis.ipynb` compares autoencoder models specifically trained on dataset1.  
`dataset2_analysis.ipynb` compares autoencoder models specifically trained on dataset2.  
Relative paths are in reference to folder in which models are stored. Please see `ae_layers_funcs.py` for more details
on how code works for all notebooks in terms of loading data and graphing such respectively.

# Student Contributors This Quarter:

Emily Tong (emilytong@uchicago.edu)

Aaron Zhang

Isaac Harlem (isaacharlem@uchicago.edu)
                                        
Simon Katz (simonkatz@uchicago.edu)

Kevin Lin (klin665@uchicago.edu)

Kristen Wallace (kwallace2@uchicago.edu)

# Student Contributors From Previous Quarters:

Victor Brown

Josh Li

Keegan Ballantyne

Carina Kane

Doug Williams

Victor Brown

Josh Li

Grey Singh
