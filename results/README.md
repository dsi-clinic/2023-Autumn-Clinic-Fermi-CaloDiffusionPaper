# Results

In this folder, we will collect scripts and notebooks that allow us to share and reproduce the results we've made over the course of the quarter.

To keep track of each result, we need the following: 
 1. A short name to describe the result, like "optimal batch size for AE on dataset 3". 
 2. A script or notebook (possibly multiple) one can run to reproduce the results.
 3. Any hard-coded paths in the script or notebook. For example, if the result needs a pre-trained autoencoder's weights, that should be noted in this column.

| Result Name | Script or Notebook | Hard-coded paths | 
| --- | --- | --- |
| Interpolate to achieve fractional compression factor | sbatch/train_ae_4x64.sh will run train_ae_class, which utilizes a new class in ae_models.py. See FractionalResizeTrilinear in ae_models for more details on the interpolation method. Results for compress factor of 1.2 vs the original convolution/transposed convolution method with a compress factor of 2 are visualized in compression_factor1.2.png. | Set this line 'CYLINDRICAL': True, of the config file to 'RESIZE_METHOD' : 'cylindrical-frac-interpolate',. See config_dataset1_photon.json for more details, and ResizeMethod in ae_models for descriptions of the config options.|


## Make GIF

To make a GIF, run the `make_gif.sh` script. This will take all the PDFs in the output from running the plot.py script and convert them to a GIF. This allows us to share a single GIF that shows the results through the layers instead of having to share all the PDFs. You need to edit the `make_gif.sh` script to point to the correct directory where the PDFs are located and adjust the file name to match the pdf file names.

```bash
./make_gif.sh
```