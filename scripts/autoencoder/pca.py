from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import gaussian_kde
import argparse

import sys
import numpy as np
import os 
from datetime import datetime
from datetime import date


if __name__ == "__main__" :

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
        

        layer_sizes_layers_dict = {'default': {'raw_batch': (4500000, -1), 'encoded': (3200000, -1)}, 
                            '32_32_32_32': {'raw_batch': (44528000, -1), 'encoded': (108416000, -1)}, 
                            '16_16_16_16': {'raw_batch': (4500000, -1), 'encoded': (307200000, -1)}, 
                            '16_16_32_32': {'raw_batch': (4500000, -1), 'encoded': (307200000, -1)}}

        dict_info = [layer_sizes_enum_dict[layer_sizes], layer_sizes_layers_dict[layer_sizes]]
        if layer_sizes == 'default':
            return dict_info + [None] 
        return dict_info + [list(map(int, layer_sizes.split('_')))]

    ##############################################################

    today = date.today()
    month_day = today.strftime("%m-%d-")
    now = datetime.now()
    dt_string = now.strftime("%H:%M")


    parser = argparse.ArgumentParser()

    parser.add_argument('--data_folder', default='/net/projects/fermi-1/doug/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/data/dataset_2/', help='Folder containing data and MC files')
    parser.add_argument('-c', '--config', default='/net/projects/fermi-1/doug/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/configs/config_dataset2.json', help='Config file with training parameters')
    parser.add_argument('--nevts', type=int,default=-1, help='Number of events to load')
    parser.add_argument('--binning_file', type=str, default='/net/projects/fermi-1/doug/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/CaloChallenge/code/binning_dataset_2.xml')
    parser.add_argument('--model_loc', default='/net/projects/fermi-1/doug/ae_models/dataset2_AE/downsample_2_8_hrs/final.pth', help='Location of model')

    flags = parser.parse_args()

    layer_sizes_name, layer_sizes_layers, layer_sizes = find_layer_sizes(flags.model_loc)
    
    print(f'layer sizes: {layer_sizes}')
    # other = True

    figure_directory = f'./autoencoder/pca_plots/victor_plots/'+month_day+dt_string+f'-{layer_sizes_name}/'
    os.mkdir(figure_directory)


    # Pull Raw and Encoded Data From Files
    pull_path = f'./../data_for_latent/raw_dataset_victor1_{layer_sizes_name}.npz'
    pull_path2 = f'./../data_for_latent/encoded_dataset_victor1_{layer_sizes_name}.npz'

    print(f'pull_path : {pull_path}')

    raw_data_full = np.load(pull_path)
    encoded_data = np.load(pull_path2)


    encoded_data = encoded_data['arr_0']

    raw_data = raw_data_full['data'] #(100000, 1, 45, 16, 9)
    raw_E = raw_data_full['E']


    #RESHAPING IS THE ERROR

    # Reshape Batches Raw Data

    raw_sizes, encoded_sizes = layer_sizes_layers['raw_batch'], layer_sizes_layers['encoded']


    print(f'raw sizes: {raw_sizes}')
    print(f'encoded sizes: {encoded_sizes}')

    batch_reshape = raw_data.reshape(len(raw_data), -1) #(100000, 6480) #4500000
    batch_reshape_stack = raw_data.reshape(raw_sizes[0], raw_sizes[1]) #4500000, 

    
    # if other:
    #     batch_reshape_stack = raw_data.reshape(44528000, -1) #4500000, 
    # else:
    #     batch_reshape_stack = raw_data.reshape(4500000, -1)


    # ONLY WORKS FOR THE DEFAULT BC SUPER HARD CODED?

    # batch_reshape_radial = np.transpose(raw_data, (0, 1, 3, 2, 4))  #(100000, 1, 16, 45, 9) ##############ERROR HERE
    # batch_reshape_radial = batch_reshape_radial.reshape(1600000, -1)
  
    # batch_reshape_angular = np.transpose(raw_data, (0, 1, 4, 2, 3)) #(100000, 1, 9, 45, 16)
    # batch_reshape_angular = batch_reshape_angular.reshape(900000, -1) #(900000, 45, 16)
 
    # Reshape Batches Encoded Data
    #encoded_data.shape = (3200000, 12, 4, 2)


    # if other:
    #     encoded_data = encoded_data.reshape(108416000, -1) 
    # else:
    #     encoded_data = encoded_data.reshape(307200000, -1) 

    encoded_data = encoded_data.reshape(encoded_sizes[0], encoded_sizes[1]) #3200000

    # Sample of Raw and Encoded Data
    sample_raw_data = batch_reshape[np.random.choice(batch_reshape.shape[0], 5000, replace=False)]
    sample_encoded_data = encoded_data[np.random.choice(encoded_data.shape[0], 5000, replace=False)]

    # sample_raw_data_radial = batch_reshape_radial[np.random.choice(batch_reshape_radial.shape[0], 5000, replace=False)]
    sample_raw_data_stack = batch_reshape_stack[np.random.choice(batch_reshape_stack.shape[0], 5000, replace=False)]


############################
    #PCA Raw Data
    X = sample_raw_data 
    pca = PCA(n_components = 0.95)

    Xt = pca.fit_transform(X)

        #Plot 1st and 2nd PC againt each other
    x = Xt[:,0]
    y = Xt[:,1]
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    plt.figure()
    plot = plt.scatter(x, y, c=z)
    plt.savefig(figure_directory+'raw_pca_plot.png')
    plt.close() 

        #Plot Variance Explained
    features = range(len(pca.explained_variance_))  
    print("Number of principal components to explain 95 percent of data (RAW)=", len(features))
    plt.figure()
    total_variance = np.sum(pca.explained_variance_)
    explained_variance = (pca.explained_variance_ / total_variance) * 100
    plot2 = plt.bar(features[:10], explained_variance[:10])
    plt.savefig(figure_directory+'raw_pca_variance.png')
    plt.close() 

    # Making Y-Axis Log Scale
    plt.figure()
    plt.yscale('log')
    plot3 = plt.bar(features, explained_variance)
    plt.savefig(figure_directory+'raw_pca_variance_log.png')
    plt.close()


        #Same as above but Scaled
    pca = PCA()
    pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
    Xt = pipe.fit_transform(X)
    x1 = Xt[:,0]
    y1 = Xt[:,1]
    z1 = Xt[:,2]
    xy1 = np.vstack([x1,y1])
    z1 = gaussian_kde(xy1)(xy1)
    plt.figure()
    plot = plt.scatter(x1, y1, c=z1)
    plt.savefig(figure_directory+'raw_scaled_pca_plot.png')

    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('white')
   
    ax.scatter(x1, y1, z1,  s=40)
    ax.set_xlabel("First PC", fontsize=14)
    ax.set_ylabel("Second PC", fontsize=14)
    ax.set_zlabel("Third PC", fontsize=14)
    plt.title("3D Raw PCA Plot", fontsize=16)
    plt.savefig(figure_directory+'raw_3d_pca_plot.png')

############################
    
    #PCA Raw Data (Stack)
    X = sample_raw_data_stack
    pca = PCA(n_components = 0.95)
    Xt = pca.fit_transform(X)

    print(f'xt: {Xt}')

        #Plot 1st and 2nd PC againt each other
    x = Xt[:,0]
    y = Xt[:,1]
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    plt.figure()
    plot = plt.scatter(x, y, c=z)
    plt.savefig(figure_directory+'raw_stack_pca_plot.png')
    plt.close() 

        #Plot Variance Explained
    features = range(len(pca.explained_variance_))  
    print("Number of principal components to explain 95 percent of data (Radial Raw) =", len(features))
    plt.figure()
    total_variance = np.sum(pca.explained_variance_)
    explained_variance = (pca.explained_variance_ / total_variance) * 100
    plot2 = plt.bar(features[:10], explained_variance[:10])
    plt.savefig(figure_directory+'raw_stack_pca_variance.png')
    plt.close() 


        #Same as above but Scaled
    pca = PCA()
    pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
    Xt = pipe.fit_transform(X)
    x1 = Xt[:,0]
    y1 = Xt[:,1]
    z1 = Xt[:,2]
    xy1 = np.vstack([x1,y1])
    z1 = gaussian_kde(xy1)(xy1)
    plt.figure()
    plot = plt.scatter(x1, y1, c=z1)
    plt.savefig(figure_directory+'raw_stack_scaled_pca_plot.png')


    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('white')
   
    ax.scatter(x1, y1, z1,  s=40)
    ax.set_xlabel("First PC", fontsize=14)
    ax.set_ylabel("Second PC", fontsize=14)
    ax.set_zlabel("Third PC", fontsize=14)
    plt.title("3D Raw PCA Plot (Radial)", fontsize=16)
    plt.savefig(figure_directory+'raw_stack_3d_pca_plot.png')

############################

   #PCA encoded Data
    X = sample_encoded_data 
    pca = PCA(n_components = 0.95)
    Xt = pca.fit_transform(X)

        #Plot 1st and 2nd PC againt each other
    x = Xt[:,0]
    y = Xt[:,1]
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    plt.figure()
    plot = plt.scatter(x, y, c=z)
    plt.savefig(figure_directory+'encoded_pca_plot.png')
    plt.close() 

        #Plot Variance Explained
    features = range(len(pca.explained_variance_))  
    print("Number of principal components to explain 95 percent of data (Encoded)=", len(features))
    plt.figure()
    total_variance = np.sum(pca.explained_variance_)
    explained_variance = (pca.explained_variance_ / total_variance) * 100
    plot2 = plt.bar(features[:10], explained_variance[:10])
    plt.savefig(figure_directory+'encoded_pca_variance.png')
    plt.close() 


        #Same as above but Scaled
    pca = PCA()
    pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
    Xt = pipe.fit_transform(X)
    x1 = Xt[:,0]
    y1 = Xt[:,1]
    z1 = Xt[:,1]
    xy1 = np.vstack([x1,y1])
    z1 = gaussian_kde(xy1)(xy1)
    plt.figure()
    plot = plt.scatter(x1, y1, c=z1)
    plt.savefig(figure_directory+'encoded_scaled_pca_plot.png')
    plt.close() 



    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('white')
   
    ax.scatter(x1, y1, z1,  s=40)
    ax.set_xlabel("First PC", fontsize=14)
    ax.set_ylabel("Second PC", fontsize=14)
    ax.set_zlabel("Third PC", fontsize=14)
    plt.title("3D PCA Encoded Data Plot", fontsize=16)
    plt.savefig(figure_directory+'encoded_3d_pca_plot.png')





   
