In this directory, you can find implementation of single coil deep learning partial fourier reconstruction.
Official implementation can be found here : https://github.com/Linfang-mumu/PF-MRI-Reconstruction/tree/main

Data Preparation :
First we download and divide dataset. We will use Calgary campinas single coil data of brain images. 
https://sites.google.com/view/calgary-campinas-dataset/download?authuser=0

Then make a new directory singlecoil_dataset and divide .npy files into train(17 files), valid(7 files) and test(1 file) folder.
You can make the data distribution different from this.
 
In order to view the content inside data and preprocessing operations performed on single coil data, see the single_testing file . Next run the data_prepare_singlecoil jupyter notebook to generate partial fourier dataset. Here we used 0.55 partial fourier but code can be changed for 5/8 and 6/8. 

