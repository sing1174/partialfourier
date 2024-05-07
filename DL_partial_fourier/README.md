In this directory, you can find implementation of single coil deep learning partial fourier reconstruction.
Official implementation can be found here : https://github.com/Linfang-mumu/PF-MRI-Reconstruction/tree/main

### Data Preparation: 
First we download and divide dataset. We will use Calgary campinas single coil data of brain images:
https://sites.google.com/view/calgary-campinas-dataset/download?authuser=0

Then make a new directory singlecoil_dataset and divide .npy files into train(17 files), valid(7 files) and test(1 file) folder.You can make the data distribution different from this.
 
In order to view the content inside data and preprocessing operations performed on single coil data, see the single_testing jupyter notebook file . Next run the data_prepare_singlecoil notebook to generate partial fourier dataset. This will generate new folder called generated_dataset. Here we used 0.55 partial fourier but code can be changed for 5/8 and 6/8. (Note: We made changes in original code data_prepare.py)

### Model changes: 
Next step is to train our code. Before that we need to make changes to model2_cpx.py file. Since we are running only single coil model, we make changes to class Net_cpx only, from the original code. We reduced number of features and residual blocks to make the code run faster. Also, in this class, you need to changes pf from 0.45 if needed. Now, the model should run fine. 

No changes were made in losses.py code but we can explore changes in it for future to make model better.

Before we proceed with training, we need to activate GPU for faster training. We used Nvidia 3050ti inbuild in gaming laptop for our purpose. We ran all code on windows 11 using WSL2 (Windows Subsystem for Linux). You can follow their guide given on Nvidia website : https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl-2 . See requirement.txt (created using pipreqs) for the python libraries used. 

### Training: 
Run training_singlecoil notebook for training model and viewing result. From original code, we removed the testing part and kept training and validation part only. 


Other files directory contains other methods used for code construction.





