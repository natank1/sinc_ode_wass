# sinc_ode_wass
Solving speech problems with neural ode and sincnet
The folder contains several files that we will describe here:

train.yaml - Simply yaml file with the data params

Train.py- This is the main routine code that runs the entire process 

Create_data - The data upload management , really suggest to write it by yourself and simply call dataloader immediately  afterwards.

loss_method - I tried in addition to KL divergence some other loss functions such as Wasserstein and Bahtacharyaa .

resnet.py- Simply resnet.

ode-hleprs - These are helpers for the nural ode
