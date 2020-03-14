# tr_lst=data_lists/TIMIT_train.scp
# te_lst=data_lists/TIMIT_test.scp
# lab_dict=data_lists/TIMIT_labels.npy
# data_folder=./TIMIT_norm_nosil
# output_folder=data/
# pt_file=none

# [windowing]
fs=8000
cw_len=200
cw_shift=10

# [cnn]
cnn_N_filt=120,120,120
cnn_len_filt=251,5,5
cnn_max_pool_len=3,3,3
cnn_use_laynorm_inp=True
cnn_use_batchnorm_inp=False
cnn_use_laynorm=True,True,True
cnn_use_batchnorm=False,False,False
cnn_act="leaky_relu","leaky_relu","leaky_relu"
cnn_drop=0.0,0.0,0.0

# [dnn]
fc_lay=2048,2048,2048
fc_drop=0.0,0.0,0.0
fc_use_laynorm_inp=True
fc_use_batchnorm_inp=False
fc_use_batchnorm=True,True,True
fc_use_laynorm=False,False,False
fc_act="leaky_relu","leaky_relu","leaky_relu"

# [class]
class_lay=462
class_drop=0.0
class_use_laynorm_inp=False
class_use_batchnorm_inp=False
class_use_batchnorm=False
class_use_laynorm=False
# class_act=softmax

# [optimization]
lr=0.001
batch_size=128
N_epochs=1500
N_batches=800
N_eval_epoch=8
seed=1234