import numpy as np
# from torch.utils import data
from .pre_proc_struc import  data_proc




class  Gglobal_Organ():
    def __init__(self,parser,inp_dim,pars_db,dir_meta,  base_dir_cor=""):
        self.pars_db = pars_db
        self.nb_time = inp_dim


        self.base_dir_cor = self.pars_db + base_dir_cor
        self.xtrain = np.empty((0, self.nb_time))

        self.batch_size =parser['batch_size']
        self.ytrain =[]

        self.datarep = data_proc(parser, self.pars_db + dir_meta)



        self.datarep = data_proc(parser, self.pars_db + dir_meta)
        l_dev_utt, d_label_dev = self.datarep.procedure(folder_prefix=self.base_dir_cor)

        self.list_IDs = l_dev_utt
        self.labels = d_label_dev
        # sum=0
        # for j in self.labels.items():
        #
        #
        #     sum+=j[1]
        # print (sum,len(self.labels))



    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        X = np.load( ID + '.npy')

        nb_time = X.shape[1]
        if nb_time > self.nb_time:
            start_idx = np.random.randint(low=0,
                                          high=nb_time - self.nb_time)
            X = X[:, start_idx:start_idx + self.nb_time]
        elif nb_time < self.nb_time:
            nb_dup = int(self.nb_time / nb_time) + 1
            X = np.tile(X, (1, nb_dup))[:,  :self.nb_time]
        X =np.squeeze(X)

        y = self.labels[ID]

        return X, y


if __name__ =="__main__":
    ss= "Y:\\DataBases\\synthetic_speech\\db\\LA\\ASVspoof2019_LA_eval\\sinc_magnitude_2048_400_160\\LA_E_1000273.npy"
    print (ss)
    x= np.load(ss)
    print (x.shape)
    snt_len = x.shape[1]
    snt_beg = np.random.randint(snt_len - 3000 - 1)  # randint(0, snt_len-2*wlen-1)
    snt_end = snt_beg + 3000
    y= x[:,snt_beg: snt_end]
    print (y.shape)