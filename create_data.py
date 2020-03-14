import numpy as np



class  Data_prepare():
    def __init__(self,use_balance, inp_dim, pars_db,dir_meta,  base_dir_cor):
        self.pars_db = pars_db
        self.inp_dim = inp_dim
        self.dir_meta_file =pars_db+dir_meta

        self.base_dir_cor = self.pars_db + base_dir_cor
        self.use_balance =use_balance

        self.xtrain = np.empty((0, self.inp_dim))

        self.ytrain =[]
        list_of_id, d_meta = self.creaete_lists_for_data_loader()

        self.list_IDs = list_of_id
        self.labels =d_meta


    def creaete_lists_for_data_loader(self):
        l_gen, l_spo = [], []
        d_meta = {}

        with open(self.dir_meta_file, 'r') as f:
            l_meta = f.readlines()
        for line in l_meta:
            _, key, _, _, label = line.strip().split(' ')  # special foramt
            clean_key = self.base_dir_cor + key
            d_meta[clean_key] = 1 if label == 'bonafide' else 0

        for k in d_meta.keys():
            if d_meta[k] == 1:
                l_gen.append(k)
            else:
                l_spo.append(k)
        if self.use_balance:
            list_of_id= self.balance_classes(l_gen, l_spo,0)
        else:
            list_of_id = l_gen.extend((l_spo))

        return list_of_id, d_meta

    def balance_classes(self,lines_small, lines_big, np_seed):
        '''
        Balance number of sample per class.
        Designed for Binary(two-class) classification.
        '''

        len_small_lines = len(lines_small)
        len_big_lines = len(lines_big)
        idx_big = list(range(len_big_lines))

        np.random.seed(np_seed)
        np.random.shuffle(lines_big)
        new_lines = lines_small + lines_big[:len_small_lines]
        np.random.shuffle(new_lines)
        # print(new_lines[:5])

        return new_lines

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        X = np.load( ID + '.npy')

        nb_time = X.shape[1]
        if nb_time > self.inp_dim:
            start_idx = np.random.randint(low=0,
                                          high=nb_time -self.inp_dim)
            X = X[:, start_idx:start_idx + self.inp_dim]
        elif nb_time < self.inp_dim:
            nb_dup = int(self.inp_dim / nb_time) + 1
            X = np.tile(X, (1, nb_dup))[:,  :self.inp_dim]
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