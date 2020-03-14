import os
import numpy as np

class data_proc():
    def __init__(self,parser,dir_meta):
        self.dir_meta =dir_meta
        self.parser=parser
        return



    def procedure(self,folder_prefix):
        l_gen_dev, l_spo_dev, d_label_dev = self.split_genSpoof(folder_prefix,True)
        l_dev_utt = self.balance_classes(l_gen_dev, l_spo_dev, 0)  # for speed-up only
        return l_dev_utt, d_label_dev

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



    def split_genSpoof(self,folder_prefix, return_dic_meta=False):
        l_gen, l_spo = [], []
        d_meta = {}

        with open(self.dir_meta, 'r') as f:
            l_meta = f.readlines()
        for line in l_meta:
            _, key, _, _, label = line.strip().split(' ')
            clean_key = folder_prefix + key
            d_meta[clean_key] = 1 if label == 'bonafide' else 0

        for k in d_meta.keys():
            if d_meta[k] == 1:
                l_gen.append(k)
            else:
                l_spo.append(k)

        if return_dic_meta:

            return l_gen, l_spo, d_meta
        else:
            return l_gen, l_spo

    def exter_split(self, src_dir, return_dic_meta=True):
        l_in =self.ext_get_utt_lis(src_dir)
        # def split_genSpoof(l_in, dir_meta, return_dic_meta=False):
        l_gen, l_spo = [], []
        d_meta = {}
        for j in l_in:
            d_meta[j] = 1

        for k in d_meta.keys():
            if d_meta[k] == 1:
                l_gen.append(k)
            else:
                l_spo.append(k)

        if return_dic_meta:

            return l_gen, l_spo, d_meta
        else:
            return l_gen, l_spo

    def ext_get_utt_lis(self,src_dir):
        '''
        Designed for ASVspoof2019 PA
        '''
        l_utt = []
        for r, ds, fs in os.walk(src_dir):
            for f in fs:
                if f[-3:] != 'npy': continue
                l_utt.append(f.split('.')[0])

        return l_utt
