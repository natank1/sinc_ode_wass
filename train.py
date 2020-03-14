from comet_ml import Experiment
from tqdm import tqdm
import os
import yaml
import numpy as np
import torch
from torch import nn
# import torch.nn as nn
from torch.utils import data
from torch_for_privae.organ_sinc import Gglobal_Organ as global_organ
from torch_for_privae.penatlies import  penalty, trival
from torch_for_privae.test_tools import test_tools
import torch_for_privae.helpers as helpers
import torch_for_privae.eval_block as eval_block
from torch_for_privae.resnet_struct import  resnet_arc
from torch_for_privae.create_data import Data_prepare


if __name__ == '__main__':

    _abspath = os.path.abspath(__file__)
    dir_yaml = os.path.splitext(_abspath)[0] + '.yaml'
    with open(dir_yaml, 'r') as f_yaml:
        parser = yaml.load(f_yaml)
    inp_dim = parser['input_dim']
    test_dev = test_tools(parser['only_eval'],parser['non_vanilla_target'])


    # device setting
    device = helpers.bring_device(parser['gpu_idx'][0])

   
    # devsetL = global_organ(parser,inp_dim, parser['DB_LA'], parser['dir_LA_meta_dev'], parser["DB_dev"])
    devsetL= Data_prepare(parser['use_balance'],inp_dim, parser['DB_LA'],parser['data_eval'], parser["lab_eval"])


    devset_gen = data.DataLoader(devsetL,
                                 batch_size=parser['batch_size'],
                                 shuffle=False,
                                 drop_last=False,
                                 num_workers=parser['nb_proc_db'])

   

    # set save directory
    save_dir = helpers.create_res_folder(parser)


    # to comet server
    experiment = helpers.experiment_flag(parser['comet_disable'], parser['name'])

    helpers.experiment_proc(parser,experiment)
    model = resnet_arc( parser['model'],inp_dim ,device, 1, 1, 0,parser['use_ode']).to(device)

    eval_dev_struc = eval_block.eval_proc(devset_gen,device,parser['w_point_fp'],save_dir)

    with open(save_dir + 'summary.txt', 'w+') as f_summary:
        if len(parser['gpu_idx']) > 1:
              model = nn.DataParallel(model, device_ids=parser['gpu_idx'])

          # set ojbective funtions


        criterion =helpers.set_critetion( parser['vanilla_kl'], parser['non_vanilla_target'], device, parser['model']['nb_classes'])
        penalty_func =penalty if parser['use_penalty']  else trival
        # heilinger
        # set optimizer
        params = list(model.parameters())
        if parser['optimizer'].lower() == 'sgd':
            optimizer = torch.optim.SGD(params,
                                        lr=parser['lr'],
                                        momentum=parser['opt_mom'],
                                        weight_decay=parser['wd'],
                                        nesterov=bool(parser['nesterov']))

        elif parser['optimizer'].lower() == 'adam':
            optimizer = torch.optim.Adam(params,
                                         lr=parser['lr'],
                                         weight_decay=parser['wd'],
                                         amsgrad=bool(parser['amsgrad']))



        for epoch in tqdm(range(parser['epoch'])):

            trnsetL = Data_prepare(parser['use_balance'], inp_dim, parser['DB_LA'], parser['data_trn'],
                                   parser["lab_trn"])

            trnset_gen = data.DataLoader(trnsetL,
                                         batch_size=parser['batch_size'],
                                         shuffle=True,
                                         drop_last=True,
                                         num_workers=parser['nb_proc_db'])
            print (len(trnsetL.list_IDs))
            # train phase
            model.train()

            with tqdm(total=len(trnset_gen), ncols=70) as pbar:
                cntr = 0
                for m_batch, m_label in trnset_gen:
                    m_label =  m_label.to(device)

                    code, output= model(m_batch)
                    cce_loss = criterion(output, m_label)

                    penalty_score = penalty_func(m_label,output,device)
                    loss =cce_loss+penalty_score


                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


                    pbar.set_description('epoch%d,loss:%.3f' % (epoch, loss))
                    pbar.update(1)
            eval_dev_struc.eval_process(model,tqdm,test_dev,epoch,True,  parser,experiment)

    print ("Final EVAL test")

