import os
import torch
from comet_ml import Experiment
from torch import nn
import torch_for_privae.loss_methods  as  loss
from torch_for_privae.penatlies import  penalty
def bring_device (parser_val):
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:%s' % parser_val if cuda else 'cpu')
    return device


def experiment_flag(parser_com_dis, parser_name) :
    if not bool(parser_com_dis):
        experiment = Experiment(api_key="PUT YOUR COMET API KEY",
                                project_name="spoof19", workspace="WORKSPACE",
                                disabled=bool(parser_com_dis))
        experiment.set_name(parser_name)
        return experiment
    return []


def experiment_proc(parser,experiment):
    # to comet server
    if not bool(parser['comet_disable']):
        experiment.log_parameters(parser)
        experiment.log_parameters(parser['model'])

def create_res_folder(parser):
    save_dir = parser['save_dir'] + parser['name'] + '\\'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir + 'results/'):
        os.makedirs(save_dir + 'results/')
    if not os.path.exists(save_dir + 'models/'):
        os.makedirs(save_dir + 'models/')

        # log experiment parameters to local and comet_ml server
        # to local

    f_params = open(save_dir + 'f_params.txt', 'w')
    for k, v in parser.items():
        f_params.write('{}:\t{}\n'.format(k, v))
    f_params.write('DNN model params\n')

    for k, v in parser['model'].items():
        f_params.write('{}:\t{}\n'.format(k, v))
    f_params.close()
    return save_dir


def bring_model (model_file, flag_data ,model, parser_model, device):
        if device =='cpu':
            model.load_state_dict(torch.load(model_file, map_location='cpu'))
        else:
            model.load_state_dict(torch.load(model_file))


        return model


def one_hot_v2(batch,device,depth):
    ones = torch.sparse.torch.eye(depth).to(device)
    return ones.index_select(0,batch)

def wasser_coonv(target,device,depth):
    yy=2*target-1
    return yy.to(device)

def generate_probability (input,output,device):
    a, _ =input.shape
    xr = torch.FloatTensor(a, 1).uniform_(0, 1.0)
    xr=xr.to(device)
    t2 ,_=torch.min(input,dim=1)
    t2=t2.unsqueeze(dim=1)
    new_vals =torch.mul(t2,xr)+(1-t2)
    # new_vals=new_vals.unsqueeze(dim=1)
    new_min_val  = 1-new_vals
    all_p = torch.cat((new_min_val,new_vals),dim=1).squeeze()
    x3 = torch.where(torch.tensor(output == 1.).unsqueeze(dim=1), all_p, all_p.flip(1, ))
    return x3

def set_critetion(vanilla_kl,target_f,device, nb_classes):
    if vanilla_kl:
        # return loss.bhat(loss.kl_with_penalty, target_f, device,nb_classes)
        return nn.CrossEntropyLoss()
    else:
        if target_f == 1:
            return loss.bhat(loss.bhatscore, target_f, device,nb_classes)
        elif target_f == 2:
            return loss.bhat(loss.heilinger, target_f, device,nb_classes)
        elif target_f == 3:
            return loss.bhat(loss.wasserstein_score, target_f, device,nb_classes)