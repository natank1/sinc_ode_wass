import torch
from torch import nn

class bhat(nn.Module):
    def __init__(self,func0,targ_mode, device,nb_classes):
        super( bhat, self).__init__()
        self.loss= func0
        self.device = device
        self.one_hot_flag = targ_mode
        # self.one_hot_flag =mode_target.one_hot.value
        self.nb_classes = nb_classes
        self.ones= torch.sparse.torch.eye(nb_classes).to(device)
        self.label_proc = self.bring_label_proc()

    def one_hot_v2(self,target):
        # ones = torch.sparse.torch.eye(depth).to(device)
        return self.ones.index_select(0, target)

    def wasser_coonv(self,target):
        yy = 2 * target - 1
        return yy.to(self.device)

    def bring_label_proc(self):

        if self.one_hot_flag < 3:
            return  self.one_hot_v2

        # Bayesian solution na/ right now
        # elif self.one_hot_flag == 4:
        #     target1 = helpers.generate_probability(input, target, self.device)
        else:
            return self.wasser_coonv

    def forward(self,input, target):
        target1= self.label_proc(target)
        return self.loss(input, target1)



def bhatscore(input, target):
    out = torch.mean(-torch.log(torch.sum(torch.sqrt(0.000005+torch.mul(input, target)), dim=1)))
    return out

def heilinger(input, target):
    xx=torch.sqrt(input+0.00005)-torch.sqrt(0.00005+target)
    yy = torch.mean(torch.norm(xx,p=2,dim=1))
    return yy
def wasserstein_score(input, target):
    z=torch.tensor( target.unsqueeze(dim=1)).float()
    uu=torch.mul(input,z)
    zz= torch.mean(uu)
    r1=zz.item()
    if r1>-10.0:
        return zz
    else:
        return -torch.sqrt(-zz)

    # return -torch.sqrt(torch.mean(uu))
