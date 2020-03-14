import torch
oneex = float(1.0)
slope =float(-2000.)
incep = float(790.)
coef= float(1.0)
half0 =float(0.5)

def trival (y_true,y_pred0,device):

    return torch.tensor(0.).type((torch.FloatTensor))
def penalty(y_true,y_pred0,device):

    const_coeff= y_true.type(torch.FloatTensor).to(device)
    denom = torch.sum(const_coeff)
    if denom == 0.0:
        return 0.0

    ss =torch.exp(y_pred0)
    ss1 =torch.sum(ss,dim=1)
    y_pred = ss[:,0]/ss1

    res=torch.tanh_(slope*y_pred+incep)

    res= half0*(res+oneex)
    m1 =res*const_coeff
    result= torch.sum(m1)/denom

    return coef*result
