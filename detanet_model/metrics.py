import torch
#from .utils import get_internal_pos,get_group_index


def l2loss(out,target,dim=None):
    '''Mean Square Error(MSE)'''
    diff = out-target
    return torch.mean(diff ** 2,dim=dim)


def l1loss(out,target,dim=None):
    '''Mean Absolute Error(MAE)'''
    return torch.mean(torch.abs(out-target),dim=dim)


def rmse(out, target, dim=None):
    '''Root Mean Square Eroor(rmse) (also known as RMSD)'''
    return torch.sqrt(torch.mean((out - target) ** 2,dim=dim))



def R2(out,target):
    '''coefficient of determination,Square of Pearson's coefficient, used to assess regression accuracy'''
    mean=torch.mean(target)
    SSE=torch.sum((out-target)**2)
    SST=torch.sum((mean-target)**2)
    return 1-(SSE/SST)


def phase_less_l1loss(out, target):
    neg = torch.square(target - out).unsqueeze(-1)
    pos = torch.square(target + out).unsqueeze(-1)
    vec = torch.cat((pos, neg), dim=-1)
    return torch.mean(torch.min(vec, dim=-1)[0])


def phase_less_l2loss(out, target):
    neg = torch.square(target - out).unsqueeze(-1)
    pos = torch.square(target + out).unsqueeze(-1)
    vec = torch.cat((pos, neg), dim=-1)
    return torch.sqrt(torch.mean(torch.min(vec, dim=-1)[0]))


def ef_combine_loss(out_e, targ_e, out_f, targ_f, coff=1000):
    loss_e=l2loss(out_e, targ_e)
    loss_f=l2loss(out_f, targ_f)
    return loss_e + (coff * loss_f)

