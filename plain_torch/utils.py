import torch
from torch.optim.lr_scheduler import _LRScheduler

class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
    

def evaInfo(score,label):

    RMAE = torch.sqrt(torch.mean(torch.abs(score - label)))
    RMSE = torch.sqrt(torch.mean(torch.abs(score - label) ** 2))

    info = ' RMSE : {:.4f} , \t  RMAE : {:.4f}'.format(
               RMSE,  RMAE) 

    return info

