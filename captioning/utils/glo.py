import torch

def _init():
    global HOT
    HOT = torch.zeros([10, 10], dtype=torch.float)

def add_value(attn):
    global HOT
    tmp = torch.mean(torch.mean(attn.detach(), dim=1), dim=0).cpu()
    if True in tmp.isnan():
        return
    L = tmp.shape[0]
    if L >= 10:
        HOT += tmp[:10, :10]
    else:
        HOT[:L, :L] += tmp

def get_value():
    return HOT
