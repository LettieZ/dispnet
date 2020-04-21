import dataloader
import torch
from torch.utils.data import DataLoader
import model

output=torch.randn(4,1,6,12)
y=torch.randn(4,1,6,12)



def EPE_ERROR(y,output):
    total_EPE=0

    for i in range(4):
        total_EPE = total_EPE + torch.dist(y[i][0],output[i][0],p=2)

    average_EPE=total_EPE/4

    return average_EPE


d=EPE_ERROR(y,output)

print(d)
print(d.item())