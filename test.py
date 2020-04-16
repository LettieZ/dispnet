import torch
y=torch.randn([4,1,384,768])
x=torch.randn([4,1,6,12])



downsampled_y=torch.nn.functional.interpolate(y,size=x.shape)
print(downsampled_y.shape)