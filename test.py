import dataloader
import torch
from torch.utils.data import DataLoader
import model

if torch.cuda.is_available():
    device=torch.device("cuda:0")
    print("GPU is available")
else:
    device=torch.device("cpu")
    print("GPU is unavailable")


def accuracy_compute(gt,output):
    EPE=torch.dist(gt,output,p=2)
    return EPE

left_image_path="/Users/liuchunpu/kitti/stereoAndMV/data_scene_flow/training/image_2/"
right_image_path="/Users/liuchunpu/kitti/stereoAndMV/data_scene_flow/training/image_3/"
gt_path="/Users/liuchunpu/kitti/stereoAndMV/data_scene_flow/training/disp_occ_0/"

test_dataset=dataloader.Stereo_Dataset(left_image_path,right_image_path,gt_path)
train_dataloader=DataLoader(dataset=test_dataset,batch_size=1,shuffle=False,num_workers=4)

net=model.DispNet()
# need to load the pretrained model





total_EPE=0

for input,y in train_dataloader:
    input=input.to(device)
    y=y.to(device)

    pr6, pr5, pr4, pr3, pr2, pr1 = net(input)
    output=pr1

    y=y.float()
    downsampled_y=torch.nn.functional.interpolate(y,size=output[0][0].shape)
    total_EPE=total_EPE+accuracy_compute(downsampled_y[0][0],output[0][0])



average_EPE=total_EPE/200


