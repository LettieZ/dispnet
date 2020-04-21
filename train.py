import model
import torch
import dataloader
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import argparse,os
import test

torch.set_num_threads(4)
if torch.cuda.is_available():
    device=torch.device("cuda:0")
    print("GPU is available")
else:
    device=torch.device("cpu")
    print("GPU is unavailable")

writer=SummaryWriter("data_visualization")

parser=argparse.ArgumentParser()
parser.add_argument("--left_image_path",help="the absolute path of left images")
parser.add_argument("--right_image_path",help="the absolute path of right images")
parser.add_argument("--ground_truth_path",help="the absolute path of ground truth")
args=parser.parse_args()


left_image_path=args.left_image_path
right_image_path=args.right_image_path
gt_path=args.ground_truth_path


left_image_path="/Users/liuchunpu/kitti/stereoAndMV/data_scene_flow/training/image_2/"
right_image_path="/Users/liuchunpu/kitti/stereoAndMV/data_scene_flow/training/image_3/"
gt_path="/Users/liuchunpu/kitti/stereoAndMV/data_scene_flow/training/disp_occ_0/"



EPOCH=600
BATCH_SIZE=4
dataset=dataloader.Stereo_Dataset(left_image_path,right_image_path,gt_path)
train_dataloader=DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)
print("dataloader competed")



net=model.DispNet()


if os.path.exists("current_model.pth"):
    print("trained model exists")
    net.load_state_dict(torch.load("current_model.pth"))
else:
    # for the initialization of weights and biases
    print("new model initialization")
    net.weight_bias_init()
    print("initialization completed")

net.to(device)

print("model built")





optimzer=torch.optim.Adam(net.parameters(),lr=1e-4,weight_decay=0.0004,betas=[0.9,0.999])
print("optimizer completed")
scheduler=torch.optim.lr_scheduler.MultiStepLR(optimzer,gamma=0.1,milestones=[300,600,900,1200,1500])
print("scheduler completed")
loss_func = torch.nn.L1Loss(reduction='mean')
print("loss function defined")


for i in range(EPOCH):
    for step,(input,y) in enumerate(train_dataloader):
        input=input.to(device)
        y=y.to(device)/256.0/256

        total_iter = i * len(train_dataloader) + step

        pr6, pr5, pr4, pr3, pr2, pr1 = net(input)
        if total_iter<1500:
            output=pr6
        elif (total_iter>=1500)&(total_iter<3000):
            output=pr5
        elif (total_iter>=3000)&(total_iter<4500):
            output=pr4
        elif (total_iter>=4500)&(total_iter<6000):
            output=pr3
        elif (total_iter>=6000)&(total_iter<7500):
            output=pr2
        else:
            output=pr1


        y=y.float()
        downsampled_y=torch.nn.functional.interpolate(y,size=output[0][0].shape)

        loss=loss_func(downsampled_y,output)
        average_EPE=test.EPE_ERROR(downsampled_y,output)

        # loss value record
        # print("epoch:",i,"step:",step,"total_iter",total_iter,"loss:",loss.item())
        writer.add_scalar("loss",loss.item(),total_iter)
        if(total_iter>=7500):
            writer.add_scalar("EPE:",average_EPE.item(),total_iter)

        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
        scheduler.step()




torch.save(net.state_dict(),"current_model.pth")
writer.close()