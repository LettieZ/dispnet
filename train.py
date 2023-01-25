import model
import torch
import dataloader
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import argparse,os
import test

# torch.set_num_threads(int thread)  

torch.set_num_threads(4) # 多线程
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


left_image_path="kitti/training/image_2/"
right_image_path="kitti/training/image_3/"
gt_path="kitti/training/disp_occ_0/"



EPOCH=1000
#EPOCH=70
BATCH_SIZE=4
dataset=dataloader.Stereo_Dataset(left_image_path,right_image_path,gt_path)
train_dataloader=DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)  # num_workers=4改为0,多进程需要在main函数中运行
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


# optimzer=torch.optim.Adam(net.parameters(),lr=1e-4,weight_decay=0.0004,betas=[0.9,0.999])
# print("optimizer completed")

# scheduler=torch.optim.lr_scheduler.MultiStepLR(optimzer,gamma=0.5,milestones=[1500,3000,4500,6000,7500])
# print("scheduler completed")

loss_func = torch.nn.L1Loss(reduction='mean')
print("loss function defined")


for i in range(EPOCH):

    if i<10:
        optimzer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0.0004, betas=[0.9, 0.999])
        print("optimizer completed")
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimzer,gamma=0.5,milestones=[300,600,900,1200,1500])
        print("scheduler completed")
    elif (i>=10)&(i<20):
        optimzer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0.0004, betas=[0.9, 0.999])
        print("optimizer reset")
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimzer,gamma=0.5,milestones=[300,600,900,1200,1500])
        print("scheduler reset")
    elif (i>=20)&(i<30):
        optimzer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0.0004, betas=[0.9, 0.999])
        print("optimizer reset")
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimzer,gamma=0.5,milestones=[300,600,900,1200,1500])
        print("scheduler reset")
    elif (i>=30)&(i<40):
        optimzer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0.0004, betas=[0.9, 0.999])
        print("optimizer reset")
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimzer,gamma=0.5,milestones=[300,600,900,1200,1500])
        print("scheduler reset")
    elif (i>=40)&(i<50):
        optimzer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0.0004, betas=[0.9, 0.999])
        print("optimizer reset")
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimzer,gamma=0.5,milestones=[300,600,900,1200,1500])
        print("scheduler reset")
    else:
        optimzer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0.0004, betas=[0.9, 0.999])
        print("optimizer reset")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimzer, gamma=0.5, milestones=[300,600,900,1200,1500])
        print("scheduler reset")


    for step,(input,y) in enumerate(train_dataloader):
        input=input.to(device)
        y=y.to(device)/256.0/256

        total_iter = i * len(train_dataloader) + step

        pr6, pr5, pr4, pr3, pr2, pr1 = net(input)
        if i<100:
            output=pr6
        elif (i>=100)&(i<200):
             output=pr5
        elif (i>=200)&(i<300):
             output=pr4
        elif (i>=300)&(i<400):
             output=pr3
        elif (i>=400)&(i<500):
             output=pr2
        else:
             output=pr1


        y=y.float()
        downsampled_y=torch.nn.functional.interpolate(y,size=output[0][0].shape)

        loss=loss_func(downsampled_y,output)
        average_EPE=test.EPE_ERROR(downsampled_y,output)

        # loss value record
        print("epoch:",i,"step:",step,"total_iter",total_iter,"loss:",loss.item())
        writer.add_scalar("loss",loss.item(),total_iter)
        if(EPOCH>500):
            writer.add_scalar("EPE:",average_EPE.item(),total_iter)

        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
        scheduler.step()




torch.save(net.state_dict(),"current_model.pth")
writer.close()