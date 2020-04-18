import model
import torch
import dataloader
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import argparse,os

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



EPOCH=1000
BATCH_SIZE=5
dataset=dataloader.Stereo_Dataset(left_image_path,right_image_path,gt_path)
train_dataloader=DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)
print("dataloader competed")



net=model.DispNet()
if os.path.exists("current_model.pth"):
    print("trained model exists")
    net.load_state_dict(torch.load("current_model.pth"))
else:
    print("new model initialization")

net.to(device)

print("model built")


# for the initialization of weights and biases
net.weight_bias_init()
print("initialization completed")


optimzer=torch.optim.Adam(net.parameters(),lr=1e-4,weight_decay=0.0004,betas=[0.9,0.999])
print("optimizer completed")
scheduler=torch.optim.lr_scheduler.MultiStepLR(optimzer,gamma=0.5,milestones=[300,600,900,1200,1500])
print("scheduler completed")
loss_func = torch.nn.L1Loss(reduction='mean')
print("loss function defined")


for i in range(EPOCH):
    for step,(input,y) in enumerate(train_dataloader):
        input=input.to(device)
        y=y.to(device)/256.0

        pr6, pr5, pr4, pr3, pr2, pr1 = net(input)
        # if step<10:
        #     output=pr6
        # elif (step>=10)&(step<15):
        #     output=pr5
        # elif (step>=15)&(step<20):
        #     output=pr4
        # elif (step>=20)&(step<30):
        #     output=pr3
        # elif (step>=30)&(step<40):
        #     output=pr2
        # else:
        output=pr1

        y=y.float()
        downsampled_y=torch.nn.functional.interpolate(y,size=output[0][0].shape)

        loss=loss_func(downsampled_y,output)

        # loss value record
        total_iter=i*len(train_dataloader)+step
        print("epoch:",i,"step:",step,"total_iter",total_iter,"loss:",loss.item())
        writer.add_scalar("loss",loss.item(),total_iter)


        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
        scheduler.step()
        print(scheduler.get_lr())



torch.save(net.state_dict(),"current_model.pth")
writer.close()