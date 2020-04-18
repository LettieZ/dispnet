# network architecture test
import model


net=model.DispNet()

net.weight_bias_init()
# import torch,dataloader
# from torch.utils.data import DataLoader
#
# left_image_path="/Users/liuchunpu/dispnet/left/"
# right_image_path="/Users/liuchunpu/dispnet/right/"
# gt_path="/Users/liuchunpu/dispnet/gt/"
#
#
#
#
# dataset=dataloader.Stereo_Dataset(left_image_path,right_image_path,gt_path)
# train_dataloader=DataLoader(dataset=dataset,batch_size=1,shuffle=True,num_workers=4)
# print("dataloader competed")
#
# x,y=dataset[0]
# for i in y[0][300]:
#     print(i)
# print(y)