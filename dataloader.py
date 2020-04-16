from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

# left_image_path="/Users/liuchunpu/kitti/stereoAndMV/data_scene_flow/training/image_2/"
# right_image_path="/Users/liuchunpu/kitti/stereoAndMV/data_scene_flow/training/image_3/"
# gt_path="/Users/liuchunpu/kitti/stereoAndMV/data_scene_flow/training/disp_occ_0/"



transformer=transforms.Compose([
    transforms.CenterCrop((384,768)),
    transforms.ToTensor()
])




# for KITTI stereo dataset,not surely suitable for FlyingThings3D dataset
class Stereo_Dataset(Dataset):

    def __init__(self,left_image_path,right_image_path,gt_path):

        self.left_image_path=left_image_path
        self.right_image_path=right_image_path
        self.gt_path=gt_path

        # for the particularity of the KITTI 2015 stereo dataset,I specify the __len__() function in a manual way
        # In most situations,we should return the length of the customed Dataset according to the number of files in the rendered path
        self.name_list = []
        for i in range(0, 200):
            self.name_list.append("000" + str(i).zfill(3) + "_10.png")



    def __len__(self):
        return len(self.name_list)


    def __getitem__(self, item):

        # left image,right image and gt are the same name
        current_data_name=self.name_list[item]

        left_image=Image.open(self.left_image_path+current_data_name).convert("RGB")
        right_image=Image.open(self.right_image_path+current_data_name).convert("RGB")
        gt_data=Image.open(self.gt_path+current_data_name)

        left_image_tensor=transformer(left_image)
        right_image_tensor=transformer(right_image)
        gt_data_tensor=transformer(gt_data)

        input_tensor=torch.cat((left_image_tensor,right_image_tensor),0)

        return input_tensor,gt_data_tensor