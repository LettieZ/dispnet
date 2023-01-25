# network architecture test
import dataloader
from PIL import Image
import numpy as np
from torchvision import transforms


transformer2=transforms.Compose([
    transforms.CenterCrop((384,768)),
    transforms.ToTensor()
])



left_image_path="kitti/training/image_2/"
right_image_path="kitti/training/image_3/"
gt_path="kitti/training/disp_occ_0/"



dataset=dataloader.Stereo_Dataset(left_image_path=left_image_path,right_image_path=right_image_path,gt_path=gt_path)


x,y=dataset[0]

print(y.shape)
print(y[0][300][500])


y_img=Image.open(gt_path+"000000_10.png")
transformer=transforms.Compose([
    transforms.CenterCrop((384,768))])

yt=transformer(y_img)

pix=yt.getpixel((500,300))

print(pix)


y2=transformer2(y_img)

print(y[0][300][500])


# print(yt.size)
#
# y_np=np.asarray(yt)
#
# print(y_np.shape)
# print(y_np[300][500][2])
