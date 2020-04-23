import model,torch
from PIL import Image
from torchvision import transforms


# module load
device=torch.device('cpu')
net=model.DispNet()
trained_model_path="/Users/liuchunpu/dispnet/current_model.pth"
net.load_state_dict(torch.load(trained_model_path, map_location=lambda storage, loc: storage))
net.eval()

# left_image_path="/Users/liuchunpu/kitti/stereoAndMV/data_scene_flow/training/image_2/"
# right_image_path="/Users/liuchunpu/kitti/stereoAndMV/data_scene_flow/training/image_3/"
# gt_path="/Users/liuchunpu/kitti/stereoAndMV/data_scene_flow/training/disp_occ_0/"



left_image_path="/Users/liuchunpu/kitti/stereoAndMV/data_scene_flow/training/image_2/000000_10.png"
right_image_path="/Users/liuchunpu/kitti/stereoAndMV/data_scene_flow/training/image_3/000000_10.png"
gt_path="/Users/liuchunpu/kitti/stereoAndMV/data_scene_flow/training/disp_occ_0/000000_10.png"

left_image=Image.open(left_image_path).convert("RGB")
right_image=Image.open(right_image_path).convert("RGB")
gt_image=Image.open(gt_path)

transformer=transforms.Compose([
    transforms.CenterCrop((384,768)),
    transforms.ToTensor()
])


gt_tensor=transformer(gt_image)
gt_tensor=gt_tensor.unsqueeze(0)
gt_tensor=gt_tensor.float()
downsampled_gt=torch.nn.functional.interpolate(gt_tensor,size=[192,384])
gt_img=transforms.ToPILImage()(downsampled_gt[0])
gt_img.show()

print("-------")

left_image_tensor=transformer(left_image)
right_image_tensor=transformer(right_image)

input_tensor=torch.cat((left_image_tensor,right_image_tensor),0)
input_tensor=input_tensor.unsqueeze(0)


# predictions of various scales
pr6,pr5,pr4,pr3,pr2,output_tensor=net(input_tensor)
output_tensor=output_tensor[0][0]*256*256

output_img=transforms.ToPILImage()(output_tensor)
output_img.show()











