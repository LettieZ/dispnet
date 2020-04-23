import model
from torchvision import transforms
from PIL import Image
import torch


def warp_fn(original_img_tensor,disparity_tensor):
#     assume that the original_img_tensor is of 1*3*192*384 size and disparity_tensor is of 1*1*192*384 size
    warped_img_tensor=torch.zeros(original_img_tensor.shape)
    width=len(original_img_tensor[0][0][0])

    for i in range(len(original_img_tensor[0][0])):
        for j in range(len(original_img_tensor[0][0][0])):

            disparity=int(disparity_tensor[0][0][i][j])
            if ((j-disparity)>=0)&((j-disparity)<width):
                warped_img_tensor[0][0][i][j-disparity]=original_img_tensor[0][0][i][j]
                warped_img_tensor[0][1][i][j-disparity]=original_img_tensor[0][1][i][j]
                warped_img_tensor[0][2][i][j-disparity]=original_img_tensor[0][2][i][j]

    return warped_img_tensor


left_image_path="/Users/liuchunpu/kitti/stereoAndMV/data_scene_flow/training/image_2/000000_10.png"
right_image_path="/Users/liuchunpu/kitti/stereoAndMV/data_scene_flow/training/image_3/000000_10.png"

left_image=Image.open(left_image_path).convert("RGB")
right_image=Image.open(right_image_path).convert("RGB")

transformer=transforms.Compose([
    transforms.CenterCrop((384,768)),
    transforms.ToTensor()
])


left_image_tensor=transformer(left_image)
right_image_tensor=transformer(right_image)

left_image_tensor=left_image_tensor.unsqueeze(0)
right_image_tensor=right_image_tensor.unsqueeze(0)

input_tensor=torch.cat((left_image_tensor,right_image_tensor),1)

device=torch.device('cpu')
net=model.DispNet()
trained_model_path="/Users/liuchunpu/dispnet/current_model.pth"
net.load_state_dict(torch.load(trained_model_path, map_location=lambda storage, loc: storage))
net.eval()
pr6,pr5,pr4,pr3,pr2,disparity_tensor=net(input_tensor)
disparity_tensor=disparity_tensor*256.0

downsampled_left=torch.nn.functional.interpolate(left_image_tensor,size=[192,384])
downsampled_right=torch.nn.functional.interpolate(right_image_tensor,size=[192,384])

warped_tensor=warp_fn(downsampled_left,disparity_tensor)


# print("left image")
# downsampled_left_img=transforms.ToPILImage()(downsampled_left[0])
# downsampled_left_img.show()

print("right image")
downsampled_right_img=transforms.ToPILImage()(downsampled_right[0])
downsampled_right_img.show()

print("disparity map")
disparity_map=transforms.ToPILImage()(disparity_tensor[0][0])
disparity_map.show()

print("warped image")
warped_img=transforms.ToPILImage()(warped_tensor[0])
warped_img.show()






