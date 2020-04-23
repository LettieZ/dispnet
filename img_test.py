from PIL import Image
from torchvision import transforms

gt_path="/Users/liuchunpu/kitti/stereoAndMV/data_scene_flow/training/disp_occ_0/000000_10.png"


gt_image=Image.open(gt_path)


transformer=transforms.Compose([
    # transforms.CenterCrop((384,768)),
    transforms.ToTensor()
])

gt_tensor=transformer(gt_image)
gt_tensor=gt_tensor.float()/256.0

disparity_map=transforms.ToPILImage()(gt_tensor)
disparity_map.show()
