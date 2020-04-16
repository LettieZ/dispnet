# network architecture test
import model
import torch


test_tensor=torch.randn(1,6,384,768)


net=model.DispNet()

# initialize weights(kaiming_normal) and bias(constant:0)
# net.weight_bias_init()




net(test_tensor)