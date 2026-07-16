
import torch
from torchinfo import summary
from torchvision.models.segmentation import fcn
from torchvision.models import vgg16, VGG16_Weights

from models import LeNet5, build_vgg

def check_lenet():
    model = LeNet5()
    input_size = (1, 1, 32, 32)
    # input_data = torch.rand(input_size)
    # out = model(input_data)
    # print(out.shape)

    summary(model, input_size=input_size)



if __name__ == '__main__':
    # model = build_vgg()
    # print(model)

    pretrained_model = vgg16(weights=VGG16_Weights.DEFAULT)
    print(type(pretrained_model.features))
