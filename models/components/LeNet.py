
import torch
from torch import nn
import torch.nn.functional as F


class SimpleLeNet(nn.Module):
    """
    (N, 1, 32, 32) -> (N, num_classes)
    Total params: 61,706
    Trainable params: 61,706
    Non-trainable params: 0
    Total mult-adds (Units.MEGABYTES): 0.42
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.05
    Params size (MB): 0.25
    Estimated Total Size (MB): 0.30
    """
    def __init__(self, num_classes: int=10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FeatExtractor
        # Convolution layer C1: 1 input image channel, 6 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a Tensor with size (N, 6, 28, 28), where N is the size of the batch
        c1 = F.relu(self.conv1(x))   # input(1, 32, 32) output(16, 28, 28)
        # Subsampling layer S2: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 6, 14, 14) Tensor
        s2 = self.pool1(c1)          # output(16, 14, 14)
        # Convolution layer C3: 6 input channels, 16 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a (N, 16, 10, 10) Tensor
        c3 = F.relu(self.conv2(s2))  # output(32, 10, 10)
        # Subsampling layer S4: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 16, 5, 5) Tensor
        s4 = self.pool2(c3)          # output(32, 5, 5)

        # Flatten operation: purely functional, outputs a (N, 400) Tensor
        s4 = torch.flatten(s4, 1)    # 转化为一维向量 output(32*5*5)

        # Classifier
        # Fully connected layer F5: (N, 400) Tensor input,
        # and outputs a (N, 120) Tensor, it uses RELU activation function
        f5 = F.relu(self.fc1(s4))    # output(120)
        # Fully connected layer F6: (N, 120) Tensor input,
        # and outputs a (N, 84) Tensor, it uses RELU activation function
        f6 = F.relu(self.fc2(f5))    # output(84)
        # Gaussian layer OUTPUT/F7: (N, 84) Tensor input, and
        # outputs a (N, 10) Tensor
        x = self.fc3(f6)             # output(10)
        return x 



class LeNet5(nn.Module):
    """
    Total params: 61,750
    Trainable params: 61,750
    Non-trainable params: 0
    Total mult-adds (Units.MEGABYTES): 0.42
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.10
    Params size (MB): 0.25
    Estimated Total Size (MB): 0.35
    """
    def __init__(self, num_classes: int=10) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0), # C1 : 1->6, 32x32 -> 28x28
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)) # S2 : 28x28 -> 14x14
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0), # C3 : 6->16, 14x14 -> 10x10
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)) # S4 : 10x10 -> 5x5
        # an affine operation: y = Wx + b
        self.fc = nn.Linear(16*5*5, 120) # C5
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(120, 84) # C6
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(84, num_classes) # C7

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FeatExtractor
        out = self.layer1(x) # [C1, S2] (1, 32, 32) -> (6, 28, 28) -> (6, 14, 14)
        out = self.layer2(out) # [C3, S4] (6, 14, 14) -> (16, 10, 10) -> (16, 5, 5)
        # Classifier
        out = torch.flatten(out, 1)
        out = self.fc(out) # C5  转化为一维向量 output(32*5*5)
        out = self.relu(out)
        out = self.fc1(out) # F6 output(84)
        out = self.relu1(out)
        out = self.fc2(out) # F7 output(10)
        return out



if __name__ == '__main__':

    try:
        from torchinfo import summary
    except ImportError as e:
        print(e)
        exit()

    model = LeNet5()
    # print(model)

    input_size=(16, 1, 32, 32)
    # input_data = torch.randn(input_size)
    # output = model(input_data)
    # print(output.shape)

    summary(model,
            input_size=input_size,
            col_width=20,
            col_names=['input_size', 'output_size', 'num_params', 'trainable'],
            row_settings=['var_names'],
            verbose=True
            )
