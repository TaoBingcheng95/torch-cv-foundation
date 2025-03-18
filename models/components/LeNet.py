import torch
from torch import nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """
    support 28x28 MNIST data
    """
    def __init__(self,padding=0):
        """
        padding: padding size for the first convolution layer
        padding=2 for MNIST data 28*28->32*32
        """
        super().__init__()
        self.feature = nn.Sequential(
            #1
            # padding=2 for 28*28->32*32-->28*28
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=padding),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 14*14
            #2
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),  # 10*10
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 5*5
            
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10),
        )
        
    def forward(self, x):
        fc1 = self.feature(x)
        return self.classifier(fc1)


class Net(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
    """

    def __init__(self, in_channels: int = 1, out_features: int = 10):
        """
        1 input image channel, 6 output channels, 5x5 square convolution
        """
        super(Net, self).__init__()
        # kernel
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_features)

    def forward(self, inputs):
        # Convolution layer C1: 1 input image channel, 6 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a Tensor with size (N, 6, 28, 28), where N is the size of the batch
        c1 = F.relu(self.conv1(inputs))
        # Subsampling layer S2: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 6, 14, 14) Tensor
        s2 = F.max_pool2d(c1, (2, 2))
        # Convolution layer C3: 6 input channels, 16 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a (N, 16, 10, 10) Tensor
        c3 = F.relu(self.conv2(s2))
        # Subsampling layer S4: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 16, 5, 5) Tensor
        s4 = F.max_pool2d(c3, 2)
        # Flatten operation: purely functional, outputs a (N, 400) Tensor
        s4 = torch.flatten(s4, 1)
        # Fully connected layer F5: (N, 400) Tensor input,
        # and outputs a (N, 120) Tensor, it uses RELU activation function
        f5 = F.relu(self.fc1(s4))
        # Fully connected layer F6: (N, 120) Tensor input,
        # and outputs a (N, 84) Tensor, it uses RELU activation function
        f6 = F.relu(self.fc2(f5))
        # Gaussian layer OUTPUT: (N, 84) Tensor input, and
        # outputs a (N, 10) Tensor
        outputs = self.fc3(f6)
        return outputs



if __name__ == '__main__':

    try:
        from torchinfo import summary
    except ImportError as e:
        print(e)
        exit()

    model = LeNet5() # Net()
    # print(net)

    input_size=(1, 1, 32, 32)
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
