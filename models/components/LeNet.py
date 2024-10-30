import torch
from torch import nn
import torch.nn.functional as F
try:
    from torchinfo import summary
except ImportError as e:
    print(e)


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
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=padding),   # padding=2 for 28*28->32*32-->28*28
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


if __name__ == '__main__':
    model = LeNet5()
    input_size = (1, 1, 32, 32)
    summary(model, 
            input_size=input_size, 
            col_width=20,
            col_names=['input_size', 'output_size', 'num_params', 'trainable'], 
            row_settings=['var_names'], 
            verbose=True
            )
