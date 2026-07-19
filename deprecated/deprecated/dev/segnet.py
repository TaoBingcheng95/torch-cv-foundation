
import torch
import torch.nn as nn
# import torch.nn.functional as F


class SegNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=21):
        super(SegNet, self).__init__()
        
        # Encoder: Convolutional layers with max pooling
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),  # Conv1
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Conv2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),  # MaxPool1
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Conv3
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Conv4
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),  # MaxPool2
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Conv5
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Conv6
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)  # MaxPool3
        )
        
        # Decoder: MaxUnpooling layers followed by convolution
        self.decoder = nn.Sequential(
            nn.MaxUnpool2d(kernel_size=2, stride=2),  # MaxUnpool3
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Conv6
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # Conv5
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.MaxUnpool2d(kernel_size=2, stride=2),  # MaxUnpool2
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Conv4
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # Conv3
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.MaxUnpool2d(kernel_size=2, stride=2),  # MaxUnpool1
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Conv2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),  # Conv1
        )
    
    def forward(self, x):
        indices_list = []
        sizes_list = []
        
        # Encoder forward pass
        for layer in self.encoder:
            if isinstance(layer, nn.MaxPool2d):
                sizes_list.append(x.size())
                x, indices = layer(x)
                indices_list.append(indices)
            else:
                x = layer(x)
        
        # Decoder forward pass
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, nn.MaxUnpool2d):
                x = layer(x, indices_list.pop(), output_size=sizes_list.pop())
            else:
                x = layer(x)
        
        return x


if __name__ == '__main__':
    num_classes = 21
    model = SegNet(in_channels=3, out_channels=num_classes)  # For a single-channel output (e.g., binary segmentation)
    
    x = torch.randn((1, 3, 256, 256))  # Example input tensor (batch_size, channels, height, width)
    output = model(x)
    print(output.shape)  # Should be (1, 1, 256, 256) for this example
