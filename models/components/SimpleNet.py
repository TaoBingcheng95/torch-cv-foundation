import torch
from torch import nn
from torchinfo import summary


class Encoder(nn.Module):
    def __init__(self, input_size: int = 784, hidden_size: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, hidden_size: int = 256, output_size: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size: int = 256, output_size: int = 10):
        super().__init__()
        self.layer = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class SegmentationHead(nn.Module):
    def __init__(self, hidden_size: int = 256, output_size: int = 784):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()  # Assuming binary segmentation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class ModernDenseNet(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 256,
        output_size: int = 10,
        segmentation_output_size: int = 784
    ) -> None:
        super().__init__()

        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(hidden_size, hidden_size)
        self.classification_head = ClassificationHead(hidden_size, output_size)
        self.segmentation_head = SegmentationHead(hidden_size, segmentation_output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)

        x = self.encoder(x)
        x = self.decoder(x)

        # Determine the task based on the shape of the input
        if x.size(1) == 784:  # Assume 784 indicates a segmentation task
            return self.segmentation_head(x)
        else:
            return self.classification_head(x)


class SimpleDenseNet(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ) -> None:
        """
        Initialize a `SimpleDenseNet` modules.

        :param input_size: The number of input features.
        :param lin1_size: The number of output features of the first linear layer.
        :param lin2_size: The number of output features of the second linear layer.
        :param lin3_size: The number of output features of the third linear layer.
        :param output_size: The number of output features of the final linear layer.
        """
        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(),

            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            
            nn.ReLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        batch_size, channels, width, height = x.size()
        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)
        return self.model(x)



########################################


class OriginUnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # parameters: in_channels, out_channels, kernel_size, padding
        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)

        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64 * 2, 32, 3, 1)
        self.upconv1 = self.expand_block(32 * 2, out_channels, 3, 1)


    def forward(self, x):
        try:
            # downsampling part
            conv1 = self.conv1(x)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)

            upconv3 = self.upconv3(conv3)
            upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
            upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

            return upconv1
        except Exception as e:
            print(f"Error in forward pass: {e}")
            raise


    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            *self._common_layers(in_channels, out_channels, kernel_size, padding),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(
            *self._common_layers(in_channels, out_channels, kernel_size, padding),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        return expand

    @staticmethod
    def _common_layers(in_channels, out_channels, kernel_size, padding):
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]


class ContractBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ContractBlock, self).__init__()
        self.contract = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.contract(x)


class ExpandBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ExpandBlock, self).__init__()
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.expand(x)


class UNetEncoder(nn.Module):
    def __init__(self, in_channels):
        super(UNetEncoder, self).__init__()
        self.enc1 = ContractBlock(in_channels, 64, 7, 3)
        self.enc2 = ContractBlock(64, 128, 3, 1)
        self.enc3 = ContractBlock(128, 256, 3, 1)
        self.enc4 = ContractBlock(256, 512, 3, 1)
        self.bottleneck = ContractBlock(512, 1024, 3, 1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        bottleneck = self.bottleneck(enc4)
        return enc1, enc2, enc3, enc4, bottleneck


class UNetDecoder(nn.Module):
    def __init__(self):
        super(UNetDecoder, self).__init__()
        self.dec1 = ExpandBlock(1024, 512, 3, 1)
        self.dec2 = ExpandBlock(512 * 2, 256, 3, 1)
        self.dec3 = ExpandBlock(256 * 2, 128, 3, 1)
        self.dec4 = ExpandBlock(128 * 2, 64, 3, 1)

    def forward(self, enc1, enc2, enc3, enc4, bottleneck):
        dec1 = self.dec1(bottleneck)
        dec1 = torch.cat((enc4, dec1), dim=1)

        dec2 = self.dec2(dec1)
        dec2 = torch.cat((enc3, dec2), dim=1)

        dec3 = self.dec3(dec2)
        dec3 = torch.cat((enc2, dec3), dim=1)

        dec4 = self.dec4(dec3)
        dec4 = torch.cat((enc1, dec4), dim=1)

        return dec4


class UNetSegmentationTaskHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetSegmentationTaskHead, self).__init__()
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.final_conv(x)


class UNetClassificationTaskHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNetClassificationTaskHead, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(SimpleUNet, self).__init__()
        self.encoder = UNetEncoder(in_channels)
        self.decoder = UNetDecoder()
        self.task_head = UNetSegmentationTaskHead(128, num_classes)
        self.classification_head = UNetClassificationTaskHead(1024, num_classes)  # 分类头输入通道数为1024

    def forward(self, x):
        enc1, enc2, enc3, enc4, bottleneck = self.encoder(x)
        dec4 = self.decoder(enc1, enc2, enc3, enc4, bottleneck)
        segmentation_output = self.task_head(dec4)
        # classification_output = self.classification_head(bottleneck)
        return segmentation_output # segmentation_output, classification_output



def unet_demo():
    input_size = (1, 3, 512, 512)
    model = SimpleUNet(in_channels=3, num_classes=5) 
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

def densenet_demo():
    input_size = (1, 1, 28, 28)
    model = ModernDenseNet()
    model.eval()
    input_data = torch.randn(input_size)
    output = model(input_data)
    print(output.shape)
    # summary(model, 
    #         input_size=input_size,
    #         col_width=20,
    #         col_names=['input_size', 'output_size', 'num_params', 'trainable'], 
    #         row_settings=['var_names'], 
    #         verbose=True
    #         )



if __name__ == "__main__":
    # unet_demo()

    densenet_demo()

    # # model = SimpleDenseNet()
    # model = ModernDenseNet()
    # input_size = (1, 1, 28, 28)
    # summary(model, input_size=input_size)

    # model.eval()
    # inputs = torch.randn(input_size)
    # output = model(inputs)
    # print(output.shape)

    # input_size = (1, 3, 512, 512)
    # model = SimpleUNet(3, 1, num_classes=2) # 3个输入通道（RGB图像），1个输出通道（二分类）

    # summary(model, input_size=input_size)

    # # inputs = torch.randn(input_size)
    # # output = model(inputs)
    # # print(inputs.shape, ' -> ', output.shape)


