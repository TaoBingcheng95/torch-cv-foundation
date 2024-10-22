# from torchvision.models import resnet50
import torch
from torch import nn
from torchinfo import summary


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

class UNetTaskHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetTaskHead, self).__init__()
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.final_conv(x)

class UNetClassificationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNetClassificationHead, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

#     def __init__(self, in_channels, out_channels, num_classes):
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(UNet, self).__init__()
        self.encoder = UNetEncoder(in_channels)
        self.decoder = UNetDecoder()
        self.task_head = UNetTaskHead(128, out_channels)
        self.classification_head = UNetClassificationHead(1024, num_classes)  # 分类头输入通道数为1024

    def forward(self, x):
        enc1, enc2, enc3, enc4, bottleneck = self.encoder(x)
        dec4 = self.decoder(enc1, enc2, enc3, enc4, bottleneck)
        segmentation_output = self.task_head(dec4)
        # classification_output = self.classification_head(bottleneck)
        return segmentation_output # segmentation_output, classification_output



if __name__ == '__main__':
    input_size = (1, 3, 512, 512)
    model = UNet(3, 1, num_classes=2) # 3个输入通道（RGB图像），1个输出通道（二分类）

    summary(model, input_size=input_size)

    # inputs = torch.randn(input_size)
    # output = model(inputs)
    # print(inputs.shape, ' -> ', output.shape)


