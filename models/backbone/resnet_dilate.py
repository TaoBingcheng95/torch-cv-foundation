import torch
import torch.nn as nn

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


def conv3x3(in_channels, out_channels, stride=1, dilation=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=stride,padding=dilation,dilation=dilation,bias=False)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,stride=stride,bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,in_channels,out_channels,stride=1,downsample=None,dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels,out_channels,stride,dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels,out_channels,dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self,x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self,in_channels,out_channels,stride=1,downsample=None,dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_channels,out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels,out_channels,stride,dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv1x1(out_channels,out_channels*self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self,x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,block,layers,num_classes=1000,zero_init_residual=False,norm_layer=nn.BatchNorm2d,replace_stride_with_dilation=None):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.dilation = 1
        self.conv1 = nn.Conv2d(3,self.in_channels,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self._make_layer(block,64,layers[0])
        self.layer2 = self._make_layer(block,128,layers[1],stride=2,norm_layer=norm_layer,dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,256,layers[2],stride=2,norm_layer=norm_layer,dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,512,layers[3],stride=2,norm_layer=norm_layer,dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion,num_classes)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            if isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight,0)
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight,0)


    def _make_layer(self,block,in_channels,num_block,stride=1,norm_layer=nn.BatchNorm2d,dilate=False):
        pre_dilation = self.dilation
        downsample = None

        if dilate:
            self.dilation*= stride
            stride = 1

        if stride != 1 or in_channels*block.expansion != self.in_channels :
            downsample = nn.Sequential(
                conv1x1(self.in_channels,in_channels*block.expansion,stride),
                norm_layer(in_channels*block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels,in_channels,stride,downsample,dilation=pre_dilation))

        self.in_channels = in_channels*block.expansion
        for i in range(1,num_block):
            layers.append(block(self.in_channels,in_channels,dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        #不需要用到全连接层
        # x = self.avgpool(x)
        # x = x.view(x.size(0),-1)
        # x = self.fc(x)

        return x1,x2,x3,x4

def resnet18(**kwargs):
    model = ResNet(BasicBlock,[2,2,2,2],replace_stride_with_dilation=[False,True,True],**kwargs)
    return model

def resnet34(**kwargs):
    model = ResNet(BasicBlock,[3,4,6,4],replace_stride_with_dilation=[False,True,True],**kwargs)
    return model

def resnet50(**kwargs):
    model = ResNet(Bottleneck,[3,4,6,3],replace_stride_with_dilation=[False,True,True],**kwargs)
    return model

def resnet101(**kwargs):
    model = ResNet(Bottleneck,[3,4,23,3],replace_stride_with_dilation=[False,True,True],**kwargs)
    return model

def resnet152(**kwargs):
    model = ResNet(Bottleneck,[3,8,36,3],replace_stride_with_dilation=[False,True,True],**kwargs)
    return model

if __name__ == '__main__':
    model = resnet18()
    print(model)
    img = torch.randn(1, 3, 224, 224)
    output = model(img)
    # print(output.size())
