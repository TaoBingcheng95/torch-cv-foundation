import torch
from torch import nn

class DenseBlock(nn.Module):
    # 默认不使用瓶颈层，即使用DenseNet-A模型
    def __init__(self, num_input_chennel, growth_rate, drop_rate, bottleneck=0):
        super().__init__()
        self.bottle = bottleneck
        if bottleneck:
            self.bottleneck = growth_rate * bottleneck
        else:
            self.bottleneck = num_input_chennel
        self.norm1 = nn.BatchNorm2d(int(num_input_chennel))
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(int(num_input_chennel), int(self.bottleneck), kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm2d(int(self.bottleneck))
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(int(self.bottleneck), int(growth_rate), kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = float(drop_rate)

    def bn_function(self, inputs):
        concated_features = torch.cat(inputs, 1)
        if self.bottle:
            bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
            return bottleneck_output
        else:
            return concated_features
    
    def forward(self,x):
        x = self.bn_function(x)
        x = self.conv2(self.relu2(self.norm2(x))) 
        if self.drop_rate > 0.0:
            x = nn.Dropout(p=self.drop_rate)(x)
        return x
    
class DenseLayer(nn.ModuleDict):
    def __init__(self,num_layers,num_input_chennel,growth_rate,drop_rate,bottleneck):
        super().__init__()
        for i in range(num_layers):
            layer = DenseBlock(num_input_chennel + i * growth_rate,growth_rate=growth_rate,bottleneck=bottleneck,drop_rate=drop_rate)
            self.add_module(f"denselayer{(i + 1)}", layer) 
                
    def forward(self, x): 
        #torch.cat()函数将列表/元组中的元素按某个维度进行合并，因此每一层的结果加到列表中
        x_list = [x]
        for name ,layer in self.items():
            x = layer(x_list)
            x_list.append(x)
        return torch.cat(x_list,1)
    
class Transition(nn.Module):
    def __init__(self, num_input_chennel, theta):
        super().__init__()
        self.norm = nn.BatchNorm2d(int(num_input_chennel))
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(int(num_input_chennel),int(theta * num_input_chennel),kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(self.conv(self.relu(self.norm(x))))
        return x

class DenseNets(nn.Module):
    def __init__(self,dataset:str,
                 growth_rate:int,
                 denselayer_config:list,
                 drop_rate:float,
                 bottleneck:float=0,
                 compression:float=1):
        super().__init__()
        # 设置瓶颈层的调节通道数
        if bottleneck:
            bottleneck *= growth_rate
        # 原文献中提到，如果模型是 DenseNet-BC，则初始通道数是2k，否则是16
        if bottleneck and compression < 1:
            init_input_channel = 2*growth_rate
        else:
            init_input_channel = 16
        if dataset != 'ImageNet':
            # 输入为3*32*32，输出为(16 or 2k)*32*32
            conv1 = nn.Conv2d(3, init_input_channel, kernel_size=1, stride=1, padding=0, bias=False)
            # 这里不需要池化层
            self.densenet = nn.Sequential(conv1)
            # 根据配置列表，创建DenseLayer
            for i,num_layers in enumerate(denselayer_config):
                if i == 0:
                    num_input_chennel = init_input_channel
                else:
                    num_input_chennel = compression * num_input_chennel
                dense = DenseLayer(num_input_chennel=num_input_chennel,num_layers=num_layers,growth_rate=growth_rate,drop_rate=drop_rate,bottleneck = bottleneck)
                self.densenet.add_module(f"DenseLayer{(i + 1)}", dense)

                num_input_chennel += num_layers * growth_rate
                # 最后的DenseLayer后面不需要过渡层   
                if i != len(denselayer_config) - 1:
                    trans = Transition(num_input_chennel=num_input_chennel, theta = compression)
                    self.densenet.add_module(f"TransitionLayer{(i + 1)}", trans)
                # 如果是输入的是32*32尺寸的图片，经过4层DenseLayers后，最终输出大小为4*4
        else:
            # 输入为3*224*224，输出为(16 or 2k)*112*112
            conv1 = nn.Conv2d(3, init_input_channel, kernel_size=7, stride=2, padding=3, bias=False)
            # 输入为(16 or 2k)*112*112，输出为(16 or 2k)*56*56
            pooling1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.densenet = nn.Sequential(conv1, pooling1)

            for i,num_layers in enumerate(denselayer_config):
                if i == 0:
                    num_input_chennel = init_input_channel
                else:
                    num_input_chennel = compression * num_input_chennel
                dense = DenseLayer(num_input_chennel=num_input_chennel,num_layers=num_layers,growth_rate=growth_rate,drop_rate=drop_rate,bottleneck = bottleneck)
                self.densenet.add_module(f"DenseLayer{(i + 1)}", dense)

                num_input_chennel += num_layers * growth_rate
                 
                if i != len(denselayer_config) - 1:
                    trans = Transition(num_input_chennel=num_input_chennel, theta = compression)
                    self.densenet.add_module(f"TransitionLayer{(i + 1)}", trans)
                # 如果是输入的是224*224尺寸的图片，经过4层DenseLayers后，最终输出大小为7*7
        
        # 全局池化为1*1大小的张量
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        # 去掉后两个维度
        self.flatten = nn.Flatten()
        # 两层的MLP
        self.class_conv = nn.Linear(in_features=int(num_input_chennel),out_features=256,bias=True)
        self.acti = nn.ReLU()
        self.linear = nn.Linear(in_features=256,out_features=10,bias=True)   

        # 随机初始化模型参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x = self.densenet(x)
        print(x.shape)
        x = self.linear(self.acti(self.class_conv(self.flatten(self.global_pooling(x)))))
        return x
