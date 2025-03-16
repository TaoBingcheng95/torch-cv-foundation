
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm



class ViT(nn.Module):
    def __init__(self, out_channels, vit_model='vit_base_patch16_224', pretrained=False):
        super(ViT, self).__init__()
        
        # 加载预训练的 ViT 模型
        self.vit = timm.create_model(vit_model, pretrained=pretrained)
        
        # 获取ViT的特征维度
        vit_features = self.vit.embed_dim
        
        # 获取 patch_size
        self.patch_size = self.vit.patch_embed.patch_size[0]
        # print(f"Patch size: {self.patch_size}")
        
        # 设置ViT的输入尺寸
        self.vit_img_size = self.vit.default_cfg['input_size'][1]  # ViT 默认输入尺寸 224x224
        # print(f"ViT input size: {self.vit_img_size}")
        
        # 删除ViT模型的分类头
        self.vit.head = nn.Identity()
        
        # Decoder: 将ViT的输出恢复到图像的尺寸
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(vit_features, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )
    
    def forward(self, x):
        shape = (x.shape[2], x.shape[3])
        # 输入图像调整为 ViT 需要的输入尺寸 (224x224)
        x = F.interpolate(x, size=(self.vit_img_size, self.vit_img_size), mode='bilinear', align_corners=False)
        
        # Forward through ViT
        vit_output = self.vit.forward_features(x)
        
        # 移除 class token
        vit_output = vit_output[:, 1:, :]  # 移除 class token，保留 patches
        
        # 将ViT的输出 reshape 成 2D feature map
        B, N, C = vit_output.shape  # N: Number of patches
        h = w = int((N ** 0.5))  # 计算height和width
        
        # 检查维度是否匹配 patch 的数量
        assert h * w == N, f"Patch size mismatch: expected {N} patches but got h={h}, w={w}"
        
        vit_output = vit_output.permute(0, 2, 1).reshape(B, C, h, w)
        
        # Decoder: 恢复到原始图像尺寸
        segmentation_output = self.decoder(vit_output)
        
        # 将输出调整为原始图像的尺寸
        return F.interpolate(segmentation_output, size=shape, mode='bilinear', align_corners=False)



if __name__ == "__main__":

    try:
        from torchinfo import summary
    except ImportError as e:
        print(e)
    
    num_classes = 21
    model = ViT(out_channels=num_classes, 
                vit_model='vit_base_patch16_224', 
                pretrained=False)

    input_size = (1, 3, 512, 512)
    # x = torch.randn(input_size)
    # Forward pass
    # output = model(x)
    # print(output.shape)

    summary(model, 
            input_size=input_size, 
            col_width=20,
            col_names=['input_size', 'output_size', 'num_params', 'trainable'], 
            row_settings=['var_names'], 
            verbose=True
            )
    