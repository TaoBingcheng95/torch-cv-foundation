# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
# https://medium.com/correll-lab/building-a-vision-transformer-model-from-scratch-a3054f707cc6
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import timm
# import segmentation_models_pytorch as smp
# from tqdm import tqdm
# import numpy as np
# from pprint import pprint

class ViT(nn.Module):
    def __init__(self, out_channels, vit_model='vit_base_patch16_224', pretrained=False):
        super(ViT, self).__init__()

        # 加载预训练的 ViT 模型
        self.vit = timm.create_model(vit_model, pretrained=pretrained,
                                     # num_classes=out_channels
                                     )

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

        # self.to_latent = nn.Identity()
        # self.mlp_head = nn.Linear(vit_features, num_classes)

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
        print(vit_output.shape)

        # Decoder: 恢复到原始图像尺寸
        segmentation_output = self.decoder(vit_output)

        # 将输出调整为原始图像的尺寸
        output = F.interpolate(segmentation_output, size=shape, mode='bilinear', align_corners=False)
        return output


####################


class PatchEmbedding(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, n_channels:int=3,
                 embed_dim:int=768,
                 img_size:int=244,
                 patch_size:int=16,
                 norm_layer:bool=False):
        """
        n_channels: 输入图像的通道数（例如，对于RGB图像，通道数为3）
        embed_dim: 模型的维度，即每个小块（patch）映射后的特征向量的大小。
        img_size: 输入图像的尺寸，通常是一个包含高度和宽度的元组或列表（虽然在这段代码中未直接使用）。
        patch_size: 每个小块的大小，通常是一个包含高度和宽度的元组或列表。
        norm_layer : 是否采用LayerNorm
        """
        super().__init__()
        self.embed_dim = embed_dim  # Dimensionality of Model
        self.img_size = img_size  # Image Size
        self.patch_size = patch_size  # Patch Size
        self.n_channels = n_channels  # Number of Channels
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        '''
        使用大小为16，stride为16的卷积核实现embedding，
        输出14*14大小，通道为768（768 = 16*16*3，相当于将每个patch部分转换为1维向量）的patch
        '''
        # 初始化线性投影层
        self.linear_project = nn.Conv2d(self.n_channels, self.embed_dim,
                                        kernel_size=self.patch_size,
                                        stride=self.patch_size)
        '''
        如果norm_layer为true则使用layerNorm，这里作者没有使用，
        所以self.norm = nn.Identity()，对输入不做任何改变直接输出
        '''
        self.norm = nn.LayerNorm(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """
        B: Batch Size
        C: Image Channels
        H: Image Height
        W: Image Width
        P_col: Patch Column
        P_row: Patch Row
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."

        '''
        self.proj(x):[B,3,224,224]->[B,768,14,14]
        flatten(2):[B,768,14,14]->[B,768,14*14]=[B,768,196]
        transpose(1, 2):[B,768,196]->[B,196,768]
        self.norm(x)不对输入做处理直接输出
        '''
        x = self.linear_project(x)  # (B, C, H, W) -> (B, d_model, P_col, P_row)
        x = x.flatten(2)  # (B, d_model, P_col, P_row) -> (B, d_model, P)
        x = x.transpose(1, 2)  # (B, d_model, P) -> (B, P, d_model)
        x = self.norm(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_length):
        """
        embed_dim: 模型的维度，即嵌入向量的大小。
        max_seq_length: 序列的最大长度。
        """
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim)) # Classification Token
        # Creating positional encoding
        # 可学习的位置编码
        self.positional_embedding = nn.Embedding(max_seq_length, embed_dim)
        # 初始化位置编码（可选，PyTorch 默认会进行初始化）
        # nn.init.normal_(self.positional_embedding.weight, mean=0.0, std=d_model**-0.5)
        # pe = torch.zeros(max_seq_length, d_model)
        # for pos in range(max_seq_length):
        #     for i in range(d_model):
        #         if i % 2 == 0:
        #             pe[pos][i] = np.sin(pos/(10000 ** (i/d_model)))
        #         else:
        #             pe[pos][i] = np.cos(pos/(10000 ** ((i-1)/d_model)))
        # self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        x: 输入序列的嵌入表示，其形状通常为 (batch_size, seq_length, d_model)。
        """
        batch_size, seq_length, _ = x.size()
        # 创建位置索引，形状为 (batch_size, seq_length)
        positions = torch.arange(0, seq_length, device=x.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        # 获取位置编码，形状为 (batch_size, seq_length, d_model)
        positional_encodings = self.positional_embedding(positions)
        # Create a special positional encoding for cls_token (could be zero or any other special value)
        cls_positional_encoding = self.positional_embedding(torch.tensor([0], device=x.device, dtype=torch.long)).unsqueeze(0).expand(batch_size, 1, -1)

        # Expand to have class token for every image in batch
        tokens_batch = self.cls_token.expand(batch_size, -1, -1)
        # Adding class tokens to the beginning of each embedding
        x = torch.cat((tokens_batch,x), dim=1)
        # Add positional encoding to embeddings
        # x = x + self.pe
        # Add positional encodings (including special one for cls_token)
        x = x + torch.cat((cls_positional_encoding, positional_encodings), dim=1)
        return x


class AttentionHead(nn.Module):
    def __init__(self, embed_dim:int=768,
                 head_size:int=64,
                 qkv_bias:bool=False,
                 qk_scale=None,
                 attn_drop_ratio:float=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_size = head_size
        self.scale = qk_scale or head_size ** -0.5
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.query = nn.Linear(self.embed_dim, self.head_size, bias=qkv_bias)
        self.key = nn.Linear(self.embed_dim, self.head_size, bias=qkv_bias)
        self.value = nn.Linear(self.embed_dim, self.head_size, bias=qkv_bias)

    def forward(self, x):
        # Obtaining Queries, Keys, and Values
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        # Dot Product of Queries and Keys
        attention = Q @ K.transpose(-2,-1)
        # Scaling
        attention = attention / self.scale # (self.head_size ** 0.5)
        attention = torch.softmax(attention, dim=-1)
        attention = self.attn_drop(attention)
        attention = attention @ V
        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12,
                 qkv_bias:bool=False,
                 qk_scale=None,
                 attn_drop_ratio:float=0.,
                 proj_drop_ratio:float=0.):
        """
        num_heads = 12
        head_dim = 768 // 12 = 64 （Attention is all you need论文中提到的dk=dv=dmodel/h）
        scale = 64 ^ -0.5 = 1/8（Attention is all you need论文中Scaled Dot-Product Attention提到的公式Attention(Q,K,V)中的根号dk分之一）
        qkv:将输入线性映射到q,k,v
        proj：Attention is all you need论文中Multi-Head Attention最后的融合矩阵 Wo，使用 Linear 的实现
        """
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.attn_drop_ratio = attn_drop_ratio
        self.proj_drop_ratio = proj_drop_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale

        self.head_size = embed_dim // num_heads
        self.W_o = nn.Linear(self.embed_dim, self.embed_dim) # self.proj
        self.proj_drop = nn.Dropout(proj_drop_ratio)

        # attn_head = AttentionHead(embed_dim=self.embed_dim,
        #                           head_size=self.head_size,
        #                           qk_scale=self.qk_scale,
        #                           qkv_bias=self.qkv_bias,
        #                           attn_drop_ratio=attn_drop_ratio)
        # self.heads = nn.ModuleList([attn_head] * n_heads)
        self.heads = nn.ModuleList([AttentionHead(embed_dim=self.embed_dim, head_size=self.head_size, ) for _ in range(num_heads)])


    def forward(self, x):
        """
        B = batch_size
        N = 197
        C = 768
        """
        # Combine attention heads
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.W_o(out)
        x = self.proj_drop(x)
        return x


class MHA(nn.Module):
    def __init__(self,
                 embed_dim=768,  # 输入token的dim 768
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(MHA, self).__init__()
        '''
        num_heads = 12
        head_dim = 768 // 12 = 64 （Attention is all you need论文中提到的dk=dv=dmodel/h）
        scale = 64 ^ -0.5 = 1/8（Attention is all you need论文中Scaled Dot-Product Attention提到的公式Attention(Q,K,V)中的根号dk分之一）
        qkv:将输入线性映射到q,k,v
        proj：Attention is all you need论文中Multi-Head Attention最后的融合矩阵 Wo，使用 Linear 的实现
        '''
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        """
        B = batch_size
        N = 197
        C = 768
        """
        B, N, C = x.shape

        '''
        qkv(x) : [B,197,768] -> [B,197,768*3]
        reshape : [B,197,768*3] -> [B,197,3,12,64] (3分别代表qkv，12个head，每个head为64维向量)
        permute：[B,197,3,12,64] -> [3,B,12,197,64]
        '''

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        '''
        q,k,v = [B,12,197,64]
        '''
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        '''
        K.transpose(-2, -1) : [B,12,197,64] = [B,12,64,197]
        q @ K.transpose(-2, -1) : [B,12,197,64] @ [B,12,64,197] = [B,12,197,197]
        attn : [B,12,197,197]
        attn.softmax(dim=-1)对最后一个维度（即每一行）进行softmax处理
        '''
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        '''
        attn @ v = [B,12,197,197] @ [B,12,197,64] = [B,12,197,64]
        transpose(1, 2) : [B,197,12,64]
        reshape : [B,197,768]
        '''
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, r_mlp:int=4, drop:float=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # Sub-Layer 1 Normalization
        self.ln1 = nn.LayerNorm(embed_dim)
        # Multi-Head Attention
        # self.mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.mha = MHA(embed_dim=embed_dim, num_heads=num_heads)
        # Sub-Layer 2 Normalization
        self.ln2 = nn.LayerNorm(embed_dim)
        # Multilayer Perception
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*r_mlp),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim*r_mlp, embed_dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        # Residual Connection After Sub-Layer 1
        out = x + self.mha(self.ln1(x))
        # Residual Connection After Sub-Layer 2
        out = out + self.mlp(self.ln2(out))
        return out


class VisionTransformerScratch(nn.Module):
    def __init__(self,
                 n_channels=1,
                 n_classes=1000,
                 embed_dim=768,
                 img_size=224,
                 patch_size=16,
                 num_heads=12,
                 num_layers=12):
        """
        embed_dim: 模型的维度，即每个小块（patch）映射后的特征向量的大小。
        img_size: 输入图像的尺寸，通常是一个包含高度和宽度的元组或列表（虽然在这段代码中未直接使用）。
        patch_size: 每个小块的大小，通常是一个包含高度和宽度的元组或列表。
        n_channels: 输入图像的通道数（例如，对于RGB图像，通道数为3）
        num_heads : 所使用的注意力头数
        num_layers : TransformerEncoder数量
        """
        super().__init__()

        assert img_size % patch_size == 0 , "img_size dimensions must be divisible by patch_size dimensions"
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by n_heads"

        self.embed_dim = embed_dim  # Dimensionality of model
        self.n_classes = n_classes  # Number of classes
        self.img_size = img_size  # Image size
        self.patch_size = patch_size  # Patch size
        self.n_channels = n_channels  # Number of channels
        self.num_heads = num_heads  # Number of attention heads
        self.num_layers = num_layers

        self.n_patches = (self.img_size * self.img_size) // (self.patch_size * self.patch_size)
        self.max_seq_length = self.n_patches + 1

        self.patch_embedding = PatchEmbedding(embed_dim=self.embed_dim,
                                              img_size=self.img_size,
                                              patch_size=self.patch_size,
                                              n_channels=self.n_channels,
                                              norm_layer=False)
        self.positional_encoding = PositionalEncoding(embed_dim=self.embed_dim,
                                                      max_seq_length=self.max_seq_length)
        self.transformer_encoder = nn.Sequential(
            *[TransformerEncoder(embed_dim=self.embed_dim, num_heads=self.num_heads) for _ in range(self.num_layers)])

        # Classification MLP
        self.classifier = ClassificationHead(emb_size=self.embed_dim, n_classes=n_classes)

        # Segmentation Head
        self.segmentation_head = SegmentationHead(in_channels=self.embed_dim, out_channels=n_classes)


    def forward(self, images):
        shape = (images.shape[2], images.shape[3])
        x = self.patch_embedding(images)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)

        # classify task
        # classification_output = self.classifier(x[:, 0])

        # 移除 class token
        x = x[:, 1:, :]  # 移除 class token，保留 patches
        # 将ViT的输出 reshape 成 2D feature map
        B, N, C = x.shape  # N: Number of patches
        h = w = int((N ** 0.5))  # 计算height和width
        # 检查维度是否匹配 patch 的数量
        assert h * w == N, f"Patch size mismatch: expected {N} patches but got h={h}, w={w}"
        x = x.permute(0, 2, 1).reshape(B, C, h, w)
        # 分割任务
        segmentation_output = self.segmentation_head(x, shape)

        return segmentation_output


class ClassificationHead(nn.Module):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__()
        self.emb_size = emb_size
        self.n_classes = n_classes
        self.cls = nn.Sequential(
            nn.Linear(self.emb_size, self.n_classes),
            nn.Softmax(dim=-1)
        )
    def forward(self, inputs):
        x = self.cls(inputs)
        return x


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegmentationHead, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def forward(self, x, shape):
        # Decoder: 恢复到原始图像尺寸
        segmentation_output = self.decoder(x)
        # 将输出调整为原始图像的尺寸
        output = F.interpolate(segmentation_output, size=shape, mode='bilinear', align_corners=False)
        return output


def demo_1():
    num_classes = 21
    model = ViT(out_channels=num_classes,
                vit_model='vit_base_patch16_224',
                pretrained=False)

    # model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)

    input_size = (1, 3, 224, 224)
    x = torch.randn(input_size)
    # Forward pass
    output = model(x)
    print(output.shape)



if __name__ == "__main__":

    try:
        from torchinfo import summary
    except ImportError as e:
        print(e)

    num_classes = 1000 # 21
    # model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
    # # print(model.embed_dim) # 768 get_classifier 'num_classes', 'num_features',
    # pprint(model.default_cfg)

    model = ViT(out_channels=num_classes,
                vit_model='vit_base_patch16_224',
                pretrained=False)
    # model = VisionTransformerScratch(n_channels = 3,
    #                           n_classes=1000,
    #                           img_size = 224,
    #                           patch_size = 16,
    #                           embed_dim = 768,
    #                           num_heads = 12,
    #                           num_layers = 12)

    input_size = (1, 3, 224, 224)
    # x = torch.randn(input_size)
    # # Forward pass
    # output = model(x)
    # print(output.shape)

    summary(model,
            input_size=input_size,
            col_width=20,
            col_names=['input_size', 'output_size', 'num_params', 'trainable'],
            row_settings=['var_names'],
            verbose=True
            )
