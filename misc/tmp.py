import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import timm
import segmentation_models_pytorch as smp
from tqdm import tqdm
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        """
        d_model: 模型的维度，即嵌入向量的大小。
        max_seq_length: 序列的最大长度。
        """
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)) # Classification Token
        # Creating positional encoding
        # 可学习的位置编码
        self.positional_embedding = nn.Embedding(max_seq_length, d_model)
        # 初始化位置编码（可选，PyTorch 默认会进行初始化）
        nn.init.normal_(self.positional_embedding.weight, mean=0.0, std=d_model**-0.5)

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
        x = x + torch.cat((cls_positional_encoding, positional_encodings), dim=1)  # Add positional encodings (including special one for cls_token)

        return x
