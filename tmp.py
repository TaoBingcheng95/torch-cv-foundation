import platform

import numpy as np
import torch
import torch.nn as nn
from  torchvision import transforms



if __name__ == '__main__':
    
    ckpt_fn = 'checkpoints\\20260822_132422\\best.pt'
    ckpt = torch.load(ckpt_fn)
    print(ckpt.keys())
