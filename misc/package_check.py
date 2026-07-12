# https://lightning.ai/docs/pytorch/latest/versioning.html#compatibility-matrix
import os
import numpy
import scipy
import torch

# 设置 CUDA 可见设备
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

"""
CUDA 12.1

numpy Version : 1.24.3
scipy Version : 1.1.0.1
pytorch Version : 2.3.1
Scikit-image Version : 0.24.0
pytorch-lightning : 2.4.0
transformers Version : 4.49.0
albumentations Version : 1.3.1
kornia Version : 0.8.0
ultralytics Version : 8.3.91
transformers Version : 4.49.0
"""

def check_cuda_availability():
    try:
        if torch.cuda.is_available():
            print("CUDA is available")
            print("GPU Numbers : ", torch.cuda.device_count())
            print("Current Device : ", torch.cuda.current_device())
            for i in range(torch.cuda.device_count()):
                print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
        else:
            print("CUDA is not available")
    except Exception as e:
        print(f"Error: {e}")


def check_cudnn_availability():
    # 检查 cuDNN 是否可用
    cudnn_available = torch.backends.cudnn.is_available()
    print(f"cuDNN available: {cudnn_available}")
    if cudnn_available:
        cudnn_version = torch.backends.cudnn.version()
        print(f"cuDNN version: {cudnn_version}")
    else:
        print("cuDNN is not available. Please ensure that CUDA and cuDNN are properly installed and configured.")


def extent_package():
    try:
        import lightning
        print(f"pytorch-lightning Version : {lightning.__version__}")
    except ImportError as e:
        print(e)

    try:
        import ultralytics
        print(f"ultralytics Version : {ultralytics.__version__}")
    except ImportError as e:
        print(e)

    try:
        import transformers
        print(f"transformers Version : {transformers.__version__}")
    except ImportError as e:
        print(e)

    try:
        import albumentations
        print(f"albumentations Version : {albumentations.__version__}")
    except ImportError as e:
        print(e)

    try:
        import kornia
        print(f"kornia Version : {kornia.__version__}")
    except ImportError as e:
        print(e)

    try:
        import segmentation_models_pytorch as smp
        import timm
        print(f"segmentation_models_pytorch Version : {smp.__version__}")
        print(f"timm Version : {timm.__version__}")
    except ImportError as e:
        print(e)



if __name__ == '__main__':
    print(f"Numpy Version : {numpy.__version__}")
    print(f"SciPy Version : {scipy.__version__}")
    print(f"PyTorch version : {torch.__version__}")
    # print('CUDA version : ', torch.version.cuda)
    # print('CuDnn version : ', torch.backends.cudnn.version())
    # print(torch.__config__.show())

    check_cuda_availability()
    check_cudnn_availability()
    extent_package()
