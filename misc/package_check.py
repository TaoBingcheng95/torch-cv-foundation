# https://lightning.ai/docs/pytorch/latest/versioning.html#compatibility-matrix
import os
import numpy
import scipy
import torch


# 设置 CUDA 可见设备
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
        import transformers
        import albumentations
        print(f"pytorch-lightning Version : {lightning.__version__}")
        print(f"albumentations Version : {albumentations.__version__}")
        print(f"transformers Version : {transformers.__version__}")
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
