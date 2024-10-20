import os
import torch
# 设置 CUDA 可见设备
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def check_cuda_availability():
    try:
        if torch.cuda.is_available():
            print("CUDA is available")
            print("GPU Numbers : ", torch.cuda.device_count())
            print("current device : ", torch.cuda.current_device())
            for i in range(torch.cuda.device_count()):
                print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
        else:
            print("CUDA is not available")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    print('torch version : ', torch.__version__)
    print('cuda version : ', torch.version.cuda)
    print('cudnn version : ', torch.backends.cudnn.version())

    check_cuda_availability()
