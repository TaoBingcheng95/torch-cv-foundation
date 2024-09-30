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

print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())

check_cuda_availability()
