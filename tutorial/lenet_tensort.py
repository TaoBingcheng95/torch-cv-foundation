# # https://docs.pytorch.org/TensorRT/_notebooks/lenet-getting-started.html
# https://docs.pytorch.org/TensorRT/tutorials/notebooks.html
import time
import numpy as np

import torch
import torch.backends.cudnn as cudnn

cudnn.benchmark = True


def benchmark(model, input_shape=(1024, 1, 32, 32), dtype='fp32', nwarmup=50, nruns=1000):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    if dtype=='fp16':
        input_data = input_data.half()
    
    model.to("cuda")

    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()

    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns+1):
            start_time = time.time()
            features = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i%100==0:
                print('Iteration %d/%d, ave batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))

    print("Input shape:", input_data.size())
    print("Output features size:", features.size())
    print('Average batch time: %.2f ms'%(np.mean(timings)*1000))



if __name__ == '__main__':
    from models import LeNet5
    
    model = LeNet5()
    model.eval()
    # benchmark(model)
    traced_model = torch.jit.trace(model.to("cuda"), torch.empty([1,1,32,32]).to("cuda"))
    # print(traced_model)

    # benchmark(traced_model)

    script_model = torch.jit.script(model)


