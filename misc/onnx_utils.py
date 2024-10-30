import onnxruntime as ort
import numpy as np
from matplotlib import pyplot as plt

from dataset.my_dl import TianchiDataset, NAIPDataset, JiageDataset, WHDLDDataset


def inference_onnx(onnx_fn, x, y, plot=False):
    # import onnxruntime as ort
    ort_session = ort.InferenceSession(onnx_fn)

    # input_data = np.random.randn(1, 3, 512, 512).astype(np.float32)
    input_data = x[np.newaxis,:].astype(np.float32)

    # 准备输入字典
    inputs = {ort_session.get_inputs()[0].name: input_data}
    # 进行推理
    outputs = ort_session.run(None, inputs)
    perd = np.argmax(outputs[0].squeeze(0),axis=0)
    
    if plot:
        fig, axis = plt.subplots(1,3)
        axis[0].imshow(x.transpose((1,2,0)))
        axis[1].imshow(y)
        axis[2].imshow(perd)
        plt.savefig('t.png')


if __name__ == '__main__':

    Tianchi_dir = "/data/tbc/segmentation/tianchi"
    ds = TianchiDataset(root=Tianchi_dir)
    x, y = next(iter(ds))
    
    # ort_session = ort.InferenceSession('your_model.onnx')

    # # 创建示例输入数据
    # # 替换 (1, 3, 224, 224) 为你的模型的实际输入尺寸
    # # input_data = np.random.randn(1, 3, 512, 512).astype(np.float32)
    # input_data = x[np.newaxis,:].astype(np.float32)

    # # 准备输入字典
    # inputs = {ort_session.get_inputs()[0].name: input_data}

    # # 进行推理
    # outputs = ort_session.run(None, inputs)

    # # 获取输出
    # print(type(outputs[0]))
    # print("模型输出：", outputs[0].shape)
    # perd = np.argmax(outputs[0].squeeze(0),axis=0)
    # print(perd.shape)
    # fig, axis = plt.subplots(1,3)
    # axis[0].imshow(x.transpose((1,2,0)))
    # axis[1].imshow(y)
    # axis[2].imshow(perd)
    # plt.savefig('t.png')

    
