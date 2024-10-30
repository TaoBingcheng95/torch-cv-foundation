import os
import numpy as np
from dataset.my_dl import JiageDataset, WHDLDDataset, TianchiDataset

if __name__ == '__main__':
    jiage_dir = '/data/tbc/segmentation/jiage_building/'
    WHDLD_dir = '/data/tbc/segmentation/WHDLD'
    Tianchi_dir = "/data/tbc/segmentation/tianchi" #'D:\\myspace\\dataset\\segemnt\\tianchi' # 
    
    # dl = WHDLDDataset(root=WHDLD_dir)
    # dl = JiageDataset(root=jiage_dir)
    dl = TianchiDataset(root=Tianchi_dir)

    
    x, y = next(iter(dl))
    print(len(dl))
    print(x.shape, y.shape)
    print(x.dtype, y.dtype)
    dl.plot(save=True)
    # classes = []
    # for i in range(len(dl)):
    #     x, y = dl[i]
    #     tmp = np.unique(y)
    #     classes.extend(tmp)
    # class_num = set(classes)
    # print(class_num)
