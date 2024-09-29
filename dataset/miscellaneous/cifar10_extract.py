import os
from skimage.io import imsave


def unpickle(fn):
    import pickle
    with open(fn, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


if __name__ == '__main__':

    # 训练集
    filename = 'cifar-10-batches-py'  # cifar10路径
    meta = unpickle(filename + '/batches.meta')
    label_name = meta[b'label_names']
    print(label_name)  # 打印标签

    for i in range(len(label_name)):  # 建立文件夹train
        file = label_name[i].decode()
        path = '../dataset_laoder/train/' + file
        os.makedirs(path, exist_ok=True)

    # 验证集
    for i in range(len(label_name)):
        file = label_name[i].decode()
        path = '../dataset_laoder/valid/' + file
        os.makedirs(path, exist_ok=True)
    class_dict = {'frog': 0,
                  'deer': 0,
                  'bird': 0,
                  'horse': 0,
                  'airplane': 0,
                  'cat': 0,
                  'dog': 0,
                  'ship': 0,
                  'truck': 0,
                  'automobile': 0}
    for i in range(1, 6):
        content = unpickle(filename + '/data_batch_' + str(i))  # 解压后的每个data_batch_
        for j in range(10000):
            img = content[b'data'][j]
            img = img.reshape(3, 32, 32)
            img = img.transpose(1, 2, 0)

            class_dict[label_name[content[b'labels'][j]].decode()] += 1
            if class_dict[label_name[content[b'labels'][j]].decode()] % 10 == 0:
                img_label = 'valid'
            else:
                img_label = 'train'

            img_name = '../dataset_laoder/' + img_label + '/' + label_name[content[b'labels'][j]].decode() + '/batch_' + str(
                i) + '_num_' + str(j) + '.jpeg'
            imsave(img_name, img)

    # 训练集改名
    path = '../dataset_laoder/train/'
    filelist = os.listdir(path)
    for item in filelist:
        pathnew = os.path.join(path, item)
        imagelist = os.listdir(pathnew)
        j = 1
        for i in imagelist:
            src = os.path.join(os.path.abspath(pathnew), i)
            dst = os.path.join(os.path.abspath(pathnew), '' + item + '.' + str(j) + '.jpeg')
            j += 1
            os.rename(src, dst)
    # 验证集改名
    path = '../dataset_laoder/valid/'
    filelist = os.listdir(path)
    for item in filelist:
        pathnew = os.path.join(path, item)
        imagelist = os.listdir(pathnew)
        j = 1
        for i in imagelist:
            src = os.path.join(os.path.abspath(pathnew), i)
            dst = os.path.join(os.path.abspath(pathnew), '' + item + '.' + str(j) + '.jpeg')
            j += 1
            os.rename(src, dst)

    # 测试集
    meta1 = unpickle(filename + '/test_batch')  # 解压test_batch
    label_name1 = meta[b'label_names']

    for i in range(len(label_name1)):
        file = label_name1[i].decode()
        path = '../dataset_laoder/test/' + file
        os.makedirs(path, exist_ok=True)

    for j in range(0, 10000):
        img = meta1[b'data'][j]
        img = img.reshape(3, 32, 32)
        img = img.transpose(1, 2, 0)

        img_name = '../dataset_laoder/test/' + label_name1[
            meta1[b'labels'][j]].decode() + '/batch_' + str(j) + '_num_' + str(j) + '.jpeg'
        imsave(img_name, img)

    # 测试集改名
    path = '../dataset_laoder/test/'
    filelist = os.listdir(path)
    for item in filelist:
        pathnew = os.path.join(path, item)
        imagelist = os.listdir(pathnew)
        j = 1
        for i in imagelist:
            src = os.path.join(os.path.abspath(pathnew), i)
            dst = os.path.join(os.path.abspath(pathnew), '' + item + '.' + str(j) + '.jpeg')
            j += 1
            os.rename(src, dst)