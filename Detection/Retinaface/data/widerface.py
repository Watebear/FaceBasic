import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
from random import shuffle
from data.data_augment import preprocess

class WiderFace(data.Dataset):
    def __init__(self, txt_path, preprocess=None):
        self.preprocess = preprocess
        self.txt_path = txt_path
        self.imgs_path, self.words  = self.process_labels()

    def process_labels(self):
        imgs_path = []
        words = []
        f =  open(self.txt_path, 'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()  # 删除结尾的空格
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    words.append(labels_copy)
                    labels.clear()  # 清除所元素
                path = line[2:]
                path = self.txt_path.replace('label.txt', 'images/') + path
                #print(path)
                imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)
        words.append(labels)
        return imgs_path, words

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape

        labels = self.words[index]
        annotations = np.zeros((0, 15))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

            # landmarks
            annotation[0, 4] = label[4]    # l0_x
            annotation[0, 5] = label[5]    # l0_y
            annotation[0, 6] = label[7]    # l1_x
            annotation[0, 7] = label[8]    # l1_y
            annotation[0, 8] = label[10]   # l2_x
            annotation[0, 9] = label[11]   # l2_y
            annotation[0, 10] = label[13]  # l3_x
            annotation[0, 11] = label[14]  # l3_y
            annotation[0, 12] = label[16]  # l4_x
            annotation[0, 13] = label[17]  # l4_y
            if (annotation[0, 4]<0):
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.preprocess is not None:
            img, target = self.preprocess(img, target)

        return torch.from_numpy(img), target

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)


if __name__ == "__main__":
    train_txt = 'G:/Datasets/face_detect/widerface/widerface-retinaface/train/label.txt'
    img_dim = 640
    rgb_mean = (104, 117, 123) # bgr order
    batch_size = 3
    num_workers = 4
    max_epoch = 10
    dataset = WiderFace(txt_path=train_txt, preprocess=preprocess(img_dim, rgb_mean))

    '''
    for i, data in enumerate(dataset):
        img, label = data
        print(np.shape(img))
        print(label)
        if i == 3:
            break

    '''
    import math
    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size
    start_iter = 0
    epoch = 0

    batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))


    images, targets = next(batch_iterator)
    print(np.shape(images))
    print(targets)






