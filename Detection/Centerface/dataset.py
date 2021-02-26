from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.utils.data as data
import cv2
import numpy as np
import os
import math
import matplotlib.pyplot as plt

from gaussian import draw_gaussian, gaussian_radius
from gaussian_v2 import gt_creator

class WiderFaceDetection(data.Dataset):
    def __init__(self, split, input_res=640, down_ratio=4, transform=None):
        super(WiderFaceDetection, self).__init__()
        # configs
        self.split = split  # in [train, val]
        self.dataset_path = '/home/dddzz/worksapce/Datasets/Widerface-retinaface'
        self.num_classes = 1 # face
        self.num_joints = 5
        self.input_res = input_res
        self.down_ratio = down_ratio
        self.output_res = self.input_res // self.down_ratio

        # model
        self.min_face = 0

        # init
        self.imgs_path, self.targets = self._load_widerface()
        self.transform = transform

    def __len__(self):
        return len(self.imgs_path)

    def _load_widerface(self):
        if self.split == 'train':
            imgs_path = []
            targets = []
            txt_path = os.path.join(self.dataset_path, self.split, "wider_{}.txt".format(self.split))
            f = open(txt_path, 'r')
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
                        targets.append(labels_copy)
                        labels.clear()  # 清除所元素
                    path = line[2:]
                    path = txt_path.replace("wider_{}.txt".format(self.split), 'images/') + path
                    imgs_path.append(path)
                else:
                    line = line.split(' ')
                    label = [float(x) for x in line]
                    labels.append(label)
            targets.append(labels)
            return imgs_path, targets
        else:
            # val
            imgs_path = []
            targets = []
            txt_path = os.path.join(self.dataset_path, self.split, "wider_{}.txt".format(self.split))
            f = open(txt_path, 'r')
            lines = f.readlines()
            for line in lines:
                line = line.rstrip()
                path = txt_path.replace("wider_{}.txt".format(self.split), 'images') + line
                imgs_path.append(path)
            return imgs_path, targets

    def pull_item(self, index, vis=False):
        # read image
        im_path = self.imgs_path[index]
        image = cv2.imread(im_path, cv2.IMREAD_COLOR)
        height, width, _ = image.shape
        # load labels
        targets = self.targets[index]
        annotations = np.zeros((0, 15))

        if len(targets) == 0:
            return annotations

        # load anno for each box
        for idx, target in enumerate(targets):
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = target[0]  # x1
            annotation[0, 1] = target[1]  # y1
            annotation[0, 2] = target[0] + target[2]  # x2
            annotation[0, 3] = target[1] + target[3]  # y2
            # landmarks
            annotation[0, 4] = target[4]  # l0_x
            annotation[0, 5] = target[5]  # l0_y
            annotation[0, 6] = target[7]  # l1_x
            annotation[0, 7] = target[8]  # l1_y
            annotation[0, 8] = target[10]  # l2_x
            annotation[0, 9] = target[11]  # l2_y
            annotation[0, 10] = target[13]  # l3_x
            annotation[0, 11] = target[14]  # l3_y
            annotation[0, 12] = target[16]  # l4_x
            annotation[0, 13] = target[17]  # l4_y
            if (annotation[0, 4]<0):
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1
            annotations = np.append(annotations, annotation, axis=0)
        # check loaded data
        if vis:
            boxes = annotations[:, :4].copy()
            landms = annotations[:, 4:-1].copy()
            self.vis_dets(image, boxes, landms)
            #print(annotations)
        return image, annotations

    def vis_dets(self, image, boxes, landms, percent_mod=False):
        img = image.copy()
        height, width, _ = image.shape
        if percent_mod:
            bx = [0, 2]
            by = [1, 3]
            boxes[:, bx] *= width
            boxes[:, by] *= height

            mask = [landms[:, 0] == -1]
            lx = [0, 2, 4, 6, 8]
            ly = [1, 3, 5, 7, 9]
            landms[:, lx] *= width
            landms[:, ly] *= height
            landms[mask] = [-1] * 10

        for id in range(len(boxes)):
            box = boxes[id]
            landm = landms[id]

            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 255), 3)
            if landm[0] != -1:
                for i in range(5):
                    cv2.circle(img, (int(landm[2*i]), int(landm[2*i+1])), 5, (0, 255, 255), 1)

        cv2.imshow('image', img)
        cv2.waitKey(0)


    def pull_image(self, index):
        im_path = self.imgs_path[index]
        image = cv2.imread(im_path, cv2.IMREAD_COLOR)
        #print(im_path)
        im_sub, im_name = im_path.rstrip().split('/')[-2:]
        #print(im_sub)
        #print(im_name)
        return image, im_sub, im_name


    def pull_heatmaps(self, index, vis=False):
        image, targets = self.pull_item(index)

        boxes = targets[:, :4].copy()
        landms = targets[:, 4:-1].copy()
        labels = targets[:, -1].copy()

        image, boxes, landms, labels = self.transform(image, boxes, landms, labels)
        #print('num_boxes: ', len(boxes))
        #print('num_landms: ', len(landms))
        #print('num_labels: ', len(labels))
        #self.vis_dets(image, boxes, landms, percent_mod=True)

        # box in x1y1x2y2_percent mod
        hm = np.zeros((self.output_res, self.output_res, self.num_classes), dtype=np.float32)
        reg = np.zeros((self.output_res, self.output_res, 2), dtype=np.float32)
        wh = np.zeros((self.output_res, self.output_res, 2), dtype=np.float32)
        reg_mask = np.zeros((self.output_res, self.output_res), dtype=np.float32)
        ldms = np.zeros((self.output_res, self.output_res, 10), dtype=np.float32)
        ldms_mask = np.zeros((self.output_res, self.output_res), dtype=np.float32)

        for id in range(len(boxes)):
            box = boxes[id].copy()
            landm = landms[id].copy()

            box[[0, 2]] = np.clip(box[[0, 2]], 0, self.output_res - 1)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, self.output_res - 1)
            cls_id = 0
            
            box *= self.output_res
            box_h, box_w = box[3] - box[1], box[2] - box[0]

            if box_h > 0 and box_w > 0 and box_h * box_w >= self.min_face:
                radius = gaussian_radius((math.ceil(box_h), math.ceil(box_w))) 
                radius = max(0, int(radius))

                # ct: [cx, cy]
                ct = np.array([(box[0]+box[2])/2, (box[1]+box[3])/2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                hm[:, :, cls_id] = draw_gaussian(hm[:, :, cls_id], ct_int, radius)
                wh[ct_int[1], ct_int[0]] = np.log(1. * box_w), np.log(1. * box_h)
                reg[ct_int[1], ct_int[0]] = ct - ct_int
                reg_mask[ct_int[1], ct_int[0]] = 1
                
                if landm[0] > 0:
                    x_id = [0, 2, 4, 6, 8]
                    y_id = [1, 3, 5, 7, 9]
                    ldms[ct_int[1], ct_int[0], x_id] = (landm[x_id] - ct[0]) / (box_w * self.down_ratio)
                    ldms[ct_int[1], ct_int[0], y_id] = (landm[y_id] - ct[1]) / (box_h * self.down_ratio)
                    ldms_mask[ct_int[1], ct_int[0]] = 1

        image = np.array(image, dtype=np.float32)
        if vis:
            self.vis_hms(image, hm, boxes, ldms_mask=ldms_mask, percent_mod=True)

        return image, hm, reg, wh, reg_mask, ldms, ldms_mask


    def pull_heatmaps_v2(self, index, vis=True):
        image, targets = self.pull_item(index)

        boxes = targets[:, :4].copy()
        landms = targets[:, 4:-1].copy()
        labels = targets[:, -1].copy()

        image, boxes, landms, labels = self.transform(image, boxes, landms, labels)
        hm, reg, wh, reg_mask = gt_creator(input_size=self.input_res, stride=self.down_ratio, label_lists=boxes)

        if vis:
            self.vis_hms(image, hm, boxes)

        return image, hm, reg, wh, reg_mask#, ldms, ldms_mask


    def vis_hms(self, image, heatmap, boxes, ldms_mask=None, percent_mod=True):
        img = image.copy()
        hm = heatmap.copy()
        height, width, _ = image.shape
        print('height: {}, width: {} of image'.format(height, width))

        if percent_mod:
            bx = [0, 2]
            by = [1, 3]
            boxes[:, bx] *= width
            boxes[:, by] *= height

        for id in range(len(boxes)):
            box = boxes[id]
            #print(box)
            img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 255), 3)

        img = np.array(img, dtype=np.float32)
        if ldms_mask is not None:
            print('box with lms: {}'.format(ldms_mask.sum()), ldms_mask.shape)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(122)
        plt.imshow(hm)
        plt.show()

    def __getitem__(self, index):
        image, hm, reg, wh, reg_mask, ldms, ldms_mask = self.pull_heatmaps(index, vis=False)
        return image, hm, reg, wh, reg_mask, ldms, ldms_mask


def widerface_collate(batch):
    batch_imgs = []
    batch_hms, batch_regs, batch_whs, batch_reg_masks = [], [], [], []
    batch_ldms, batch_ldms_masks = [], []
    for image, hm, reg, wh, reg_mask, ldms, ldms_mask in batch:
        batch_imgs.append(image)
        batch_hms.append(hm)
        batch_regs.append(reg)
        batch_whs.append(wh)
        batch_reg_masks.append(reg_mask)
        batch_ldms.append(ldms)
        batch_ldms_masks.append(ldms_mask)

    batch_imgs = np.array(batch_imgs)
    batch_hms = np.array(batch_hms)
    batch_regs = np.array(batch_regs)
    batch_whs = np.array(batch_whs)
    batch_reg_masks = np.array(batch_reg_masks)
    batch_ldms = np.array(batch_ldms)
    batch_ldms_masks = np.array(batch_ldms_masks)

    batch_imgs = batch_imgs.transpose((0, 3, 1, 2))

    return batch_imgs, batch_hms, batch_regs, batch_whs, \
           batch_reg_masks, batch_ldms, batch_ldms_masks



if __name__ == "__main__":

    from transform import SSDAugmentation

    transform = SSDAugmentation(size=800)

    train = WiderFaceDetection('train', transform=transform, input_res=800)
    #val = WiderFaceDetection('val', transform=transform)
    #train.pull_item(5, vis=True)
    for i in range(20):
        train.pull_heatmaps(3, vis=True)


















