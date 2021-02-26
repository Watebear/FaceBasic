import numpy as np
import random
import cv2



def random_choose(anno, choose_num):
    total = np.arange(len(anno))
    chosed_id = np.random.choice(total, choose_num, replace=False)
    chosed_anno = anno[chosed_id]
    return chosed_anno


def Data_anchor_sample(image, targets):
    maxSize = 12000
    infDistance = 9999999

    boxes = targets[:, :4].copy()
    landms = targets[:, 4:-1].copy()
    height, width, _ = image.shape

    boxArea = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    rand_idx = random.randint(0, len(boxArea) - 1)
    rand_Side = boxArea[rand_idx] ** 0.5

    anchors = [16, 32, 48, 64, 96, 128, 256, 512]
    distance = infDistance
    anchor_idx = 5
    for i, anchor in enumerate(anchors):
        if abs(anchor - rand_Side) < distance:
            distance = abs(anchor - rand_Side)  # 选择最接近的anchors
            anchor_idx = i

    target_anchor = random.choice(anchors[0:min(anchor_idx + 1, 5)])  # 随机选择一个相对较小的anchor，向下
    ratio = float(target_anchor) / rand_Side  # 缩放的尺度
    ratio = ratio * (2 ** random.uniform(-1, 1))  # [ratio/2, 2ratio]的均匀分布

    if int(height * ratio * width * ratio) > maxSize * maxSize:
        ratio = (maxSize * maxSize / (height * width)) ** 0.5

    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = random.choice(interp_methods)

    image = cv2.resize(image, None, None, fx=ratio, fy=ratio, interpolation=interp_method)
    boxes *= ratio
    landms *= ratio

    return image, boxes, landms

def get_border(border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i


if __name__ == "__main__":
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    print(random_choose(data, 15))




