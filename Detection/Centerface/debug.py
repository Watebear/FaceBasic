import numpy as np
import torch
import torch.nn as nn
import cv2

def get_debug_data(dataset, index):
    image, hm, reg, wh, reg_mask, ldms, ldms_mask = dataset.pull_heatmaps(index)
    origin_image, im_sub, im_name = dataset.pull_image(index)

    hm = np.expand_dims(hm, axis=0)
    reg = np.expand_dims(reg, axis=0)
    wh = np.expand_dims(wh, axis=0)
    ldms = np.expand_dims(ldms, axis=0)

    hm = torch.from_numpy(hm).type(torch.FloatTensor)
    reg = torch.from_numpy(reg).type(torch.FloatTensor)
    wh = torch.from_numpy(wh).type(torch.FloatTensor)
    ldms = torch.from_numpy(ldms).type(torch.FloatTensor)

    return origin_image, hm, reg, wh, ldms


def pool_nms(heat, kernel=10):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def iou(b1, b2):
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)

    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                 np.maximum(inter_rect_y2 - inter_rect_y1, 0)

    area_b1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area_b2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / np.maximum((area_b1 + area_b2 - inter_area), 1e-6)
    return iou

def nms(results, nms):
    outputs = []
    for i in range(len(results)):
        detections = results[i]
        unique_class = np.unique(detections[:, -1])

        best_box = []
        if len(unique_class) == 0:
            results.append(best_box)
            continue

        for c in unique_class:
            cls_mask = detections[:, -1] == c

            detection = detections[cls_mask]
            scores = detection[:, 4]
            arg_sort = np.argsort(scores)[::-1]
            detection = detection[arg_sort]
            while np.shape(detection)[0]>0:
                best_box.append(detection[0])
                ious = iou(best_box[-1],detection[1:])
                detection = detection[1:][ious<nms]
        outputs.append(best_box)
    return outputs


def debug_decode(pred_hms, pred_regs, pred_whs, pred_ldms, threshold=0.1, topk=1000):
    bs, res_h, res_w, c = pred_hms.shape
    detects = []
    for i in range(bs):
        heatmap = pred_hms[i].view([-1, c])
        oft = pred_regs[i].view([-1, 2])
        wh = pred_whs[i].view([-1, 2])
        ldms = pred_ldms[i].view([-1, 10])

        yv, xv = torch.meshgrid(torch.arange(0, res_h), torch.arange(0, res_w))
        xv, yv = xv.flatten().float(), yv.flatten().float()

        class_conf, class_pred = torch.max(heatmap, dim=-1)
        mask = class_conf > threshold

        masked_wh = wh[mask]
        masked_oft = oft[mask]
        masked_ldm = ldms[mask]

        if len(masked_wh) == 0:
            detects.append([])
            continue

        xv_mask = torch.unsqueeze(xv[mask] + masked_oft[..., 0], dim=-1)
        yv_mask = torch.unsqueeze(yv[mask] + masked_oft[..., 1], dim=-1)

        half_w, half_h = torch.exp(masked_wh[..., 0:1]) / 2, torch.exp(masked_wh[..., 1:2]) / 2

        bboxes = torch.cat([xv_mask - half_w, yv_mask - half_h, xv_mask + half_w, yv_mask + half_h], dim=1)
        # ldmks


        # to percent coords
        bboxes[:, [0, 2]] /= res_w
        bboxes[:, [1, 3]] /= res_h

        # detect: [box_coord, score, class_id]
        detect = torch.cat([bboxes, torch.unsqueeze(class_conf[mask], -1), torch.unsqueeze(class_pred[mask],-1).float()], dim=-1)
        detects.append(detect.cpu().numpy()[:topk])

    return detects


def vis_decode(image, detects):
    height, width, _ = image.shape

    print(np.shape(detects))
    detects[:, [0, 2]] *= width
    detects[:, [1, 3]] *= height

    for i in range(len(detects)):
        dt = detects[i]
        cv2.rectangle(image, (int(dt[0]), int(dt[1])), (int(dt[2]), int(dt[3])), (0, 0, 255), 2)

    cv2.imshow('a', image)
    cv2.waitKey(0)

if __name__ == "__main__":
    from transform import SSDAugmentation
    from dataset import WiderFaceDetection

    transform = SSDAugmentation(size=640, debugmode=True)
    train_dataset = WiderFaceDetection('train', transform=transform)
    index = 200

    origin_image, hm, reg, wh, ldms = get_debug_data(train_dataset, index)
    print(hm.shape)

    detects = debug_decode(hm, reg, wh, ldms)

    detects = np.array(nms(detects, 0.00001))

    vis_decode(origin_image, detects[0])


