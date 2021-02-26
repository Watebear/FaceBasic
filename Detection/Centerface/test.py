from mobile_model import Centerface_MobileNetv2
from dataset import WiderFaceDetection
from transform import SSDAugmentation_infer
from decode import decode, nms, vis_decode

import cv2
import numpy as np
import torch

def test_one_image(image, transform, model, use_cuda):
    img = transform(image)
    img = torch.from_numpy(img).type(torch.FloatTensor)
    img = img.permute(2, 0, 1).unsqueeze(0)
    #print(img.shape)
    if use_cuda:
        img = img.cuda()

    with torch.no_grad():
        hm, wh, offset, ldmks = model(img)

    detects = decode(hm, offset, wh, ldmks, threshold=0.1, topk=1000)
    detects = np.array(nms(detects, nms=0.01))
    vis_decode(image, detects[0])


if __name__ == "__main__":
    # config
    use_cuda = True
    model_path = './logs/log5/Epoch200-Total_Loss0.7259.pth'

    # model
    model = Centerface_MobileNetv2(pretrain=False)
    #model = load_model(model, model_path)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['net'])
    model.eval()

    if use_cuda:
        model = model.cuda()

    # dataset
    transform = SSDAugmentation_infer(size=800)
    train_dataset = WiderFaceDetection('train', transform=transform, input_res=800)

    # detect one image
    index = 9000
    image, im_sub, im_name = train_dataset.pull_image(index)
    #cv2.imshow('a', image)
    #cv2.waitKey(0)
    test_one_image(image, transform, model, use_cuda)