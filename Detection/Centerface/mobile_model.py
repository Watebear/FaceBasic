from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import math
import numpy as np

from mobilenet_v2 import get_model


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

# input channel:  x:out_dim / y:channel
# ouput channel:  out_dim
class UpSample(nn.Module):
    def __init__(self, out_dim, channel):
        super(UpSample, self).__init__()
        self.out_dim = out_dim
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                out_dim, out_dim, kernel_size=2, stride=2, padding=0,
                output_padding=0, groups=out_dim, bias=False),
            nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.1),
            nn.ReLU())
        self.conv = nn.Sequential(
            nn.Conv2d(channel, out_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.1),
            nn.ReLU(inplace=True))

    # layers = [x, y]
    def forward(self, layers):
        layers = list(layers)
        x = self.up(layers[0])
        y = self.conv(layers[1])
        out = x + y
        return out

class FPN(nn.Module):
    def __init__(self, out_dim=64, feat_channel=[320, 96, 32, 24]):
        super(FPN, self).__init__()
        self.conv_first = nn.Sequential(
                    nn.Conv2d(feat_channel[0], out_dim,
                              kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.1),
                    nn.ReLU(inplace=True))

        self.UpSample1 = UpSample(out_dim, feat_channel[1])
        self.UpSample2 = UpSample(out_dim, feat_channel[2])
        self.UpSample3 = UpSample(out_dim, feat_channel[3])

        self.conv_last = nn.Sequential(
                    nn.Conv2d(out_dim, out_dim,
                              kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(out_dim, eps=1e-5, momentum=0.01),
                    nn.ReLU(inplace=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                fill_up_weights(m)
    '''
    [feature6, feature4, feature2, feature1]
    '''
    def forward(self, feats):
        x = self.conv_first(feats[0])
        y = feats[1]
        x = self.UpSample1([x, y])
        y = feats[2]
        x = self.UpSample2([x, y])
        y = feats[3]
        x = self.UpSample3([x, y])
        x = self.conv_last(x)
        return x

class Head(nn.Module):
    def __init__(self, num_classes=1, channel=24, bn_momentum=0.1):
        super(Head, self).__init__()
        '''
        320,25,25 -> 96,50,50 -> 32,100,100 -> 24,200,200  ==> 24,200,200
                     96,50,50    32,100,100    24,200,200
        '''
        # heat map
        self.cls_head = nn.Conv2d(channel, num_classes, kernel_size=1, stride=1, padding=0)
        # width height
        self.wh_head = nn.Conv2d(channel, 2, kernel_size=1, stride=1, padding=0)
        # center offset
        self.reg_head = nn.Conv2d(channel, 2, kernel_size=1, stride=1, padding=0)
        # landmarks head
        self.ldmks_head = nn.Conv2d(channel, 10, kernel_size=1, stride=1, padding=0)

        # init
        self.cls_head.bias.data.fill_(-2.19)
        for head in [self.wh_head, self.reg_head, self.ldmks_head]:
            nn.init.normal_(head.weight, std=0.001)
            nn.init.constant_(head.bias, 0)

    def forward(self, x):
        hm = self.cls_head(x).sigmoid_()
        wh = self.wh_head(x)
        offset = self.reg_head(x)
        ldmks = self.ldmks_head(x)
        return hm, wh, offset, ldmks



class Centerface_MobileNetv2(nn.Module):
    def __init__(self,  pretrain=False):
        super(Centerface_MobileNetv2, self).__init__()
        self.backbone = get_model(pretrained=pretrain) # [feature_1 2 4 6]
        self.fpn = FPN(feat_channel=[320, 96, 32, 24])
        self.head = Head(num_classes=1, channel=64)
        self.unfreeze_backbone()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        feat1 = self.backbone[0](x)
        feat2 = self.backbone[1](feat1)
        feat4 = self.backbone[2](feat2)
        feat6 = self.backbone[3](feat4)

        merged = self.fpn([feat6, feat4, feat2, feat1])

        hm, wh, offset, ldmks = self.head(merged)

        return hm, wh, offset, ldmks


def load_parrallal_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    state_dict_ = checkpoint['net']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]

    model_state_dict = model.state_dict()
    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}.'.format( k, model_state_dict[k].shape, state_dict[k].shape))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k))

    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    epoch = checkpoint['epoch']

    return model, epoch

def load_optimizer_state(optimizer, model_path, init_lr):
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr
    return optimizer





if __name__ == "__main__":
    model = Centerface_MobileNetv2()
    data = torch.ones([1, 3, 800, 800])
    out = model(data)

    print(model)


