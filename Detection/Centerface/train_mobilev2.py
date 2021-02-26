import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from mobile_model import Centerface_MobileNetv2, load_parrallal_model, load_optimizer_state
from losses import focal_loss, reg_l1_loss, reg_l1_loss_ldmks
from dataset import WiderFaceDetection, widerface_collate
from transform import SSDAugmentation


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def adjust_learning_rate(optimizer, epoch, init_lr=5e-4, decay_epoch=[80, 110]):
    if epoch <= decay_epoch[0]:
        lr = init_lr
    elif epoch <= decay_epoch[1]:
        lr = init_lr * 0.1
    else:
        lr = init_lr * 0.1 * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def fit_one_epoch(net, epoch, epoch_size, gen, Epoch, cuda):

    total_wh_loss = 0
    total_oft_loss = 0
    total_pt_loss = 0
    total_c_loss = 0
    total_loss = 0

    net.train()
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            with torch.no_grad():
                if cuda:
                    batch = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in batch]
                else:
                    batch = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in batch]

            imgs, hms_t, regs_t, whs_t, reg_masks_t, ldmks_t, ldmk_masks_t = batch

            optimizer.zero_grad()
            hm, wh, offset, ldmks = net(imgs)
            c_loss = focal_loss(hm, hms_t)
            # pred, gt, ind, mask
            off_loss = reg_l1_loss(offset, regs_t,  reg_masks_t)
            wh_loss = 0.1 * reg_l1_loss(wh, whs_t, reg_masks_t,)
            ldmk_loss = 0.1 * reg_l1_loss_ldmks(ldmks, ldmks_t, ldmk_masks_t)

            loss = c_loss + wh_loss + off_loss + ldmk_loss
            loss.backward()

            total_loss += loss.item()
            total_c_loss += c_loss.item()
            total_oft_loss += off_loss.item()
            total_wh_loss += wh_loss.item()
            total_pt_loss += ldmk_loss.item()

            optimizer.step()

            pbar.set_postfix(**{'total_pt_loss': total_pt_loss / (iteration + 1),
                                'total_oft_loss': total_oft_loss / (iteration + 1),
                                'total_wh_loss': total_wh_loss / (iteration + 1),
                                'total_c_loss': total_c_loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    f.write("Epoch:{} ||c_Loss:{} || oft_loss:{} || wh_loss:{} || pt_loss: {} || lr:{} +\n".format( epoch+1,
                                                                           total_c_loss / (iteration + 1),
                                                                           total_oft_loss / (iteration + 1),
                                                                           total_wh_loss / (iteration + 1),
                                                                           total_pt_loss / (iteration + 1),
                                                                                    get_lr(optimizer)))

    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f ' % (total_loss / (epoch_size + 1)))
    print('Saving state, iter:', str(epoch + 1))
    filepath = '{}'.format(log_name)+'/Epoch%d-Total_Loss%.4f.pth' % ((epoch + 1), total_loss / (epoch_size + 1))
    state = {
        'epoch': epoch,
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filepath)

    return total_loss / (epoch_size + 1)



if __name__ == "__main__":
    # configs
    seed = 317
    use_cudnn_benchmark = True
    log_name = 'logs/log9'
    if not os.path.exists(log_name):
        os.mkdir(log_name)
    f = open('{}/log.txt'.format(log_name), 'a')
    if not os.path.exists(log_name):
        os.mkdir(log_name)
    resume_mode = True
    restore_optim = True

    # training params
    init_lr = 5e-4
    Batch_size = 16
    input_res = 800

    # data setting
    transform = SSDAugmentation(size=input_res)
    train_dataset = WiderFaceDetection('train', input_res=input_res, transform=transform)
    gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=8, pin_memory=True, shuffle=True,
                     drop_last=True, collate_fn=widerface_collate)
    num_train = train_dataset.__len__()
    epoch_size = num_train // Batch_size


    # model setting
    model_path = 'logs/log9/Epoch4-Total_Loss3.0034.pth'

    net = Centerface_MobileNetv2(pretrain=True)
    if resume_mode:
        net, start_epoch = load_parrallal_model(net, model_path)

    # random seed
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = use_cudnn_benchmark

    #net = torch.nn.DataParallel(net)
    net = net.cuda()


    optimizer = optim.Adam(net.parameters(), init_lr, weight_decay=5e-4)
    if resume_mode and restore_optim:
        optimizer = load_optimizer_state(optimizer, model_path, init_lr)
        Start_Epoch = start_epoch + 1
        Stop_Epoch = 140
    else:
        Start_Epoch = 0
        Stop_Epoch = 140

    f.write("Stage1\n")

    # start training process
    for epoch in range(Start_Epoch, Stop_Epoch):
        # adjust lr
        lr = adjust_learning_rate(optimizer, epoch)
        # in one epoch
        total_loss = fit_one_epoch(net, epoch, epoch_size, gen, Stop_Epoch, cuda=True)

    f.close()

    '''
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch =  checkpoint['epoch'] + 1
    '''
