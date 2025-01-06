import torch
import os
import argparse
from datetime import datetime
from make_dataset import get_loader
from UTIls import clip_gradient, AvgMeter, poly_lr
from LICM import set_LICM
import torch.nn.functional as F
import numpy as np
from net import SENet, interpolate_pos_embed
import torch.nn as nn
import copy
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)
torch.backends.cudnn.benchmark = True

def get_parser():
    parser = argparse.ArgumentParser(description="Training program for SENet for COD and SOD")

    parser.add_argument('--epochs', type=int,
                        default=25, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=384, help='training img size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--masking_ratio', type=float,
                        default=0.05, help='masking ratio')                    
    parser.add_argument('--pretrained_mae_path', type=str,
                        default='pretrained_model/mae_visualize_vit_base.pth')#MAE pretrained weight
    
    #params that need to be modified          
    
    parser.add_argument('--weight_save_path', type=str,
                        default='checkpoints/SENet')
    parser.add_argument('--train_log_path', type=str,
                        default='log/SENet.txt')
    parser.add_argument('--task', type=str,
                        default='cod')  #'sod'for sod task, 'cod' for cod task
    parser.add_argument('--set_LICM', type=bool,
                        default=True)  #if True, set LICM
    opt = parser.parse_args()
    return opt

def patchify(imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = 16
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def dynamic_structure_loss1(pred, mask):
    weight_map = copy.deepcopy(mask)
    weit = torch.zeros_like(weight_map)
    for i in range(weight_map.shape[0]):
        tt = weight_map[i]
        a = 384*384 / (tt == 1).sum()

        tt[(tt != 0) & (tt != 1)] = a    
        tt[tt != a] = 1

        weit[i] = tt

    
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def recon_loss(imgs, pred, mask):
    """
    imgs: [N, 3, H, W]
    pred: [N, L, p*p*3]
    mask: [N, L], 0 is keep, 1 is remove, 
    """
    target = patchify(imgs)
    
    mean = target.mean(dim=-1, keepdim=True)
    var = target.var(dim=-1, keepdim=True)
    target = (target - mean) / (var + 1.e-6)**.5

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    # print(loss)
    return loss

def build_model():
    model = SENet()
    #load mae pretrained weight
    checkpoint = torch.load(opt.pretrained_mae_path, map_location='cpu')
    checkpoint_model = checkpoint['model']
    interpolate_pos_embed(model, checkpoint_model)
    model.load_state_dict(checkpoint_model, strict=False)

    if opt.set_LICM:
        set_LICM(model=model)
    
    # model = nn.DataParallel(model)
    model = model.cuda()
    
    for param in model.parameters():
        param.requires_grad = True
   
    return model

def train(train_loader, model, optimizer, epoch, loss_fn):

    model.train()
    loss_recorde = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
      
        images, gts = pack
        images = images.cuda()
        gts = gts.cuda()
        # label = label.cuda().long()
        pred, pred1, mask = model(images, mask_ratio=opt.masking_ratio)
        # ---- loss function ----
        
        seg_loss = dynamic_structure_loss1(pred, gts) #+ loss1(classify, label)
        
        reconstruction_loss = recon_loss(imgs=images, pred=pred1, mask=mask)
        loss = 0.9 * seg_loss + 0.1 * reconstruction_loss#total loss
        # ---- backward ----
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        # ---- recording loss ----
        loss_recorde.update(loss.data, opt.batchsize)
        # ---- train visualization ----
        if i % 60 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[loss: {:,.4f}]'.
                  format(datetime.now(), epoch, opt.epochs, i, total_step,
                        loss_recorde.avg))
            file.write('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                       '[loss: {:,.4f}]\n'.
                       format(datetime.now(), epoch, opt.epochs, i, total_step,
                        loss_recorde.avg))

    save_path = opt.weight_save_path
    os.makedirs(save_path, exist_ok=True)
    if (epoch + 1) % 5 == 0 or (epoch + 1) == opt.epochs:
        torch.save(model.state_dict(), save_path + 'mae-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'mae-%d.pth' % epoch)
        file.write('[Saving Snapshot:]' + save_path + 'mae-%d.pth' % epoch + '\n')
        
if __name__ == '__main__':

    opt = get_parser()
    Loss_fn = structure_loss
    model = build_model()

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    if opt.task == 'cod':
        img_root = 'dataset/COD/TrainDataset/Imgs/'
        gt_root = 'dataset/COD/TrainDataset/GT/'
    elif opt.task == 'sod':
        img_root = 'dataset/SOD/TrainDataset-DUTS-TR/Imgs/'
        gt_root = 'dataset/SOD/TrainDataset-DUTS-TR/GT/'

    train_loader = get_loader(image_root=img_root, gt_root=gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
 
    total_step = len(train_loader)
    print(total_step)

    file = open(opt.train_log_path, "a")
    print("Start Training")

    for epoch in range(opt.epochs):
        poly_lr(optimizer, opt.lr, epoch, opt.epochs)
        train(train_loader, model, optimizer, epoch, loss_fn = Loss_fn)

    file.close()
