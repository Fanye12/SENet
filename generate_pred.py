import imageio
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os, argparse
from make_dataset import test_dataset
from net import mae_vit_base_patch16_dec512d8b, mae_vit_large_patch16_dec512d8b
from LICM import set_LICM


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--checkpoint_path', type=str, default='/media/lab532/MAE_COD_SOD/checkpoints/JT_SOD2000/mae-24.pth')  #改这里
opt = parser.parse_args()

model = mae_vit_base_patch16_dec512d8b()
# set_LICM(model=model)
model = model.cuda()
model = nn.DataParallel(model)
model.load_state_dict(torch.load(opt.checkpoint_path))


model.eval()
# for _data_name in ['CAMO','CHAMELEON','COD10K','NC4K']:
for _data_name in ['DUTS-TE', 'DUT-OMRON', 'HKU-IS', 'ECSSD', 'PASCAL-S']:
    data_path = './dataset/SOD/TestDataset/{}/'.format(_data_name)
    save_path = './results/JT_SOD2000/{}/'.format(_data_name) #改这里
    
    os.makedirs(save_path, exist_ok=True)

    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res = model(image)
 
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        imageio.imwrite(save_path+name, (res*255).astype(np.uint8))

for _data_name in ['CAMO','CHAMELEON','COD10K','NC4K']:
# for _data_name in ['DUTS-TE', 'DUT-OMRON', 'HKU-IS', 'ECSSD', 'PASCAL-S']:
    data_path = './dataset/COD/TestDataset/{}/'.format(_data_name)
    save_path = './results/JT_SOD2000/{}/'.format(_data_name) #改这里
    
    os.makedirs(save_path, exist_ok=True)

    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res = model(image)
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        imageio.imwrite(save_path+name, (res*255).astype(np.uint8))  
