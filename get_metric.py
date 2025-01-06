# -*- coding: utf-8 -*-
# @Time    : 2020/11/21
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
from tqdm import tqdm

# pip install pysodmetrics
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

method='JT_SOD2000' #改这里
# for _data_name in ['CAMO','CHAMELEON','COD10K','NC4K']:
for _data_name in [ 'DUTS-TE', 'DUT-OMRON', 'HKU-IS', 'ECSSD', 'PASCAL-S']:#
    mask_root = './dataset/SOD/TestDataset/{}/GT'.format(_data_name) #
    pred_root = './results/{}/{}/'.format(method, _data_name)        #
    # pred_root = '/media/lab532/COD/SINet-master/Result/sod/{}/'.format( _data_name)
    mask_name_list = sorted(os.listdir(mask_root))
    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    M = MAE()
    for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
        mask_path = os.path.join(mask_root, mask_name)
        pred_path = os.path.join(pred_root, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        FM.step(pred=pred, gt=mask)
        WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        M.step(pred=pred, gt=mask)

    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = M.get_results()["mae"]

    results = {
        "Smeasure": sm,
        "meanEm": em["curve"].mean(),
        "wFmeasure": wfm,
        "MAE": mae,
        "adpEm": em["adp"],
        "maxEm": em["curve"].max(),
        "adpFm": fm["adp"],
        "meanFm": fm["curve"].mean(),
        "maxFm": fm["curve"].max(),
    }

    print(results)
    file=open("metric/JT_SOD2000.txt", "a")#改这里
    file.write(method+' '+_data_name+' '+str(results)+'\n')

for _data_name in ['CAMO','CHAMELEON','COD10K','NC4K']:
# for _data_name in [ 'DUTS-TE', 'DUT-OMRON', 'HKU-IS', 'ECSSD', 'PASCAL-S']:#
    mask_root = './dataset/COD/TestDataset/{}/GT'.format(_data_name) #
    pred_root = './results/{}/{}/'.format(method, _data_name)        #
    mask_name_list = sorted(os.listdir(mask_root))
    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    M = MAE()
    for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
        mask_path = os.path.join(mask_root, mask_name)
        pred_path = os.path.join(pred_root, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        FM.step(pred=pred, gt=mask)
        WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        M.step(pred=pred, gt=mask)

    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = M.get_results()["mae"]

    results = {
        "Smeasure": sm,
        "meanEm": em["curve"].mean(),
        "maxFm": fm["curve"].max(),
        "MAE": mae,
        "wFmeasure": wfm,
        "adpEm": em["adp"],
        "maxEm": em["curve"].max(),
        "adpFm": fm["adp"],
        "meanFm": fm["curve"].mean()
    }

    print(results)
    file=open("metric/JT_SOD2000.txt", "a")#改这里
    file.write(method+' '+_data_name+' '+str(results)+'\n')
