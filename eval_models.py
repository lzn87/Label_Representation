from __future__ import absolute_import, division, print_function, unicode_literals

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms
import pandas as pd

import architecture
import argparse
import cifar
import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn as nn
from utils.trainutil import (
    train_directory_setup,
    train_log_results,
    train,
    valid_highdim,
    valid_category,
    valid_lowdim,
    test_highdim,
    test_category,
    test_lowdim,
)

from torchtoolbox.nn import LabelSmoothingLoss
import json

labels = ['category', 'speech']
model_names = ['resnet110', 'vgg19']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

eval_res = {}

for label in labels:
    eval_res[label] = {}
    for model_name in model_names:
        if 'categor' in label:
            model = architecture.PytorchCategoryModel(model_name, 182)
        else:
            model = architecture.PytorchHighDimensionalModel(model_name, 182)
        
        dataset = cifar.dataset_wrapper(dataset="iwildcam", 
                                datatype='train', 
                                data_dir='/work/zli/wilds', 
                                label=label, 
                                label_root='labels/wild_sound_label.npy',
                                return_meta=True)
        
        state_dict = torch.load(f"/work/zli/iwildcam/seed100/{model_name}/model_{label}/{label}_seed100_{model_name}_best_model.pth")
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)
        
        y_targ = []
        y_pred = []
        all_meta = []

        if dataset.mels is not None:
            cls_cent = torch.from_numpy(dataset.mels).view(182, -1).to(device)
            
        with torch.no_grad():
            for i, (img, lbl, meta) in enumerate(loader):
                img = img.to(device)
                lbl = lbl.to(device)
                
                if 'categor' not in label:
                    lbl, meta = meta
                    lbl = lbl.to(device)
                    meta = meta.to(device)
                else:
                    meta = meta.to(device)

                pred, _ = model(img)
                
                if 'categor' not in label:
                    N = pred.size(0)
                    pred = pred.view(N, -1)
                    pred = torch.nn.functional.normalize(pred)
                    pred = torch.mm(pred, torch.nn.functional.normalize(cls_cent).T).argmax(dim=1)
                else:
                    pred = pred.argmax(dim=1)
                
                y_pred.append(pred)
                y_targ.append(lbl)
                all_meta.append(meta)
                
        y_pred = torch.cat(y_pred).cpu()
        y_targ = torch.cat(y_targ).cpu()
        all_meta = torch.cat(all_meta).cpu()
        res = dataset.dataset.eval(y_pred, y_targ, all_meta)
        print(label, model_name, res[-1], '\n')
        eval_res[label][model_name] = res

with open('res_train.json', 'w') as f:
    json.dump(eval_res, f)