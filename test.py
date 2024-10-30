import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from model import MMPD_DTA
import metrics
from datetime import datetime
import torch
from torch_geometric.loader import DataLoader
from dataset import TestbedDataset
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, r2_score, precision_recall_curve, auc


# chongfu = ['1vso', '2pog', '1h23', '1ps3', '1yc1', '10gs', '2j78', '2d3u', '2iwx', '1gpk', '2jdu', '1s38', '1syi', '3f3d', '1nc1', '1r5y', '1g2k', '2qbr', '1p1n', '2qnq', '1qkt', '2p4y', '2qmj', '1w3k', '1uto', '1hnn', '1nc3', '2vkm', '3bgz', '2fvd', '2qbq', '2brb', '1h22', '1eby', '2hb1', '2vw5', '2jdy', '3e92', '1z95']
# chongfu = ['2j78', '1uto', '2vw5', '2fvd', '2jdu', '1yc1', '2qbr', '2d3u', '1hnn', '2brb', '10gs', '2p4y', '2qmj', '1vso', '1r5y', '2iwx', '1z95', '1ps3', '2jdy', '1w3k', '2hb1', '1gpk', '1h23']


def test(model: nn.Module, test_loader, loss_function, device, show):
    model.eval()
    test_loss = 0
    outputs = []
    targets = []

    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(test_loader), disable=not show, total=len(test_loader)):
            data = data.to(device)
            y = data.y_s
            y_hat= model(data)
            test_loss += loss_function(y_hat.view(-1), y.view(-1)).item()
            outputs.append(y_hat.cpu().numpy().reshape(-1))
            targets.append(y.cpu().numpy().reshape(-1))
    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)
    test_loss /= len(test_loader.dataset)
    evaluation = {
        'loss': test_loss,
        'c_index': metrics.c_index(targets, outputs),
        'RMSE': metrics.RMSE(targets, outputs),
        'MAE': metrics.MAE(targets, outputs),
        'SD': metrics.SD(targets, outputs),
        'CORR': metrics.CORR(targets, outputs),
    }

    return evaluation



device = torch.device("cuda")
model = MMPD_DTA().to(device)

loss_fn = nn.MSELoss(reduction='sum')


data_loaders = {phase_name:
                DataLoader(TestbedDataset(root='data', dataset=phase_name),
                           batch_size= 64  ,
                           pin_memory=True,
                           shuffle=False,follow_batch=['x_s','x_t'])
            for phase_name in ['test2016','test2013','BDBbind']}
model.load_state_dict(torch.load('result/best/best_model.pt'))
for _p in ['test2016','test2013','BDBbind']:
    performance = test(model, data_loaders[_p], loss_fn, device, False)
    print(_p,performance)
