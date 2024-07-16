import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from model_pcl import ZHC
import metrics
from datetime import datetime
import torch
from torch_geometric.loader import DataLoader
from dataset import TestbedDataset
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import GradScaler, autocast



def test(model: nn.Module, test_loader, loss_function, device, show):
    model.eval()
    test_loss = 0
    outputs = []
    targets = []
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(test_loader), disable=not show, total=len(test_loader)):
            data = data.to(device)
            y = data.y_s
            y_hat = model(data)
            # break
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







seed = 42 ##random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
setup_seed(seed)


device = torch.device("cuda")
model = ZHC().to(device)

loss_fn = nn.MSELoss(reduction='sum')


data_loaders = {phase_name:
                DataLoader(TestbedDataset(root='data', dataset=phase_name),
                           batch_size= 64  ,
                           pin_memory=True,
                           # num_workers=8,
                           shuffle=False,follow_batch=['x_s','x_t'])
            for phase_name in ['test2016','test2013']} #'train','val','test2016','test2013'
model.load_state_dict(torch.load('result/best/best_model.pt'))
for _p in ['test2016','test2013']:
    performance = test(model, data_loaders[_p], loss_fn, device, True)
    print(_p,performance)

