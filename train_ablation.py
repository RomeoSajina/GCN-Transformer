import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random

from dataset import create_datasets

from model.ablation_model import create_model
from utils.metrics import *
from utils.config import parse_args


def set_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def loss_fnc(output, target):
    loss = torch.mean( torch.norm(output.reshape(-1,3) - target.reshape(-1,3), 2, 1) )    
    return loss


def train_one_epoch(config, model, train_loader, optimizer, scheduler):
    running_loss = 0.

    for i, x_orig in enumerate(iter(train_loader)):

        x = x_orig.clone()

        optimizer.zero_grad()

        y_pred, aux = model(x)
        
        loss = loss_fnc(y_pred, x)
        
        if "vel_loss" not in config.ablation_exclude:
            loss += loss_fnc(y_pred[:, :, 1:] - y_pred[:, :, :-1], x[:, :, 1:] - x[:, :, :-1])
        
        if "scene_loss" not in config.ablation_exclude:
            r = x.reshape(-1, 2, 30, 39)
            dist = (r[:, 0] - r[:, 1]).pow(2).sqrt()
            loss += loss_fnc(aux, dist) * 0.1

        if "vel_loss" not in config.ablation_exclude:
            loss += loss_fnc(aux[:, 1:] - aux[:, :-1], dist[:, 1:] - dist[:, :-1]) * 0.1
        
        running_loss += loss.item()

        loss.backward()

        optimizer.step()

    avg_loss = running_loss / (i+1)

    scheduler.step()

    return avg_loss


def calc_ds_loss(config, model, ds_loader):
    running_vloss = 0.0

    for i, x in enumerate(iter(ds_loader)):
        with torch.no_grad():
            y_pred = model(x)
        
            vloss = loss_fnc(y_pred[:, :, -config.output_len:], x[:, :, -config.output_len:])

            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    return avg_vloss


def calc_vim(config, model, ds_loader):
    
    y_pred = []
    y_true = []
    for i, x in enumerate(iter(ds_loader)):
        with torch.no_grad():
            y_p = model(x)

            y_p = y_p[:, :, -config.output_len:]
            y_t = x[:, :, -config.output_len:]

            y_pred.append(y_p.detach().cpu().numpy())
            y_true.append(y_t.detach().cpu().numpy())

    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    
    vims = [np.mean([(VIM(pred[0][:LEN], gt[0][:LEN]) + VIM(pred[1][:LEN], gt[1][:LEN])) / 2. for pred, gt in zip(y_pred, y_true)] ) * 100 for LEN in [2, 4, 8, 10, 14]]

    return vims


def train(config):
    
    best_avg_vim = 55

    train_loader, _, valid_loader = create_datasets(config=config)
    
    if "augmentation" in config.ablation_exclude:
        train_loader.dataset.use_augmentation = False

    model = create_model(config)
    print("#Param:", sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters()))
    
    if config.ckp is not None and len(config.ckp) > 0:
        print("Loading model from:", config.ckp)
        model.load_state_dict(torch.load(config.ckp))
        model.eval()        

    if config.finetune:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[128], gamma=0.1)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[256], gamma=0.1)

    for epoch_number in range(1, config.num_epoch+1):
        
        _lr = get_lr(optimizer)
        
        model.train(True)
        avg_loss = train_one_epoch(config, model, train_loader, optimizer, scheduler)
        model.train(False)

        avg_vloss = calc_ds_loss(config, model, valid_loader)
        
        print('EPOCH {0} | loss train: {1:.3f}, loss valid: {2:.3f} | lr: {3:.5f}'.format(epoch_number, avg_loss, avg_vloss, _lr))
        
        vims = calc_vim(config, model, valid_loader)
        avg_vim = np.mean(vims)
        
        print("Test [100ms 240ms 500ms 640ms 900ms]:", " ".join(["{0:.2f}".format(v) for v in vims]), " - {0:.2f}".format(avg_vim))
        
        if best_avg_vim > avg_vim:
            best_avg_vim = avg_vim
            print("Saving best model:", avg_vim)
            torch.save(model.state_dict(), config.log_dir + "best_epoch.pt")
        
    torch.save(model.state_dict(), config.log_dir + "last_epoch.pt")

    return model


if __name__ == '__main__':
    
    config = parse_args()
    
    config.log_dir = config.log_dir

    set_all_seeds(1234)

    print("Config:", config)

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir, exist_ok=True)
    
    model = train(config)

