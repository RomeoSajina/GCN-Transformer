import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import json
    
from dataset import create_datasets
from utils.metrics import *
from utils.config import parse_args


def pred_and_calc_metrics(config, model, ds_loader):
    
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

    mpjpes = [np.mean([(keypoint_mpjpe(pred[0][:LEN], gt[0][:LEN]) + keypoint_mpjpe(pred[1][:LEN], gt[1][:LEN])) / 2. for pred, gt in zip(y_pred, y_true)] ) * 1000 for LEN in [2, 4, 8, 10, 14]]

    return vims, mpjpes, y_pred


def test(config):
    
    if config.dataset.startswith("3dpw_ablation"):
        from model.ablation_model import create_model
    else:
        from model.model import create_model
    
    _, _, test_loader = create_datasets(config=config)
        
    model = create_model(config)
    print("#Param:", sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters()))
    
    if config.ckp is not None and len(config.ckp) > 0:
        print("Loading model from:", config.ckp)
        model.load_state_dict(torch.load(config.ckp))
        model.eval()
    
    vims, mpjpes, y_pred = pred_and_calc_metrics(config, model, test_loader)
    avg_vim = np.mean(vims)
    avg_mpjpe = np.mean(mpjpes)

    print("Test VIM [100ms 240ms 500ms 640ms 900ms]:", " ".join(["{0:.2f}".format(v) for v in vims]), " - {0:.2f}".format(avg_vim))

    print("Test MPJPE [100ms 240ms 500ms 640ms 900ms]:", " ".join(["{0:.2f}".format(v) for v in mpjpes]), " - {0:.2f}".format(avg_mpjpe))

    if not os.path.exists(config.log_dir + "predictions"):
        os.makedirs(config.log_dir + "predictions", exist_ok=True)
     
    filename = config.log_dir + "predictions/" + "_".join( config.ckp.split("/")[-2:] ).replace(".pt", "")

    np.save(filename, y_pred)

    with open(filename + ".json", "w") as outfile:
        json.dump(y_pred.reshape(-1, 2, 14, 13*3).tolist(), outfile)

    return model


if __name__ == '__main__':
    
    config = parse_args()

    print("Config:", config)
    
    assert config.ckp is not None and len(config.ckp) > 0, "Path to model file needs to be defined"

    model = test(config)

