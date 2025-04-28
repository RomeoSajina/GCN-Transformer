import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os

from dataset.data_utils import *
from dataset.amass import load_amass


def create_3dpw_test_out_json_if_needed(config):
    
    dir_path = config.somof_dataset_path
    
    if os.path.exists(dir_path + "3dpw_test_out.json"):
        return
    
    ds, _ = load_3dpw(split="test", out_type="dict")
    
    with open(dir_path + "3dpw_test_in.json") as f:
        X = np.array(json.load(f))
    with open(dir_path + "3dpw_test_frames_in.json") as f:
        X_f = np.array(json.load(f))
    
    i = X_f[0]

    y_test = []
    for i in X_f:
        key, last_idx = i[-1].split("/")[0], int( i[-1].split("/")[-1].split(".jpg")[0].split("_")[1] )

        indicies = np.arange(last_idx+2, last_idx+2+14*2, 2)

        y_i_test = ds[key][:, indicies]
        y_test.append(y_i_test)
    
    print("Creating '3dpw_test_out.json' file...")
    with open(dir_path + "3dpw_test_out.json", "w") as outfile:
        json.dump(np.array(y_test).tolist(), outfile)

    
class SoMof3DPW(Dataset):

    def __init__(self, config, name="test"):
        self.config = config
        dir_path = config.somof_dataset_path
        
        if name == "test":
            create_3dpw_test_out_json_if_needed(config)

        with open(dir_path + "3dpw_{0}_in.json".format(name)) as f:
            X = np.array(json.load(f))

        with open(dir_path + "3dpw_{0}_out.json".format(name)) as f:
            Y = np.array(json.load(f))

        X = X if X.shape[-1] == 3 else X.reshape(*X.shape[:-1], 13, 3)
        Y = Y if Y.shape[-1] == 3 else Y.reshape(*Y.shape[:-1], 13, 3)

        data = np.concatenate((X, Y), axis=2)

        self.data = torch.from_numpy(data).requires_grad_(False).to(config.device)

        print("Loaded SoMoF_3DPW '{0}', number of examples:".format(name), self.data.shape[0])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        return self.data[idx]


class Train3DPWAmass(Dataset):

    def __init__(self, config, x3dpw, x3dpw_single, xamass):
        
        self.config = config
        
        self.use_augmentation = True
        self.use_amass = True
        
        self.xamass = torch.from_numpy(xamass).to(config.device)
        self.x3dpw = torch.from_numpy(x3dpw).to(config.device)
        self.x3dpw_single = torch.from_numpy(x3dpw_single).to(config.device)
        
        self.T = self.config.input_len + self.config.output_len
        
        print("Loaded 3DPWAmass, number of examples:", len(self))

    def __len__(self):
        return self.x3dpw.shape[0] + self.x3dpw_single.shape[0] + (self.xamass.shape[0] if self.use_amass else 0)

    def __getitem__(self, idx):
        is_amass = False

        if idx < self.x3dpw.shape[0]:
            seq0 = self.x3dpw[idx, 0]
            seq1 = self.x3dpw[idx, 1]

        elif idx < self.x3dpw.shape[0] + self.x3dpw_single.shape[0]:    
            idx -= self.x3dpw.shape[0]
            partner_idx = np.random.randint(0, self.x3dpw_single.shape[0])
            seq0 = self.x3dpw_single[idx, 0]
            seq1 = self.x3dpw_single[partner_idx, 0]
                 
        elif self.use_amass and idx < self.x3dpw.shape[0] + self.x3dpw_single.shape[0] + self.xamass.shape[0]:
            idx -= self.x3dpw.shape[0] + self.x3dpw_single.shape[0]
            partner_idx = np.random.randint(0, self.xamass.shape[0])
            seq0 = self.xamass[idx, 0]
            seq1 = self.xamass[partner_idx, 0]
            is_amass = True

        if self.use_augmentation:
            seq0, seq1 = self._augment(seq0.clone(), seq1.clone(), is_amass)
            
        sample = torch.cat((seq0.unsqueeze(0), seq1.unsqueeze(0)), dim=0).requires_grad_(False).to(self.config.device)
        
        return sample

    def _augment(self, seq0, seq1, is_amass):
        
        if np.random.rand() > 0.75:
            return seq0, seq1
                        
        if np.random.rand() > 0.5: # backward movement
            seq0 = seq0[np.arange(-seq0.shape[0]+1, 1)]
            seq1 = seq1[np.arange(-seq1.shape[0]+1, 1)]
            
        if np.random.rand() > 0.5: # reversed order of people
            tmp = seq0.clone()
            seq0 = seq1.clone()
            seq1 = tmp

        if np.random.rand() > 0.5: # random scale
            r1 = 0.1 #0.8
            r2 = 5.0 #1.2
            def _rand_scale(_x):
                if np.random.rand() > 0.5:
                    rnd = ((r1 - r2) * np.random.rand() + r2)
                    scld = _x * rnd
                    scld += (_x[:, 7] - scld[:, 7]).reshape(-1, 1, 3) # restore global position
                    return scld
                return _x
            seq0 = _rand_scale(seq0)
            seq1 = _rand_scale(seq1)
            
        if np.random.rand() > 0.5:
            seq0, seq1 = random_reposition_sequences(seq0, seq1, device=self.config.device)

        if np.random.rand() > 0.75:
            seq0, seq1 = random_rotate_sequences(seq0, seq1, rotate_around="y", device=self.config.device)
        if np.random.rand() > 0.75:
            seq0, seq1 = random_rotate_sequences(seq0, seq1, rotate_around="x", device=self.config.device)
        if np.random.rand() > 0.75:
            seq0, seq1 = random_rotate_sequences(seq0, seq1, rotate_around="z", device=self.config.device)

        # Random joint order:
        if np.random.rand() > 0.75:
            order = torch.randperm(self.config.J)
            seq0 = seq0[:, order]
            order = torch.randperm(self.config.J)
            seq1 = seq1[:, order]

        # Random xyz-axis order:
        if np.random.rand() > 0.75:
            order = torch.randperm(3)
            seq0 = seq0[..., order]
            order = torch.randperm(3)
            seq1 = seq1[..., order]
        
        return seq0, seq1


def create_full_datasets(config):

    IN_L, OUT_L, PTH = config.input_len, config.output_len, config.threedpw_dataset_path
    num_workers = 0 #if config.device == "cuda" else 10
    
    if config.finetune:
        print("Loading *3dpfullw* dataset for training...")
        xy2, xy1 = [], []
        _xy2, _xy1 = load_original_3dw(PTH, input_window=IN_L, output_window=OUT_L, frequency=2)
        xy2.append(_xy2)
        xy1.append(_xy1)

        _xy2, _xy1 = load_original_3dw(PTH, input_window=IN_L, output_window=OUT_L, frequency=2, split="valid")
        xy2.append(_xy2)
        xy1.append(_xy1)

        _, _xy1 = load_original_3dw(PTH, input_window=IN_L, output_window=OUT_L, frequency=2, split="test")
        xy1.append(_xy1)
        
        pfds = Train3DPWAmass(config=config, 
                              x3dpw=np.concatenate(xy2, axis=0),
                              x3dpw_single=np.concatenate(xy1, axis=0),
                              xamass=np.array([]))
        pfds.use_amass = False
        
        train_loader = DataLoader(pfds, batch_size=config.batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        
    else:
        print("Loading *3dpwfull+amass* dataset for training...")
        x_amass = load_amass(config=config)
        
        xy2, xy1 = [], []
        for freq in range(1, 4):
            _xy2, _xy1 = load_original_3dw(PTH, input_window=IN_L, output_window=OUT_L, frequency=freq)
            xy2.append(_xy2)
            xy1.append(_xy1)

            _xy2, _xy1 = load_original_3dw(PTH, input_window=IN_L, output_window=OUT_L, frequency=freq, split="valid")
            xy2.append(_xy2)
            xy1.append(_xy1)
           
            _, _xy1 = load_original_3dw(PTH, input_window=IN_L, output_window=OUT_L, frequency=freq, split="test")
            xy1.append(_xy1)
            
        pfds = Train3DPWAmass(config=config, 
                              x3dpw=np.concatenate(xy2, axis=0), 
                              x3dpw_single=np.concatenate(xy1, axis=0),
                              xamass=x_amass.reshape(-1, 1, IN_L+OUT_L, config.J, 3))

        train_loader = DataLoader(pfds, batch_size=config.batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    print("Loading *somofvalid* dataset for validation...")
    valid_loader = DataLoader(SoMof3DPW(config=config, name="valid"), 
                              batch_size=config.batch_size, 
                              shuffle=False, 
                              num_workers=num_workers)

    print("Loading *somoftest* dataset for testing...")
    test_loader = DataLoader(SoMof3DPW(config=config, name="test"), 
                             batch_size=config.batch_size, 
                             shuffle=False, 
                             num_workers=num_workers)

    return train_loader, valid_loader, test_loader


def create_datasets(config):

    IN_L, OUT_L, PTH = config.input_len, config.output_len, config.threedpw_dataset_path
    num_workers = 0 #if config.device == "cuda" else 10
    
    if config.finetune:
        print("Loading *3dpw* dataset for training...")
        xy2, xy1 = load_original_3dw(PTH, input_window=IN_L, output_window=OUT_L, frequency=2)
        
        pfds = Train3DPWAmass(config=config, x3dpw=xy2, x3dpw_single=xy1, xamass=np.array([]))
        pfds.use_amass = False
        
        train_loader = DataLoader(pfds, batch_size=config.batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        
    else:
        print("Loading *3dpw+amass* dataset for training...")
        x_amass = load_amass(config=config)
        
        xy2, xy1 = [], []
        for freq in range(1, 4):
            _xy2, _xy1 = load_original_3dw(PTH, input_window=IN_L, output_window=OUT_L, frequency=freq)
            xy2.append(_xy2)
            xy1.append(_xy1)
            
        pfds = Train3DPWAmass(config=config, 
                              x3dpw=np.concatenate(xy2, axis=0), 
                              x3dpw_single=np.concatenate(xy1, axis=0),
                              xamass=x_amass.reshape(-1, 1, IN_L+OUT_L, config.J, 3))

        train_loader = DataLoader(pfds, batch_size=config.batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    print("Loading *somofvalid* dataset for validation...")
    valid_loader = DataLoader(SoMof3DPW(config=config, name="valid"), 
                              batch_size=config.batch_size, 
                              shuffle=False, 
                              num_workers=num_workers)

    print("Loading *somoftest* dataset for testing...")
    test_loader = DataLoader(SoMof3DPW(config=config, name="test"), 
                             batch_size=config.batch_size, 
                             shuffle=False, 
                             num_workers=num_workers)

    return train_loader, valid_loader, test_loader


def create_ablation_datasets(config):

    IN_L, OUT_L, PTH = config.input_len, config.output_len, config.threedpw_dataset_path
    num_workers = 0 #if config.device == "cuda" else 10
    
    print("Loading *3dpw* dataset for training...")
    if "augmentation" in config.ablation_exclude:
        xy2, xy1 = load_original_3dw(PTH, input_window=IN_L, output_window=OUT_L, frequency=2)
    else:
        xy2, xy1 = [], []
        for freq in range(1, 4):
            _xy2, _xy1 = load_original_3dw(PTH, input_window=IN_L, output_window=OUT_L, frequency=freq)
            xy2.append(_xy2)
            xy1.append(_xy1)            
        xy2 = np.concatenate(xy2, axis=0)
        xy1 = np.concatenate(xy1, axis=0)

    pfds = Train3DPWAmass(config=config, x3dpw=xy2, x3dpw_single=xy1, xamass=np.array([]))
    pfds.use_amass = False

    train_loader = DataLoader(pfds, batch_size=config.batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    print("Loading *somofvalid* dataset for validation...")
    valid_loader = DataLoader(SoMof3DPW(config=config, name="valid"), 
                              batch_size=config.batch_size, 
                              shuffle=False, 
                              num_workers=num_workers)

    return train_loader, None, valid_loader
