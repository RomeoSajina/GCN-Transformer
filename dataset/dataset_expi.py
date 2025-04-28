"""
https://arxiv.org/pdf/2105.08825.pdf
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os

from dataset.data_utils import *
import numpy as np
import os


def to_seq_length(seq_list, SL, freq):

    seqs = []
    for orig_seq in [x for x in seq_list if x.shape[1] >= SL*freq]:

        for sampled_s in [orig_seq[:, i::freq] for i in range(freq)]:

            seqs.extend( [sampled_s[:, i:i+SL] for i in range(sampled_s.shape[1]-SL)] ) 

    return np.array(seqs)


def readCSVasFloat(filename, with_key=True):
    returnArray = []
    lines = open(filename).readlines()
    if with_key: # skip first line
        lines = lines[1:]
    for line in lines:
        line = line.strip().split(',')
        if len(line) > 0:
            returnArray.append(np.array([np.float32(x) for x in line]))

    returnArray = np.array(returnArray)
    return returnArray


def load_expi_dataset(split="train", expi_dir="./data/expi/ExPI_mocap_data/", SL=30, freq=2):
    ORDER_3DPW_FROM_EXPI = [10, 11, 12, 13, 14, 15, 0, 4, 5, 6, 7, 8, 9]
    
    splits = {"train": "acro2", "test": "acro1", "valid": "acro2"}

    seq_list = []
    sub_fld = splits[split]
    for folder in os.listdir(expi_dir + sub_fld):
        if folder.startswith("."):
            continue

        tsv_file = expi_dir + sub_fld + os.path.sep + folder + os.path.sep + "/mocap_cleaned.tsv"
        the_sequence = readCSVasFloat(tsv_file, with_key=True)

        the_sequence = the_sequence.reshape(-1, 1, 36, 3)

        seq = np.concatenate((the_sequence[:, :, :18], the_sequence[:, :, 18:]), axis=1)

        seq = seq[:, :, ORDER_3DPW_FROM_EXPI]
        seq = seq.transpose(1, 0, 2, 3)

        # scale it down
        dists = np.linalg.norm(seq[:, 0, 0] - seq[:, 0, 2], axis=-1) # shape: N
        # scale
        scales = dists / 0.39
        scales = np.mean(scales) # keep people in the scene same height ratio
        seq = seq / scales.reshape(-1, 1, 1, 1)

        seq_list.append(seq)

    seq_list = to_seq_length(seq_list, SL, freq)

    if split == "train":
        return seq_list
    else:
        return seq_list[::SL]


class ExPI(Dataset):

    def __init__(self, config, data):
        self.config = config
        
        self.data = torch.from_numpy(data).requires_grad_(False).to(config.device)

        print("Loaded ExPI, number of examples:", self.data.shape[0])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


class TrainExPI(Dataset):

    def __init__(self, config, data):
        
        self.config = config
        
        self.use_augmentation = True

        self.data = torch.from_numpy(data).to(config.device)

        self.T = self.config.input_len + self.config.output_len
        
        print("Loaded Train ExPI, number of examples:", len(self))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        seq0 = self.data[idx, 0]
        seq1 = self.data[idx, 1]

        if self.use_augmentation:
            seq0, seq1 = self._augment(seq0.clone(), seq1.clone())
            
        sample = torch.cat((seq0.unsqueeze(0), seq1.unsqueeze(0)), dim=0).requires_grad_(False).to(self.config.device)
        
        return sample

    def _augment(self, seq0, seq1):
        
        if np.random.rand() > 0.5:
            return seq0, seq1
                        
        if np.random.rand() > 0.5: # backward movement
            seq0 = seq0[np.arange(-seq0.shape[0]+1, 1)]
            seq1 = seq1[np.arange(-seq1.shape[0]+1, 1)]
            
        if np.random.rand() > 0.5: # reversed order of people
            tmp = seq0.clone()
            seq0 = seq1.clone()
            seq1 = tmp

        if np.random.rand() > 0.5: # random scale
            r1 = 0.1
            r2 = 5.0
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


def create_datasets(config):

    num_workers = 0 #if config.device == "cuda" else 10
    
    if config.finetune:
        data = load_expi_dataset(split="train")
    else:
        data = np.concatenate([load_expi_dataset(split="train", freq=freq) for freq in range(1, 4)], axis=0)
        
    pfds = TrainExPI(config=config, data=data)
    train_loader = DataLoader(pfds, batch_size=config.batch_size, shuffle=True, num_workers=num_workers)

    test_ds = ExPI(config=config, data=load_expi_dataset(split="test"))
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, test_loader











