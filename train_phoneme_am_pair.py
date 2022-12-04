import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import sklearn
from sklearn.metrics import accuracy_score
import pandas as pd
import os
import datetime
import torchaudio
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)


from dataset import AudioDataset


default_root_path = "D:/File/study/CMU/11785/project/penstate_data/extract_phoneme"
gender = "female_processed"
phoneme_idx = 10
am_path = "D:/File/study/CMU/11785/project/penstate_data/AMs_unnormalized.csv"
am_idx = 20
MAX_LEN = 44100 * 3
batch_size = 64

train_data = AudioDataset(data_path=default_root_path,
                          am_path = am_path,
                          gender = gender, phoneme_idx = phoneme_idx, am_idx = am_idx, MAX_LEN = MAX_LEN, partition="train")

val_data = AudioDataset(data_path=default_root_path,
                          am_path = am_path,
                          gender = gender, phoneme_idx = phoneme_idx, am_idx = am_idx, MAX_LEN = MAX_LEN, partition="val1")

train_loader = torch.utils.data.DataLoader(train_data, num_workers=0,
                                           batch_size=batch_size, pin_memory= True,
                                         shuffle= False)

val_loader = torch.utils.data.DataLoader(val_data, num_workers= 0,
                                         batch_size=batch_size, pin_memory= True,
                                         shuffle= False)

print("Batch size: ", batch_size)
print("phoneme_idx: ", phoneme_idx)
print("am_idx", am_idx)

print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
print("Validation dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))

for i, data in enumerate(train_loader):
    phoneme, target_am = data
    print(phoneme.shape, target_am.shape)
    break


# model = ?
