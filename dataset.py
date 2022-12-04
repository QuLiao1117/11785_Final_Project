import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import random
import torchaudio
from torchaudio import transforms

class AudioDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, am_path, gender = "female", phoneme_idx = 4, am_idx = 1, MAX_LEN = 44100 * 2):

        self.MAX_LEN = MAX_LEN
        # get phoneme list
        self.target_phoneme_path = "/".join([data_path, gender, str(int(phoneme_idx))])
        self.phoneme_list = os.listdir(self.target_phoneme_path)
        self.length = len(self.phoneme_list)

        # get_am data
        am_data = pd.read_csv(am_path)
        self.am_data = am_data[["ID", str(am_idx)]]

    def __len__(self):
        return self.length

    def spectro_gram(self, sig, n_mels=64, n_fft=1024, hop_len=None):
        top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(44100, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return spec

    def padding(self, phoneme):
        if len(phoneme) < self.MAX_LEN:
            pad_begin_len = random.randint(0, self.MAX_LEN - len(phoneme))
            pad_end_len = self.MAX_LEN - len(phoneme) - pad_begin_len

            # Pad with 0s
            pad_begin = np.zeros(pad_begin_len)
            pad_end = np.zeros(pad_end_len)

            phoneme = np.concatenate((pad_begin, phoneme, pad_end), 0)
        else:
            phoneme = phoneme[:self.MAX_LEN]
        return phoneme

    def __getitem__(self, ind):
        item_filename = self.phoneme_list[ind]
        item_full_path = "/".join([self.target_phoneme_path, item_filename])
        phoneme = np.load(item_full_path)

        person_id = int(item_filename.split("_")[0][1:])
        target_am = self.am_data[self.am_data["ID"] == person_id].values[0][-1]

        # padding
        phoneme = self.padding(phoneme)
        phoneme = torch.tensor(phoneme, dtype=torch.float) #.reshape(1, -1)
        # apply mel transform
        phoneme = self.spectro_gram(phoneme)

        target_am = torch.tensor(target_am)
        return phoneme, target_am



if __name__ == "__main__":
    default_root_path = "D:/File/study/CMU/11785/project/penstate_data/extract_phoneme"
    gender = "female"
    phoneme_idx = 5
    am_path = "D:/File/study/CMU/11785/project/penstate_data/AMs_unnormalized.csv"
    am_idx = 10
    MAX_LEN = 44100 * 3
    batch_size = 64

    train_data = AudioDataset(data_path=default_root_path,
                              am_path = am_path,
                              gender = gender, phoneme_idx = phoneme_idx, am_idx = am_idx, MAX_LEN = MAX_LEN)

    train_loader = torch.utils.data.DataLoader(train_data, num_workers=0,
                                               batch_size=batch_size)

    print("Batch size: ", batch_size)
    print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))

    for i, data in enumerate(train_loader):
        phoneme, target_am = data
        print(phoneme.shape, target_am.shape)
        break