import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os


class AudioDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, am_path, gender = "female", phoneme_idx = 4, am_idx = 1):

        # get phoneme list
        self.target_phoneme_path = "/".join([data_path, gender, str(int(phoneme_idx))])
        self.phoneme_list = os.listdir(self.target_phoneme_path)
        self.length = len(self.phoneme_list)

        # get_am data
        am_data = pd.read_csv(am_path)
        self.am_data = am_data[["ID", str(am_idx)]]

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        item_filename = self.phoneme_list[ind]
        item_full_path = "/".join([self.target_phoneme_path, item_filename])
        phoneme = np.load(item_full_path)

        person_id = int(item_filename.split("_")[0][1:])
        target_am = self.am_data[self.am_data["ID"] == person_id].values[-1]

        return phoneme, target_am



if __name__ == "__main__":
    default_root_path = "D:/File/study/CMU/11785/project/penstate_data/extract_phoneme"
    gender = "female"
    phoneme_idx = 4

    batch_size = 64

    train_data = AudioDataset(data_path=default_root_path,
                              am_path = "D:/File/study/CMU/11785/project/penstate_data/AMs_unnormalized.csv",
                              gender = "female", phoneme_idx = 13, am_idx = 10)

    train_loader = torch.utils.data.DataLoader(train_data, num_workers=0,
                                               batch_size=64)

    print("Batch size: ", batch_size)
    print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))