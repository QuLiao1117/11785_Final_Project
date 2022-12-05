import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import seaborn as sns
import matplotlib.pyplot as plt
import random
import torchaudio
from torchaudio import transforms

class AudioDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, am_path, gender = "female", phoneme_idx = 4, am_idx = 1, MAX_LEN = 128, partition = "train"):
        """
        :param data_path: the root path of phonemes
        :param am_path: the path of am (.csv)
        :param gender: female or male
        :param phoneme_idx: the phoneme index
        :param am_idx: the index of target AM, should be int within [1, 96]
        :param MAX_LEN: max length of voice seq, if less, pad, if more, slice
        :param partition: train / val1 / val2 / test
        """

        self.MAX_LEN = MAX_LEN
        # get phoneme list
        self.target_phoneme_path = "/".join([data_path, gender, str(int(phoneme_idx))])
        phoneme_list = sorted(os.listdir(self.target_phoneme_path))
        length = len(phoneme_list)
        if partition == "train":
            self.phoneme_list = phoneme_list[:int(0.7 * length)]
        elif partition == "val1":
            self.phoneme_list = phoneme_list[int(0.7 * length):int(0.8 * length)]
        elif partition == "val2":
            self.phoneme_list = phoneme_list[int(0.8 * length):int(0.9 * length)]
        elif partition == "test":
            self.phoneme_list = phoneme_list[int(0.9 * length):]

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

    def __getitem__(self, ind):
        item_filename = self.phoneme_list[ind]
        item_full_path = "/".join([self.target_phoneme_path, item_filename])
        phoneme = np.load(item_full_path)

        person_id = int(item_filename.split("_")[0][1:7])
        try:
            target_am = self.am_data[self.am_data["ID"] == person_id].values[0][-1]
        except:
            print("person id =", person_id)
            target_am = 0.

        # padding
        phoneme = torch.tensor(phoneme, dtype=torch.float) #.reshape(1, -1)
        # apply mel transform
        phoneme = self.spectro_gram(phoneme)

        std, mean = torch.std_mean(phoneme, unbiased=False, dim=0)
        phoneme = (phoneme - mean) / (std + 1e-6)

        if len(phoneme[0]) < MAX_LEN:
            phoneme = np.pad(phoneme, ((0, 0), (0, MAX_LEN - len(phoneme[0]))), 'symmetric')
        else:
            phoneme = phoneme[:, :MAX_LEN]

        target_am = torch.tensor(target_am)
        return phoneme, target_am



if __name__ == "__main__":
    default_root_path = "D:/File/study/CMU/11785/project/penstate_data/extract_phoneme"
    gender = "female"
    phoneme_idx = 13
    am_path = "D:/File/study/CMU/11785/project/penstate_data/AMs_final.csv"
    am_idx = 20
    MAX_LEN = 32
    batch_size = 64

    train_data = AudioDataset(data_path=default_root_path,
                              am_path = am_path,
                              gender = gender, phoneme_idx = phoneme_idx, am_idx = am_idx, MAX_LEN = MAX_LEN, partition="val1")

    train_loader = torch.utils.data.DataLoader(train_data, num_workers=0,
                                               batch_size=batch_size)

    print("Batch size: ", batch_size)
    print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))

    for i, data in enumerate(train_loader):
        phoneme, target_am = data
        sns.heatmap(phoneme[0], cmap="rainbow")
        plt.show()
        print(phoneme.shape, target_am.shape)
        # break