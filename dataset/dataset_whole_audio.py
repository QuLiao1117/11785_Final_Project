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

    def __init__(self, data_path, am_path, gender="Female_processed", am_idx=1, MAX_LEN=128, partition="train"):
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
        self.target_voice_path = "/".join([data_path, gender])
        voice_list = sorted(os.listdir(self.target_voice_path))
        random.shuffle(voice_list)
        length = len(voice_list)
        if partition == "train":
            self.voice_list = voice_list[:int(0.7 * length)]
        elif partition == "val1":
            self.voice_list = voice_list[int(0.7 * length):int(0.8 * length)]
        elif partition == "val2":
            self.voice_list = voice_list[int(0.8 * length):int(0.9 * length)]
        elif partition == "test":
            self.voice_list = voice_list[int(0.9 * length):]

        # if partition == "train":
        #     self.phoneme_list = phoneme_list[:int(0.7 * length)]
        # elif partition == "val1":
        #     self.phoneme_list = phoneme_list[int(0.7 * length):]

        self.length = len(self.voice_list)

        # get_am data
        am_data = pd.read_csv(am_path)
        self.am_data = am_data[["ID", str(am_idx)]]

    def __len__(self):
        return self.length

    def spectro_gram(self, sig, rate_of_sample=44100, n_mels=64, n_fft=512, hop_len=None):
        # top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(
            sample_rate=rate_of_sample, n_fft=n_fft,
            win_length=400, hop_length=160, n_mels=n_mels)(sig)

        # Convert to decibels
        # spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return spec

    def __getitem__(self, ind):
        item_filename = self.voice_list[ind]
        item_full_path = "/".join([self.target_voice_path, item_filename])

        data_waveform, rate_of_sample = torchaudio.load(item_full_path)
        # voice = np.load(item_full_path)

        person_id = int(item_filename[1:7])
        try:
            target_am = self.am_data[self.am_data["ID"] == person_id].values[0][-1]
        except:
            print("person id =", person_id)
            target_am = 0.

        # padding
        data_waveform = torch.tensor(data_waveform, dtype=torch.float)  # .reshape(1, -1)
        # apply mel transform
        data_waveform = self.spectro_gram(data_waveform, rate_of_sample)

        std, mean = torch.std_mean(data_waveform, unbiased=False, dim=0)
        data_waveform = (data_waveform - mean) / (std + 1e-6)
        # print(data_waveform.shape)
        if data_waveform.shape[2] < MAX_LEN:
            # data_waveform = np.pad(data_waveform, ((0, 0),(0,0), (0, MAX_LEN - data_waveform.shape[2])), 'symmetric'), 'constant', constant_values=(0, 0)
            data_waveform = np.pad(data_waveform, ((0, 0), (0, 0), (0, MAX_LEN - data_waveform.shape[2])), 'constant',
                                   constant_values=(0, 0))

            data_waveform = torch.from_numpy(data_waveform)
        else:
            temp_start = random.randint(0, data_waveform.shape[2] - MAX_LEN)
            data_waveform = data_waveform[:, :, temp_start:temp_start + MAX_LEN]
        # print(data_waveform.shape)
        # phoneme = torch.from_numpy(phoneme)
        ##################################################################
        # data_waveform.unsqueeze_(0)
        ##################################################################
        target_am = torch.tensor(target_am).to(torch.float32)

        return data_waveform, target_am


if __name__ == "__main__":
    default_root_path = "./penstate_data/download/Full_voice_files"

    am_path = "./penstate_data/AMs_final.csv"

    gender = "Female_processed"  # Male_processed
    am_idx = 89

    MAX_LEN = 4096  # TODO: may be too small
    batch_size = 64
    # batch_size = 4
    train_data = AudioDataset(data_path=default_root_path,
                              am_path=am_path,
                              gender=gender, am_idx=am_idx, MAX_LEN=MAX_LEN, partition="train")

    val_data = AudioDataset(data_path=default_root_path,
                            am_path=am_path,
                            gender=gender, am_idx=am_idx, MAX_LEN=MAX_LEN, partition="val1")
    test_data = AudioDataset(data_path=default_root_path,
                             am_path=am_path,
                             gender=gender, am_idx=am_idx, MAX_LEN=MAX_LEN, partition="test")
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=0,
                                               batch_size=batch_size, shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_data, num_workers=0,
                                             batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=0,
                                              batch_size=batch_size)
    print("Batch size: ", batch_size)

    print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
    print("Validation dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))
    print("Test dataset samples = {}, batches = {}".format(test_data.__len__(), len(test_loader)))