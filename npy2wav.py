import numpy as np

import torchaudio
import torch
import matplotlib.pyplot as plt
import soundfile as sf

idx = 12

def plot_waveform(waveform, sample_rate):
    # waveform = waveform.numpy()

    num_channels, num_frames = 1, waveform.shape[0]
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
    plt.show(block=False)

for number in range(10):
# number = 4
    data = np.load(f"D:/File/study/CMU/11785/project/penstate_data/extract_phoneme/female/{idx}/F140763_{idx}_{number}.npy")
    # plot_waveform(data, 44100)
    sf.write(f'D:/File/study/CMU/11785/project/penstate_data/test{number}.wav', data, 44100)
#
# data = np.load(f"D:/File/study/CMU/11785/project/penstate_data/extract_phoneme/female/{idx}/F140763_{idx}_1.npy")
# print(data)
# sf.write('D:/File/study/CMU/11785/project/penstate_data/test2.wav', data, 44100)
#
# data = np.load(f"D:/File/study/CMU/11785/project/penstate_data/extract_phoneme/female/{idx}/F140763_{idx}_2.npy")
# print(data)
# sf.write('D:/File/study/CMU/11785/project/penstate_data/test3.wav', data, 44100)
#
# data = np.load(f"D:/File/study/CMU/11785/project/penstate_data/extract_phoneme/female/{idx}/F140763_{idx}_3.npy")
# print(data)
# sf.write('D:/File/study/CMU/11785/project/penstate_data/test4.wav', data, 44100)