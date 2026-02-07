
import tensorflow as tf
from tensorflow import keras
import scipy
import numpy as np
import os
from scipy.signal import welch
import librosa
import librosa.display
import cv2
import matplotlib.pyplot as plt


def load_audio_files_from_folders(fake_dir, real_dir):
    real=[]
    fake=[]

    for f in os.listdir(real_dir):
        if f.endswith(".wav"):
            real.append(os.path.join(real_dir, f))

    for f in os.listdir(fake_dir):
        if f.endswith(".wav"):
            fake.append(os.path.join(fake_dir, f))

    return real, fake

real_dir, fake_dir = load_audio_files_from_folders('/Users/tarunc/Personal/deep_busters/fake', '/Users/tarunc/Personal/deep_busters/real')

real_data=[]
fake_data=[]

for dir in real_dir:
    real_data.append(np.float32(librosa.load(dir, sr=16000, mono=True)[0]))

for dir in fake_dir:
    fake_data.append(np.float32(librosa.load(dir, sr=16000, mono=True)[0]))

real_spectro_data=[]
fake_spectro_data=[]

for data in real_data:
    spectro = librosa.feature.melspectrogram(y=data, sr=16000, n_fft=1024,
    hop_length=160,
    win_length=400,    
    window="hann",
    n_mels=64,
    fmin=20,
    fmax=7600)
    real_spectro_data.append(librosa.power_to_db(spectro, ref=np.max))
    
for data in fake_data:
    spectro = librosa.feature.melspectrogram(y=data, sr=16000, n_fft=1024,
    hop_length=160,
    win_length=400,
    window="hann",
    n_mels=64,
    fmin=20,
    fmax=7600)
    fake_spectro_data.append(librosa.power_to_db(spectro, ref=np.max))

print(np.array(real_spectro_data).shape)

plt.figure(figsize=(10, 4))
librosa.display.specshow(real_spectro_data[0], sr=16000, hop_length=160, x_axis="time", y_axis="mel")
plt.colorbar(format="%+2.0f dB")
plt.title("Real Log-Mel Spectrogram")
plt.tight_layout()
plt.savefig("real_spectrogram.png")
plt.close()

plt.figure(figsize=(10, 4))
librosa.display.specshow(fake_spectro_data[0], sr=16000, hop_length=160, x_axis="time", y_axis="mel")
plt.colorbar(format="%+2.0f dB")
plt.title("Fake Log-Mel Spectrogram")
plt.tight_layout()
plt.savefig("fake_spectrogram.png")
plt.close()