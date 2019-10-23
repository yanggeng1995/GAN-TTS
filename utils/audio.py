import numpy as np
import librosa
import scipy

sample_rate = 24000
n_fft = 2048
fft_bins = n_fft // 2 + 1
num_mels = 80
#frame_length_ms=50
#frame_shift_ms=12.5
hop_length = 120  #frame_shift_ms * sample_rate / 1000
win_length = 240  #frame_length_ms * sample_rate / 1000
fmin = 40
min_level_db = -100
ref_level_db = 20

def convert_audio(wav_path):
    wav = load_wav(wav_path)
    mel = melspectrogram(wav).astype(np.float32)
    return mel.transpose(), wav

def load_wav(filename) :
    x = librosa.load(filename, sr=sample_rate)[0]
    return x

def save_wav(y, filename) :
    scipy.io.wavfile.write(filename, sample_rate, y)

mel_basis = None

def linear_to_mel(spectrogram):
    global mel_basis
    if mel_basis is None:
        mel_basis = build_mel_basis()
    return np.dot(mel_basis, spectrogram)

def build_mel_basis():
    return librosa.filters.mel(sample_rate, n_fft, n_mels=num_mels, fmin=fmin)

def normalize(S):
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)

def denormalize(S):
    return (np.clip(S, 0, 1) * -min_level_db) + min_level_db

def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))

def db_to_amp(x):
    return np.power(10.0, x * 0.05)

def spectrogram(y):
    D = stft(y)
    S = amp_to_db(np.abs(D)) - ref_level_db
    return normalize(S)

def melspectrogram(y):
    D = stft(y)
    S = amp_to_db(linear_to_mel(np.abs(D)))
    return normalize(S)

def stft(y):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
