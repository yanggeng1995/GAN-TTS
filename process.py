import pickle
import os
import glob
import argparse
import numpy as np
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from utils.audio import convert_audio, hop_length, sample_rate
from tqdm import tqdm
import random

train_rate = 0.9995
test_rate  = 0.0005

def find_files(path, pattren="*.wav"):
    filenames = []
    for filename in glob.iglob(f'{path}/**/*{pattren}', recursive=True):
       filenames.append(filename)
    return filenames

def data_prepare(audio_path, mel_path, wav_file):
    mel, audio = convert_audio(wav_file)
    np.save(audio_path, audio, allow_pickle=False)
    np.save(mel_path, mel, allow_pickle=False)
    return audio_path, mel_path, mel.shape[0]

def process(output_dir, wav_files, train_dir, test_dir,  num_workers):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    results = []
    names = []

    random.shuffle(wav_files)
    train_num = int(len(wav_files) * train_rate)

    for wav_file in wav_files[0 : train_num]:
        fid = os.path.basename(wav_file).replace('.wav','.npy')
        names.append(fid)
        results.append(executor.submit(partial(data_prepare, os.path.join(train_dir, "audio", fid), os.path.join(train_dir, "mel", fid), wav_file)))

    with open(os.path.join(output_dir, "train", 'names.pkl'), 'wb') as f:
        pickle.dump(names, f)

    names = []
    for wav_file in wav_files[train_num : len(wav_files)]:
        fid = os.path.basename(wav_file).replace('.wav','.npy')
        names.append(fid)
        results.append(executor.submit(partial(data_prepare, os.path.join(test_dir, "audio", fid), os.path.join(test_dir, "mel", fid), wav_file)))

    with open(os.path.join(output_dir, "test", 'names.pkl'), 'wb') as f:
        pickle.dump(names, f)


    return [result.result() for result in tqdm(results)]

def preprocess(args):
    train_dir = os.path.join(args.output, 'train')
    test_dir = os.path.join(args.output, 'test')
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(os.path.join(train_dir, "audio"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "mel"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "audio"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "mel"), exist_ok=True)

    wav_files = find_files(args.wav_dir)
    metadata = process(args.output, wav_files, train_dir, test_dir, args.num_workers)
    write_metadata(metadata, args.output)

def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'metadata.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    frames = sum([m[2] for m in metadata])
    frame_shift_ms = hop_length * 1000 / sample_rate
    hours = frames * frame_shift_ms / (3600 * 1000)
    print('Write %d utterances, %d frames (%.2f hours)' % (len(metadata), frames, hours))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_dir', default='wavs')
    parser.add_argument('--output', default='data')
    parser.add_argument('--num_workers', type=int, default=int(cpu_count()))
    args = parser.parse_args()
    preprocess(args)

if __name__ == "__main__":
    main()
