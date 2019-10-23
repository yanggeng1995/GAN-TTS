# GAN-TTS
A pytorch implementation of the GAN-TTS: HIGH FIDELITY SPEECH SYNTHESIS WITH ADVERSARIAL NETWORKS(https://arxiv.org/pdf/1909.11646.pdf)

## Prepare dataset
* Download dataset for training. This can be any wav files with sample rate 24000Hz.
* Process: python process.py.py --wav_dir="wavs" --output="data"
* Edit configuration in utils/audio.py

## Train & Tensorboard
* python train.py --input="data/train"
* tensorboard --logdir logdir

## Attention
* I did not use the loss function mentioned in the paper. I modified the loss function and learn from MelGAN(https://arxiv.org/pdf/1910.06711.pdf).

## Notes
* This is not official implementation, some details are not necessarily correct.
* Work in progress.
