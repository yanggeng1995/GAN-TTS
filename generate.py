import torch
from utils.audio import save_wav
import argparse
import os
import time
import numpy as np
from models.modules import Generator

def attempt_to_restore(generate, checkpoint_dir, use_cuda):
    checkpoint_list = os.path.join(checkpoint_dir, 'checkpoint')

    if os.path.exists(checkpoint_list):
        checkpoint_filename = open(checkpoint_list).readline().strip()
        checkpoint_path = os.path.join(
            checkpoint_dir, "{}".format(checkpoint_filename))
        print("Restore from {}".format(checkpoint_path))
        checkpoint = load_checkpoint(checkpoint_path, use_cuda)
        generate.load_state_dict(checkpoint["generator"])

def load_checkpoint(checkpoint_path, use_cuda):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint

def create_model(args):

    generator = Generator(args.local_condition_dim, args.z_dim)
    
    return generator 

def synthesis(args):

    model = create_model(args)
    if args.resume is not None:
       attempt_to_restore(model, args.resume, args.use_cuda)

    device = torch.device("cuda" if args.use_cuda else "cpu")

    output_dir = "samples"
    os.makedirs(output_dir, exist_ok=True)

    lists = []
    for filename in os.listdir(os.path.join(args.input, 'mel')):
        lists.append(filename)
    start = time.time()
    conditions = [np.load(os.path.join(args.input, 'mel', filename))
            for filename in lists]
    lengths = [condition.shape[0] for condition in conditions]
    max_len = max(lengths)
    conditions = [np.concatenate((condition, np.zeros((max_len - condition.shape[0],
        condition.shape[1]))), axis=0) for condition in conditions]

    conditions = np.stack(conditions)
    conditions = torch.FloatTensor(conditions)
    conditions = conditions.transpose(1, 2).to(device)
    batch_size = conditions.size()[0]
    z = torch.randn(batch_size, args.z_dim).to(device)
    print(conditions.shape)
    audios = model(conditions, z)
    audios = audios.squeeze().detach().numpy()
    print(audios.shape)
    for (i, filename) in enumerate(lists):
        name = filename.split('.')[0]
        sample = np.load(os.path.join(args.input, 'audio', filename))
        save_wav(np.squeeze(sample), '{}/{}_target.wav'.format(output_dir, name))
        save_wav(np.asarray(audios[i])[:len(sample)], '{}/{}.wav'.format(output_dir, name))
    print("Time used: {:.3f}".format(time.time() - start))

def main():

    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]


    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/test', help='Directory of tests data')
    parser.add_argument('--num_workers',type=int, default=4, help='Number of dataloader workers.')
    parser.add_argument('--resume', type=str, default="ema_logdir")
    parser.add_argument('--local_condition_dim', type=int, default=80)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--use_cuda', type=_str_to_bool, default=False)

    args = parser.parse_args()
    synthesis(args)

if __name__ == "__main__":
    main()
