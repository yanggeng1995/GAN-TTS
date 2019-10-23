import torch
from utils.dataset import CustomerDataset, CustomerCollate
from torch.utils.data import DataLoader
import torch.nn.parallel.data_parallel as parallel
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import time
from models.modules import Generator, Multiple_Random_Window_Discriminators
from tensorboardX import SummaryWriter
from utils.optimizer import Optimizer
from utils.audio import hop_length
from utils.util import ExponentialMovingAverage, apply_moving_average, register_model_to_ema

def create_model(args):

    generator = Generator(args.local_condition_dim, args.z_dim)
    discriminator = Multiple_Random_Window_Discriminators(args.local_condition_dim)
    
    return generator, discriminator

def save_checkpoint(args, generator, discriminator,
                    g_optimizer, d_optimizer, step, ema=None):
    checkpoint_path = os.path.join(args.checkpoint_dir, "model.ckpt-{}.pt".format(step))

    torch.save({"generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "g_optimizer": g_optimizer.state_dict(),
                "d_optimizer": d_optimizer.state_dict(),
                "global_step": step
                }, checkpoint_path)

    if ema is not None:
        ema_checkpoint_path = os.path.join(args.ema_checkpoint_dir, "model.ckpt-{}.ema".format(step))
        averge_model = clone_as_averaged_model(args, generator, ema)
        torch.save({"generator": averge_model.state_dict(),
                    "g_optimizer": g_optimizer.state_dict(),
                    "global_step": step
                  }, ema_checkpoint_path)

    print("Saved checkpoint: {}".format(checkpoint_path))

    with open(os.path.join(args.checkpoint_dir, 'checkpoint'), 'w') as f:
        f.write("model.ckpt-{}.pt".format(step))

    with open(os.path.join(args.ema_checkpoint_dir, 'checkpoint'), 'w') as f:
        f.write("model.ckpt-{}.ema".format(step))

def attempt_to_restore(generator, discriminator, g_optimizer,
                       d_optimizer, checkpoint_dir, use_cuda):
    checkpoint_list = os.path.join(checkpoint_dir, 'checkpoint')

    if os.path.exists(checkpoint_list):
        checkpoint_filename = open(checkpoint_list).readline().strip()
        checkpoint_path = os.path.join(
            checkpoint_dir, "{}".format(checkpoint_filename))
        print("Restore from {}".format(checkpoint_path))
        checkpoint = load_checkpoint(checkpoint_path, use_cuda)
        generator.load_state_dict(checkpoint["generator"])
        g_optimizer.load_state_dict(checkpoint["g_optimizer"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        d_optimizer.load_state_dict(checkpoint["d_optimizer"])
        global_step = checkpoint["global_step"]

    else:
        global_step = 0

    return global_step

def load_checkpoint(checkpoint_path, use_cuda):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage)

    return checkpoint

def clone_as_averaged_model(args, model, ema):
    device = torch.device("cuda" if args.use_cuda else "cpu")

    assert ema is not None
    averaged_model, _ = create_model(args)
    averaged_model = averaged_model.to(device)
    averaged_model.load_state_dict(model.state_dict())
    for name, param in averaged_model.named_parameters():
        if name in ema.shadow:
            param.data = ema.shadow[name].clone()

    return averaged_model

def feature_loss_calculate(real_outputs, fake_outputs):
    loss = []
    for (real_output, fake_output) in zip(real_outputs, fake_outputs):
        loss.append(torch.abs(real_output - fake_output).mean())
    return sum(loss)

def train(args):

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.ema_checkpoint_dir, exist_ok=True)

    train_dataset = CustomerDataset(
         args.input,
         upsample_factor=hop_length,
         local_condition=True,
         global_condition=False)

    device = torch.device("cuda" if args.use_cuda else "cpu")
    generator, discriminator = create_model(args)

    print(generator)
    print(discriminator)

    num_gpu = torch.cuda.device_count() if args.use_cuda else 1

    global_step = 0

    g_parameters = list(generator.parameters())
    g_optimizer = optim.Adam(g_parameters, lr=args.g_learning_rate)

    d_parameters = list(discriminator.parameters())
    d_optimizer = optim.Adam(d_parameters, lr=args.d_learning_rate)
    
    writer = SummaryWriter(args.checkpoint_dir)

    generator.to(device)
    discriminator.to(device)

    if args.resume is not None:
        restore_step = attempt_to_restore(generator, discriminator, g_optimizer,
                                          d_optimizer, args.resume, args.use_cuda)
        global_step = restore_step

    ema = ExponentialMovingAverage(args.ema_decay)
    register_model_to_ema(generator, ema)

    customer_g_optimizer = Optimizer(g_optimizer, args.g_learning_rate,
                global_step, args.warmup_steps, args.decay_learning_rate)
    customer_d_optimizer = Optimizer(d_optimizer, args.d_learning_rate,
                global_step, args.warmup_steps, args.decay_learning_rate)

    criterion = nn.MSELoss().to(device)

    for epoch in range(args.epochs):

       collate = CustomerCollate(
           upsample_factor=hop_length,
           condition_window=args.condition_window,
           local_condition=True,
           global_condition=False)

       train_data_loader = DataLoader(train_dataset, collate_fn=collate,
               batch_size=args.batch_size, num_workers=args.num_workers,
               shuffle=True, pin_memory=True)

       #train one epoch
       for batch, (samples, conditions) in enumerate(train_data_loader):

            start = time.time()
            batch_size = int(conditions.shape[0] // num_gpu * num_gpu)

            samples = samples[:batch_size, :].to(device)
            conditions = conditions[:batch_size, :, :].to(device)
            z = torch.randn(batch_size, args.z_dim).to(device)

            #train generator
            if num_gpu > 1:
                g_outputs = parallel(generator, (conditions, z))
                _, fake_outputs, real_features, fake_features = \
                   parallel(discriminator, (samples, g_outputs, conditions))
            else:
                g_outputs = generator(conditions, z)
                _, fake_outputs, real_features, fake_features = \
                   discriminator(samples, g_outputs, conditions)

            g_d_loss = []
            for fake_output in fake_outputs:
               target = torch.ones_like(fake_output).to(device)
               g_d_loss.append(criterion(fake_output, target))
            feature_loss = feature_loss_calculate(real_features, fake_features)
            g_loss = feature_loss * args.lamda + sum(g_d_loss)
 
            customer_g_optimizer.zero_grad()
            g_loss.backward()
            nn.utils.clip_grad_norm_(g_parameters, max_norm=0.5)
            customer_g_optimizer.step_and_update_lr()

            #train discriminator
            g_outputs = g_outputs.detach()
            if num_gpu > 1:
                real_outputs, fake_outputs, _, _ = \
                    parallel(discriminator, (samples, g_outputs, conditions))
            else:
                real_outputs, fake_outputs, _, _ = \
                    discriminator(samples, g_outputs, conditions)

            fake_loss, real_loss = [], []
            for (fake_output, real_output) in zip(fake_outputs, real_outputs):
                fake_target = torch.zeros_like(fake_output).to(device)
                real_target = torch.ones_like(real_output).to(device)
                fake_loss.append(criterion(fake_output, fake_target))
                real_loss.append(criterion(real_output, real_target))
            d_loss = sum(fake_loss) + sum(real_loss)

            customer_d_optimizer.zero_grad()
            d_loss.backward()
            nn.utils.clip_grad_norm_(d_parameters, max_norm=0.5)
            customer_d_optimizer.step_and_update_lr()

            global_step += 1

            print("Step: {} --g_loss: {:.3f} --d_loss: {:.3f} --Time: {:.2f} seconds".format(
                   global_step, g_loss, d_loss, float(time.time() - start)))
            print(feature_loss.item(), sum(g_d_loss).item(), d_loss.item())
            if ema is not None:
                apply_moving_average(generator, ema)

            if global_step % args.checkpoint_step ==0:
                save_checkpoint(args, generator, discriminator,
                    g_optimizer, d_optimizer, global_step, ema)
                
            if global_step % args.summary_step == 0:
                writer.add_scalar("g_loss", g_loss.item(), global_step)
                writer.add_scalar("d_loss", d_loss.item(), global_step)

def main():

    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/train', help='Directory of training data')
    parser.add_argument('--num_workers',type=int, default=4, help='Number of dataloader workers.')
    parser.add_argument('--epochs', type=int, default=50000)
    parser.add_argument('--checkpoint_dir', type=str, default="logdir", help="Directory to save model")
    parser.add_argument('--resume', type=str, default=None, help="The model name to restore")
    parser.add_argument('--checkpoint_step', type=int, default=2000)
    parser.add_argument('--summary_step', type=int, default=1)
    parser.add_argument('--use_cuda', type=_str_to_bool, default=True)
    parser.add_argument('--g_learning_rate', type=float, default=0.0001)
    parser.add_argument('--d_learning_rate', type=float, default=0.00005)
    parser.add_argument('--warmup_steps', type=int, default=40000)
    parser.add_argument('--decay_learning_rate', type=float, default=0.5)
    parser.add_argument('--local_condition_dim', type=int, default=80)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--condition_window', type=int, default=100)
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--ema_checkpoint_dir', type=str, default="ema_logdir")
    parser.add_argument('--lamda', type=float, default=10)

    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
