import argparse
import os, time, json

import torch
import torch.optim as optim

from NeuralCellularAutomata import *
from utils import *
from trainer import *
from dataloaders import *

def get_options():
    parser = argparse.ArgumentParser()

    # General options
    parser.add_argument('--emoji', required=False, type=int, help='Select which emoji to train on (0-9).')
    parser.add_argument('--model_dir', default='models/', help='Where to save the trained model to?')
    parser.add_argument('--fig_dir', default='figures/', help='Where to save any figure/images to?')
    parser.add_argument('--save_epoch', default=100, type=int, help='Save figures and model every save_epoch epochs.')
    parser.add_argument('--conditional', action='store_true', help='Train regular GrowingNCA or ConditionalNCA.')
    parser.add_argument('--train_dir', default='data/train_vae', help='Directory to images to train on, for ConditionalNCA')
    parser.add_argument('--model_name', default='nca', help='Name your model!')

    # NCA model options
    parser.add_argument('--channel_n', default=16, type=int, help='Number of channels to represent cell state.')
    parser.add_argument('--fire_rate', default=0.5, type=float, help='Probability to update a particular cell in the stochastic cell update.')
    parser.add_argument('--hidden_size', default=128, type=int, help='Number of output feature maps in NCA\'s update rules conv layer.')

    # Training procedure options
    parser.add_argument('--steps_low', default=64, type=int, help='Number of steps (floor) to take per training step')
    parser.add_argument('--steps_high', default=96, type=int, help='Number of steps (floor) to take per training step')
    parser.add_argument('--pool_size', default=1024, type=int, help='Number of samples for the seed SamplePool with.')
    parser.add_argument('--batch_size', default=8, type=int, help='Number of samples to train at a time.')
    parser.add_argument('--damage_n', default=3, type=int, help='Number of damaged samples per batch.')
    parser.add_argument('--epochs', default=8000, type=int, help='Number of epochs to train.')
    parser.add_argument('--cuda', action='store_true', help='Enables cuda.')
    parser.add_argument('--vae', action='store_true', help='Enables VAE.')

    # Optimizer options
    parser.add_argument('--lr', default=0.001, type=float, help='Adam optimizer learning rate.')
    parser.add_argument('--beta1', default=0.5, type=float, help='Adam optimizer beta1.')
    parser.add_argument('--beta2', default=0.5, type=float, help='Adam optimizer beta2.')
    parser.add_argument('--gamma', default=0.9999, type=float, help='Exponential LR scheduler gamma discount factor.')

    # VAE options
    parser.add_argument('--latent_dims', default=5, type=int, help='Dimension of latent vector in VAE')

    opt = parser.parse_args()

    # General options asserts
    assert bool(opt.emoji) != bool(opt.conditional) # either specify emoji to train on, or conditional

    # NCA model options asserts
    assert opt.channel_n > 4
    assert opt.fire_rate > 0 and opt.fire_rate <= 1
    assert opt.hidden_size > 0

    # Training procedure options asserts
    assert opt.steps_low > 0
    assert opt.steps_low <= opt.steps_high
    assert opt.pool_size > 0
    assert opt.batch_size > 0 and opt.batch_size <= opt.pool_size
    assert opt.damage_n > 0 and opt.damage_n < opt.batch_size
    assert opt.epochs > 0

    # Optimizer options asserts
    assert opt.lr > 0
    assert opt.beta1 > 0
    assert opt.beta2 > 0
    assert opt.gamma > 0

    return opt

def get_model_name(model_dir, name):
    i = 0
    model_name = f'{name}_{i}'
    while os.path.exists(os.path.join(model_dir, model_name + '.pth')):
        i += 1
        model_name = f'{name}_{i}'
    return model_name

def run():
    opt = get_options()

    model_name = get_model_name(opt.model_dir, opt.model_name)
    model_path = os.path.join(opt.model_dir, model_name + '.pth')
    fig_dir = os.path.join(opt.fig_dir, model_name)

    # Create model_dir and save settings into json file
    mkdir_p(os.path.join(opt.model_dir))
    with open(os.path.join(opt.model_dir, model_name + '.json'), 'w') as fp:
        json.dump(opt.__dict__, fp)        

    device = torch.device('cuda' if opt.cuda else 'cpu')

    if opt.conditional:
        target = load_emoji_dict(opt.train_dir)
        nca = ConditionalNCA(device, len(target.keys()), channel_n=opt.channel_n, fire_rate=0.5, hidden_size=opt.hidden_size, enable_vae=opt.vae, latent_dims=opt.latent_dims)
        train_func = conditional_pool_train
    else:
        target = load_emoji(opt.emoji)
        nca = GrowingNCA(device, channel_n=opt.channel_n, fire_rate=opt.fire_rate, hidden_size=opt.hidden_size)
        train_func = pool_train

    optimizer = optim.Adam(nca.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, opt.gamma)

    args = [
        nca,
        target,
        optimizer,
        scheduler,
        opt.epochs,
        device,
        opt.steps_low,
        opt.steps_high,
        opt.pool_size,
        opt.batch_size,
        opt.damage_n,
        fig_dir,
        model_path,
        opt.vae
    ]

    kwargs = {
        'save_epoch': opt.save_epoch
    }

    train_func(*args, **kwargs)

if __name__ == '__main__':
    run()
