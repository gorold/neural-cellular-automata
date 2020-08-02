import argparse
import os

import torch
import torch.optim as optim

from NeuralCellularAutomata import *
from utils import *
from trainer import *

def get_options():
    parser = argparse.ArgumentParser()

    # General options
    parser.add_argument('--emoji', required=True, type=int, help='Select which emoji to train on (0-9).')
    parser.add_argument('--model_dir', default='models/', help='Where to save the trained model to?')
    parser.add_argument('--figure_dir', default='figures/', help='Where to save any figure/images to?')

    # NCA model options
    parser.add_argument('--channel_n', default=16, type=int, help='Number of channels to represent cell state.')
    parser.add_argument('--fire_rate', default=0.5, type=float, help='Probability to update a particular cell in the stochastic cell update.')
    parser.add_argument('--hidden_size', default=128, type=int, help='Number of output feature maps in NCA\'s update rules conv layer.')

    # Training procedure options
    parser.add_argument('--pool_size', default=1024, type=int, help='Number of samples for the seed SamplePool with.')
    parser.add_argument('--batch_size', default=8, type=int, help='Number of samples to train at a time.')
    parser.add_argument('--damage_n', default=3, type=int, help='Number of damaged samples per batch.')
    parser.add_argument('--epochs', default=8000, type=int, help='Number of epochs to train.')
    parser.add_argument('--cuda', action='store_true', help='Enables cuda.')

    # Optimizer options
    parser.add_argument('--lr', default=0.001, type=float, help='Adam optimizer learning rate.')
    parser.add_argument('--beta1', default=0.5, type=float, help='')
    parser.add_argument('--beta2', default=0.5, type=float, help='')
    parser.add_argument('--gamma', default=0.9999, type=float, help='')

    opt = parser.parse_args()

    # NCA model options asserts
    assert opt.channel_n > 4
    assert opt.fire_rate > 0 and opt.fire_rate <= 1
    assert opt.hidden_size > 0

    # Training procedure options asserts
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

def run():
    opt = get_options()

    target = load_emoji(opt.emoji)

    # model_dir = os.path.join(opt.model_dir, )
    # figure_dir = os.path.join(opt.figure_dir, )

    device = torch.device('cuda' if opt.cuda else 'cpu')

    nca = NCA(device, channel_n=opt.channel_n, fire_rate=opt.fire_rate, hidden_size=opt.hidden_size)
    optimizer = optim.Adam(nca.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, opt.gamma)

    pool_train(
        nca, 
        target, 
        optimizer, 
        scheduler, 
        opt.epochs, 
        device, 
        opt.pool_size, 
        opt.batch_size, 
        opt.damage_n,
    )

if __name__ == '__main__':
    run()
