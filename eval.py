import argparse
import os, time, json

import torch

from NeuralCellularAutomata import *
from utils import *
from trainer import *
from dataloaders import *

import matplotlib.pyplot as plt

@torch.no_grad()
def eval(model, x, target, steps, encoding=None):
    """
    Evaluates the accuracy of the resulting image after a `steps` steps
    """
    model.eval()

    if isinstance(model, GrowingNCA):
        x = model(x, steps=steps)

    elif isinstance(model, ConditionalNCA):
        x = model(x, target)

    loss = F.mse_loss(to_rgba(x), target)

    return loss

@torch.no_grad()
def stability_eval(model, x, steps, eval_steps, encoding=None):
    """
    Evaluates the stability of the model over `eval_steps` steps after `steps` steps
    """ 
    if isinstance(model, GrowingNCA):
            x_new = model(x, steps=steps)
    if isinstance(model, ConditionalNCA):
        x_new = model(x, encoding=encoding, steps=steps)
    diff = list()
    
    for step in range(eval_steps):
        if isinstance(model, GrowingNCA):
            x_new = model(x,)
        if isinstance(model, ConditionalNCA):
            x_new = model(x, encoding=encoding)
        loss = F.mse_loss(to_rgba(x_new), to_rgba(x))
        diff.append(float(loss.detach().cpu()))
        
    return sum(diff)/len(diff), diff  
