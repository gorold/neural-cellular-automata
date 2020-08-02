import torch
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

from utils import *

def viz_batch(x0, x):
    vis0 = to_rgb(x0).transpose(1,2).transpose(2,3)
    vis1 = to_rgb(x).transpose(1,2).transpose(2,3)
    plt.figure(figsize=[15, 5])
    for i in range(x0.size(0)):
        plt.subplot(2, x0.size(0), i+1)
        plt.imshow(vis0[i])
        plt.axis('off')
    for i in range(x.size(0)):
        plt.subplot(2, x0.size(0), i+1+x0.size(0))
        plt.imshow(vis1[i])
        plt.axis('off')
    plt.show()

def viz_loss(losses):
    plt.figure(figsize=(10, 4))
    plt.title('Loss history (log10)')
    plt.plot(torch.log10(torch.as_tensor(losses)), '.', alpha=0.1)
    plt.show()

def train_step(nca, x, target, steps, optimizer, scheduler):
    nca.train()
    x = nca(x, steps=steps)
    loss = F.mse_loss(to_rgba(x), target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return x, loss

def rank_losses(x, target):
    """
    Returns the indices of losses from descending order (highest to lowest), per sample in the batch.

    Parameters
    ----------
    x: tensor, (batch, channel_n, h, w)
    target: tensor, (4, h, w)
    """
    loss = torch.mean(torch.pow(to_rgba(x) - target, 2), dim=[1, 2, 3])
    return torch.argsort(loss, descending=True)

def pool_train(nca, target, optimizer, scheduler, epochs, device, pool_size, batch_size, damage_n):
    """

    Parameters
    ----------
    target: tensor, (4, h, w)
        Premultiplied
    
    """
    target = pad_target(target) # (4, h+16, w+16)
    h, w = target.size(1), target.size(2)
    seed = make_seed((h, w), pool_size, nca.channel_n).to(device)
    pool = SamplePool(x=seed)
    batch = pool.sample(pool_size).x

    losses = list()

    for epoch in range(1, epochs + 1):
        steps = int(torch.randint(low=64, high=96+1, size=()))
        batch = pool.sample(batch_size)
        x0 = batch.x
        ranked_losses = rank_losses(x0, target)
        x0 = batch.x[ranked_losses]
        x0[:1] = make_seed((h, w), 1, nca.channel_n).to(device) # change first sample to a seed
        
        if damage_n:
            damage = 1 - make_circle_masks(damage_n, h, w).unsqueeze(1)
            x0[-damage_n:] *= damage.to(device) # mask last damage_n tensors in the first dim

        x, loss = train_step(nca, x0, target, steps, optimizer, scheduler)

        batch.replace(x=x.detach())
        batch.commit()

        losses.append(loss.detach().cpu())

        print(f'Loss (epoch {epoch}): {loss.item()}')

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            viz_batch(x0.detach().cpu(), x.detach().cpu())
            viz_loss(losses)
