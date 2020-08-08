import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from utils import *

class SamplePool:
    """
    Implements a DataLoader for the 'sample pool' training strategy.
    Creates a pool of samples to sample from.
    The keyword argument **slots make up the attributes which will be sampled.
    """

    def __init__(self, *, _parent=None, _parent_idx=None, **slots):
        self._parent = _parent
        self._parent_idx = _parent_idx
        self._slot_names = slots.keys()
        self._size = None
        for k, v in slots.items():
            if self._size is None:
                self._size = v.size(0)
            assert self._size == v.size(0)
            setattr(self, k, torch.as_tensor(v))

    def sample(self, n):
        """
        For each attribute (from self._slot_names), sample n and create a new SamplePool object to hold them.
        Attributes must be indexable.
        """
        idx = np.random.choice(self._size, n, False)
        batch = {k: getattr(self, k)[idx] for k in self._slot_names}
        batch = SamplePool(**batch, _parent=self, _parent_idx=idx)
        return batch

    def replace(self, **new_slots):
        """
        Replace current samples with new values.
        """
        for k, v in new_slots.items():
            assert k in self._slot_names
            getattr(self, k)[:] = v

    def commit(self):
        """
        Commit the parent's indexed attribute with the child's attribute.
        """
        for k in self._slot_names:
            getattr(self._parent, k)[self._parent_idx] = getattr(self, k)

class ConditionalSamplePool:
    """
    Updates the SamplePool class in the following ways:
        1. Adds a singleton, t_container, to hold the dict of images (tensors)
    """

    t_container = None

    def __init__(self, *, targets=None, _parent=None, _parent_idx=None, **slots):
        """
        targets: dict[str->tensor]
            Maps a target class name to the tensor representing the target image of shape (4, h, w).
        **slots: kwargs
            Samples for each target class, arguments should be tensors of shape (pool_size, channel_n, h, w).
        """
        self._parent = _parent
        self._parent_idx = _parent_idx
        self._slot_names = slots.keys()
        self._size = None
        for k, v in slots.items():
            if self._size is None:
                self._size = v.size(0)
            assert self._size == v.size(0)
            setattr(self, k, torch.as_tensor(v))
        if ConditionalSamplePool.t_container is None:
            assert targets is not None
            assert all([class_name in targets.keys() for class_name in self._slot_names])
            ConditionalSamplePool.t_container = targets

    def sample(self, n):
        """
        For each attribute (from self._slot_names), sample n and create a new SamplePool object to hold them.
        Attributes must be indexable.
        """
        idx = np.random.choice(self._size, n, False)
        batch = {k: getattr(self, k)[idx] for k in self._slot_names}
        batch = ConditionalSamplePool(**batch, _parent=self, _parent_idx=idx)
        return batch

    def replace(self, x):
        """
        Replace current samples with new values.
        """
        for idx, k in enumerate(self._slot_names):
            start = self._size * idx
            end = self._size * (idx + 1)
            setattr(self, k, x[start:end, ...])

    def commit(self):
        """
        Commit the parent's indexed attribute with the child's attribute.
        """
        for k in self._slot_names:
            getattr(self._parent, k)[self._parent_idx] = getattr(self, k)

    @property
    def x_tensor(self):
        """
        Returns the data of a sample in full tensor form of shape (num_emojis*batch_size, channel_n, h, w)
        """
        return torch.cat([getattr(self, k) for k in self._slot_names], dim=0)

    @property
    def targets_tensor(self):
        """
        Returns the targets of a sample in full tensor form of shape (num_emojis*batch_size, 4, h, w)
        """
        if self._parent is None:
            raise TypeError("Should not obtain Parent SamplePool's targets.")
        return torch.cat([ConditionalSamplePool.t_container[k].unsqueeze(0).repeat(self._size, 1, 1, 1) for k in self._slot_names], dim=0)
