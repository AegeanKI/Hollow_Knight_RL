# import random
import torch
import numpy as np
from collections import deque

from pc import FileAdmin, Logger

class Counter():
    def __init__(self, init, increase, high, low):
        self.init = init
        self.val = init
        self.increase = increase
        self.high = high
        self.low = low

        self.reset()

    def step(self):
        self.val = self.val + self.increase
        self.val = min(self.val, self.high)
        self.val = max(self.val, self.low)

    def reset(self):
        self.val = self.init

class Memory():
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.buffer = deque(maxlen=maxlen)

    def append(self, val):
        self.buffer.append(val)

    def extend(self, iterable):
        self.buffer.extend(iterable)

    def __getitem__(self, idx):
        return self.buffer[idx]

    def __len__(self):
        return len(self.buffer)

    def save(self, name):
        FileAdmin.safe_save(self.buffer, name, quiet=True)

    def load(self, name):
        self.buffer = FileAdmin.safe_load(self.buffer, name, quiet=True)

def unpackbits(x, num_bits):
    if np.issubdtype(x.dtype, np.floating):
        raise ValueError("numpy data type needs to be int-like")
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2**np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits])
    res = (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])

    del xshape, x, mask
    return torch.from_numpy(res)
