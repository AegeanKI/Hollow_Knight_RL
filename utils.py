# import random

import torch
import numpy as np

from collections import deque

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


    def store(self, val):
        self.buffer.append(val)

    def get(self, idx):
        # idx = torch.randint(self.maxlen, (1,))[0]
        return self.buffer[idx]

    @property
    def is_full(self):
        return len(self.buffer) == self.maxlen

    def __len__(self):
        return len(self.buffer)




def unpackbits(x, num_bits):
    if np.issubdtype(x.dtype, np.floating):
        raise ValueError("numpy data type needs to be int-like")
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2**np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits])
    res = (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])
    return torch.from_numpy(res)
