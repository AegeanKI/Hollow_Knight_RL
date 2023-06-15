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

class Memory(deque):
    # SAFE_LEN = 10000
    SAFE_LEN = 2500

    def __init__(self, maxlen):
        self.whole_maxlen = maxlen
        self.split = (maxlen > Memory.SAFE_LEN)
        if self.split:
            self.buffer0 = deque(maxlen=Memory.SAFE_LEN)
            self.buffer1 = Memory(maxlen=(maxlen - Memory.SAFE_LEN))
        else:
            self.buffer0 = deque(maxlen=maxlen)

    def append(self, val):
        if self.split and len(self.buffer1) == self.whole_maxlen - Memory.SAFE_LEN:
            self.buffer0.append(self.buffer1[0])
            self.buffer1.append(val)
        elif self.split and len(self.buffer0) < Memory.SAFE_LEN:
            self.buffer0.append(val)
        elif self.split and len(self.buffer1) < self.whole_maxlen - Memory.SAFE_LEN:
            self.buffer1.append(val)
        else:
            self.buffer0.append(val)

    def __getitem__(self, key):
        if key < Memory.SAFE_LEN:
            return self.buffer0[key]
        return self.buffer1[key - Memory.SAFE_LEN]

    def __len__(self):
        if self.split:
            return len(self.buffer0) + len(self.buffer1)
        return len(self.buffer0)

    def save(self, name, split_id=0):
        tmp_pkl = f"{name}_{split_id}.tmp.pkl"
        target_pkl = f"{name}_{split_id}.pkl"

        Logger.indent(indent=4)
        print(f"saving {target_pkl}")

        # FileAdmin.safe_remove(tmp_pkl, indent=8)
        # FileAdmin.safe_remove(target_pkl, indent=8)
        # FileAdmin.safe_save(self.buffer0, tmp_pkl, indent=8)
        # FileAdmin.safe_copy(tmp_pkl, target_pkl, indent=8)
        # FileAdmin.safe_remove(tmp_pkl, indent=8)

        FileAdmin.safe_remove(target_pkl, indent=8)
        FileAdmin.safe_save(self.buffer0, target_pkl, indent=8)

        Logger.indent(indent=4)
        print(f"saving {target_pkl} completed")

        if self.split:
            self.buffer1.save(name, split_id + 1)

    def load(self, name, split_id=0):
        target_pkl = f"{name}_{split_id}.pkl"

        self.buffer0 = FileAdmin.safe_load(self.buffer0, target_pkl, indent=4)

        if self.split:
            self.buffer1.load(name, split_id + 1)


def unpackbits(x, num_bits):
    if np.issubdtype(x.dtype, np.floating):
        raise ValueError("numpy data type needs to be int-like")
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2**np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits])
    res = (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])

    del xshape, x, mask
    return torch.from_numpy(res)
