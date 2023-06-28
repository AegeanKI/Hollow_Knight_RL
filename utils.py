import os
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
    def __init__(self, maxlen, size_list):
        self.maxlen = maxlen

        self.buffers = [torch.zeros((maxlen, *size)) for size in size_list]
        self.count = 0

    @staticmethod
    def shift(arr, num, val):
        result = torch.zeros_like(arr)
        if num > 0:
            result[:num] = val
            result[num:] = arr[:-num]
        elif num < 0:
            result[num:] = val
            result[:num] = arr[-num:]
        else:
            result[:] = arr
        return result

    def extend(self, vals):
        assert len(vals) == len(self.buffers), f"memory append error, {len(self.val) = }, {len(self.buffers) = }"

        n = vals[0].shape[0]
        for i in range(len(self.buffers)):
            self.buffers[i] = Memory.shift(self.buffers[i], -n, vals[i].cpu())
        self.count = min(self.maxlen, self.count + n)

    def __getitem__(self, idx):
        return [buf[idx] for buf in self.buffers]

    def __len__(self):
        return self.count

    def save(self, name):
        for i in range(len(self.buffers)):
            FileAdmin.safe_save(self.buffers[i], f"{name}_{i}", quiet=True)
        with open(f"{name}_count", 'w') as f:
            f.write(f"{self.count}")

    def load(self, name):
        for i in range(len(self.buffers)):
            self.buffers[i] = FileAdmin.safe_load(self.buffers[i], f"{name}_{i}", quiet=True)
        with open(f"{name}_count", 'r') as f:
            self.count = int(f.readline())

    def random_sample(self, n_frames):
        count_done = 1
        idx = None
        while count_done:
            # (state, condition, action_idx, reward, done, next_state, next_condition)
            idx = torch.randint(self.count - n_frames + 1, (1,))[0] # idx is first
            count_done = self.buffers[4][idx:idx + n_frames - 1].sum() # last frame can be done

        return self[idx:idx + n_frames]

    def prioritize_random_sample(self, n_frames):
        count_done = 1
        idx = None
        while count_done:
            #  (state, condition, action_idx, reward, done, next_state, next_condition)
            weights = self.buffers[3].view(-1).abs() + 1 / self.maxlen
            idx = torch.multinomial(weights[n_frames - 1:], 1)[0] + n_frames - 1 # idx is last
            count_done = self.buffers[4][idx + 1 - n_frames:idx + 1 - 1].sum() # last frame can be done

        return self[idx + 1 - n_frames:idx + 1]

def unpackbits(x, num_bits):
    if np.issubdtype(x.dtype, np.floating):
        raise ValueError("numpy data type needs to be int-like")
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2 ** np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits])
    res = (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])

    del xshape, x, mask
    return torch.from_numpy(res)
