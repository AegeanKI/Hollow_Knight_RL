import os
import torch
import numpy as np
from collections import deque

from pc import FileAdmin, Logger

class Counter():
    def __init__(self, start, end, fn):
        self.start = start
        self.end = end
        self.fn = fn

        self.val = start
        self.step_count = 0
        self.bound = min if end > start else max

        self.reset()

    def step(self):
        self.val = self.fn(self.val)
        self.val = self.bound(self.val, self.end)
        self.step_count = self.step_count + 1

    def reset(self):
        self.val = self.start
        self.step_count = 0

    def final(self):
        self.val = self.end

class Memory():
    def __init__(self, maxlen, size_list, memory_dir):
        self.maxlen = maxlen

        self.buffers = [torch.zeros((maxlen, *size)) for size in size_list]
        self.count = 0
        self.memory_dir = memory_dir
        if not os.path.exists(memory_dir):
            os.makedirs(self.memory_dir)

    @property
    def full(self):
        return self.count == self.maxlen

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
        assert len(vals) == len(self.buffers), f"memory append error, {len(val) = }, {len(self.buffers) = }"

        n = vals[0].shape[0]
        for i in range(len(self.buffers)):
            self.buffers[i] = Memory.shift(self.buffers[i], -n, vals[i].cpu())
        self.count = min(self.maxlen, self.count + n)

    def __getitem__(self, idx):
        return [buf[idx] for buf in self.buffers]

    def __len__(self):
        return self.count

    def save(self, name):
        print(f"saving memory", end='\r')
        for i in range(len(self.buffers)):
            FileAdmin.safe_save(self.buffers[i], f"{self.memory_dir}/{name}_{i}", quiet=True)
        with open(f"{self.memory_dir}/{name}_count", 'w') as f:
            f.write(f"{self.count}")
        print(f"saving memory completed")

    def load(self, name):
        print(f"loading memory", end='\r')
        for i in range(len(self.buffers)):
            self.buffers[i] = FileAdmin.safe_load(self.buffers[i], f"{self.memory_dir}/{name}_{i}", quiet=True)
        with open(f"{self.memory_dir}/{name}_count", 'r') as f:
            self.count = int(f.readline())
        print(f"loading memory completed")

    def random_sample(self, n_frames, times):
        return prioritize_sample(n_frames, times, full_random=True)

    def prioritize_sample(self, n_frames, times, full_random=False):
        assert torch.sum(self.buffers[4] > 1) == 0, f"buffers[4] greater than 1, self.buffers[self.buffers[4] > 1]"
        can_be_last = torch.zeros_like(self.buffers[4])
        for i in range(1, n_frames + 1):
            can_be_last = torch.logical_or(can_be_last, torch.roll(torch.clone(self.buffers[4]), i))
            assert torch.sum(can_be_last > 1) == 0, f"can_be_last has greater than 1, {can_be_last[can_be_last > 1] = }"
        can_be_last = torch.logical_not(can_be_last)
        assert torch.sum(can_be_last < 0) == 0, f"can_be_last negative, {can_be_last[can_be_last < 0] = }"
        if full_random:
            weights = torch.ones_like(self.buffers[5]).float()
        else:
            weights = torch.clone(self.buffers[5]).float().view(-1).abs() + 1 / self.maxlen
        weights = weights * can_be_last.view(-1)

        idxs = torch.multinomial(weights[n_frames - 1:], times) + n_frames - 1
        for idx in idxs:
            assert torch.clone(self.buffers[4])[idx + 1 - n_frames:idx + 1 - 1].sum() == 0, f"contain done, {torch.clone(self.buffers[4])[idx + 1 - n_frames:idx + 1 - 1].sum() = }, {idx = }"
            yield self[idx + 1 - n_frames:idx + 1]

        # count_done = 1
        # idx = None
        # while count_done:
        #     #  (state, condition, action_idx, reward, done, affect, next_state, next_condition)
        #     candidates = candidates + n_frames - 1 # idx is last
        #     idx = candidates[torch.randint(len(candidates), (1,))][0]
        #     count_done = self.buffers[4][idx + 1 - n_frames:idx + 1 - 1].sum() # last frame can be done

        # return self[idx + 1 - n_frames:idx + 1]

def unpackbits(x, num_bits):
    if np.issubdtype(x.dtype, np.floating):
        raise ValueError("numpy data type needs to be int-like")
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2 ** np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits])
    res = (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])

    del xshape, x, mask
    return torch.from_numpy(res)
