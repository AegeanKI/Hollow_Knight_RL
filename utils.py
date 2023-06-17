# import random
import os
import torch
import numpy as np
from collections import deque
from collections.abc import Iterable

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
    SAFE_LEN = 2000

    def __init__(self, save_dir, maxlen):
        self.save_dir = save_dir
        self.whole_maxlen = maxlen
        self.memory_num = int((maxlen + Memory.SAFE_LEN - 1) // Memory.SAFE_LEN)
        self.each_maxlen = [Memory.SAFE_LEN] * (self.memory_num - 1) + [maxlen - Memory.SAFE_LEN * (self.memory_num - 1)]
        self.load()

    def _save_menu(self, val, prefix):
        menu_file = f"{self.save_dir}/{prefix}_menu"
        with open(menu_file, 'w') as f:
            f.write(str(val))

    def _read_menu(self, prefix):
        menu_file = f"{self.save_dir}/{prefix}_menu"
        if not os.path.exists(menu_file):
            self._save_menu(0, prefix=prefix)

        with open(menu_file, 'r') as f:
            last_memory_idx = int(f.readline())
        assert last_memory_idx < self.memory_num, "menu file is not compatible"
        return last_memory_idx

    def _save_memory(self, cur_memory, cur_memory_idx, prefix, indent=0, quiet=False):
        cur_memory_name = f"{prefix}_memory_{cur_memory_idx}"
        FileAdmin.safe_save(cur_memory, f"{self.save_dir}/{cur_memory_name}", indent=indent, quiet=quiet)

    # def _remove_memory(self, cur_memory_idx, prefix="default"):
    #     cur_memory_name = f"{prefix}_memory_{cur_memory_idx}"
    #     FileAdmin.safe_remove(f"{self.save_dir}/{cur_memory_name}")

    def _load_memory(self, cur_memory_idx, prefix, indent=0, quiet=False):
        cur_memory_name = f"{prefix}_memory_{cur_memory_idx}"
        cur_memory = deque(maxlen=self.each_maxlen[cur_memory_idx])
        cur_memory = FileAdmin.safe_load(cur_memory, f"{self.save_dir}/{cur_memory_name}", indent=indent, quiet=quiet)
        assert isinstance(cur_memory, deque), f"memory file {cur_memory_idx} is not correct format"
        assert cur_memory.maxlen == self.each_maxlen[cur_memory_idx], f"memory file {cur_memory_idx} is not compatible"
        return cur_memory

    def load(self, new_prefix="default", indent=0):
        self.prefix = new_prefix
        # print(f"loading {new_prefix} memory")

        self.last_memory_idx = self._read_menu(prefix=new_prefix)

        self.last_memory = self._load_memory(self.last_memory_idx, prefix=new_prefix, indent=indent + 4)

        for idx in range(self.last_memory_idx):
            assert os.path.exists(f"{self.save_dir}/{new_prefix}_memory_{idx}"), "lack of memory files"

        # print(f"loading {new_prefix} memory completed")

    def append(self, val, indent=0):
        print(f"appending {self.prefix} memory")
        if len(self.last_memory) != self.each_maxlen[self.last_memory_idx] or self.memory_num == 1:
            self.last_memory.append(val)
            if len(self.last_memory) == self.each_maxlen[self.last_memory_idx]:
                self._save_memory(self.last_memory, self.last_memory_idx, prefix=self.prefix, indent=indent + 4)
            return


        if self.last_memory_idx != self.memory_num - 1:
            self._save_memory(self.last_memory, self.last_memory_idx, prefix=self.prefix, indent=indent + 4)

            self._save_menu(self.last_memory_idx + 1, prefix=self.prefix)
            self.last_memory_idx = self.last_memory_idx + 1

            self.last_memory = self._load_memory(self.last_memory_idx, prefix=self.prefix, indent=indent + 4)
            self.append(val)
            return

        for cur_memory_idx in range(self.last_memory_idx, -1, -1):
            cur_memory = self._load_memory(cur_memory_idx, prefix=self.prefix, indent=indent + 4)

            assert isinstance(cur_memory, deque), f"memory file {cur_memory_idx} is not correct format"
            assert cur_memory.maxlen == self.each_maxlen[cur_memory_idx], f"memory file {cur_memory_idx} is not compatible"

            next_val = cur_memory[0]
            cur_memory.append(val)
            self._save_memory(cur_memory, cur_memory_idx, prefix=self.prefix, indent=indent + 4)

            val = next_val

            # del cur_memory
        print(f"appending {self.prefix} memory completed")

    def __len__(self):
        return Memory.SAFE_LEN * self.last_memory_idx + len(self.last_memory)

    def extend(self, iterable, indent=0):
        print(f"extending {self.prefix} memory")
        assert isinstance(iterable, Iterable)
        if len(iterable) == 0:
            return

        if len(self.last_memory) != self.each_maxlen[self.last_memory_idx]:
            for idx, val in enumerate(iterable):
                self.last_memory.append(val)
                if len(self.last_memory) == self.each_maxlen[self.last_memory_idx]:
                    self._save_memory(self.last_memory, self.last_memory_idx, prefix=self.prefix, indent=indent + 4)
                    self.extend(iterable[idx + 1:], indent)
                    break
            return

        if self.memory_num == 1:
            self.last_memory.extend(iterable)
            if len(self.last_memory) == self.each_maxlen[self.last_memory_idx]:
                self._save_memory(self.last_memory, self.last_memory_idx, prefix=self.prefix, indent=indent + 4)
            return

        if self.last_memory_idx != self.memory_num - 1:
            self._save_memory(self.last_memory, self.last_memory_idx, prefix=self.prefix, indent=indent + 4)

            self._save_menu(self.last_memory_idx + 1, prefix=self.prefix)
            self.last_memory_idx = self.last_memory_idx + 1

            self.last_memory = self._load_memory(self.last_memory_idx, prefix=self.prefix, indent=indent + 4)
            self.extend(iterable)
            return

        for cur_memory_idx in range(self.last_memory_idx, -1, -1):
            cur_memory = self._load_memory(cur_memory_idx, prefix=self.prefix, indent=indent + 4)
            assert isinstance(cur_memory, deque), f"memory file {cur_memory_idx} is not correct format"
            assert cur_memory.maxlen == self.each_maxlen[cur_memory_idx], f"memory file {cur_memory_idx} is not compatible"

            next_iterable = deque([], maxlen=len(iterable))
            for val in iterable:
                if len(cur_memory) > 0:
                    next_iterable.append(cur_memory[0])
                cur_memory.append(val)

            self._save_memory(cur_memory, cur_memory_idx, prefix=self.prefix, indent=indent + 4)

            iterable = next_iterable
        print(f"extending {self.prefix} memory completed")

    def save(self, new_prefix="default", indent=0):
        # print(f"saving {new_prefix} memory")
        if self.prefix == new_prefix:
            return

        self._save_memory(self.last_memory, self.last_memory_idx, prefix=self.prefix, indent=indent + 4)

        for cur_memory_idx in range(self.last_memory_idx, -1, -1):
            cur_memory = self._load_memory(cur_memory_idx, prefix=self.prefix, indent=indent + 4)
            self._save_memory(cur_memory, cur_memory_idx, prefix=new_prefix, indent=indent + 4)

        self._save_menu(self.last_memory_idx, prefix=new_prefix)
        self.prefix = new_prefix
        # print(f"saving {new_prefix} memory completed")

    def batch_getitem(self, keys, indent=0):
        self._save_memory(self.last_memory, self.last_memory_idx, prefix=self.prefix, indent=indent, quiet=True)

        sorted_keys, index = keys.sort()
        cur_memory_idx = -1
        cur_memory = None
        sorted_res = []
        for idx, key in enumerate(sorted_keys):
            print(f"sampling {idx + 1} / {len(sorted_keys)}", end='\r')
            if int(key / Memory.SAFE_LEN) != cur_memory_idx:
                cur_memory_idx = int(key / Memory.SAFE_LEN)
                cur_memory = self._load_memory(cur_memory_idx, prefix=self.prefix, indent=indent, quiet=True)

            sorted_res.append(cur_memory[key % self.each_maxlen[cur_memory_idx]])
            # try:
            #     sorted_res.append(cur_memory[key % self.each_maxlen[cur_memory_idx]])
            # except:
            #     print(f"{key = }, {self.each_maxlen = }, {cur_memory_idx = }")
            #     sorted_res.append(cur_memory[key % self.each_maxlen[cur_memory_idx]])
        res = deque()
        for reverse_index in index.argsort():
            res.append(sorted_res[reverse_index])

        del sorted_keys, index
        del cur_memory, cur_memory_idx
        del sorted_res
        return res

    def __getitem__(self, key, indent=0):
        key_memory_idx = int(key / Memory.SAFE_LEN)
        key_memory = self._load_memory(key_memory_idx, prefix=self.prefix, indent=indent, quiet=True)
        return key_memory[key % Memory.SAFE_LEN]

# class Memory(deque):
#     # SAFE_LEN = 10000
#     SAFE_LEN = 2500

#     def __init__(self, maxlen):
#         self.whole_maxlen = maxlen
#         self.split = (maxlen > Memory.SAFE_LEN)
#         if self.split:
#             self.buffer0 = deque(maxlen=Memory.SAFE_LEN)
#             self.buffer1 = Memory(maxlen=(maxlen - Memory.SAFE_LEN))
#         else:
#             self.buffer0 = deque(maxlen=maxlen)

#     def append(self, val):
#         if self.split and len(self.buffer1) == self.whole_maxlen - Memory.SAFE_LEN:
#             self.buffer0.append(self.buffer1[0])
#             self.buffer1.append(val)
#         elif self.split and len(self.buffer0) < Memory.SAFE_LEN:
#             self.buffer0.append(val)
#         elif self.split and len(self.buffer1) < self.whole_maxlen - Memory.SAFE_LEN:
#             self.buffer1.append(val)
#         else:
#             self.buffer0.append(val)

#     def __getitem__(self, key):
#         if key < Memory.SAFE_LEN:
#             return self.buffer0[key]
#         return self.buffer1[key - Memory.SAFE_LEN]

#     def __len__(self):
#         if self.split:
#             return len(self.buffer0) + len(self.buffer1)
#         return len(self.buffer0)

#     def save(self, name, split_id=0):
#         tmp_pkl = f"{name}_{split_id}.tmp.pkl"
#         target_pkl = f"{name}_{split_id}.pkl"

#         Logger.indent(indent=4)
#         print(f"saving {target_pkl}")

#         # FileAdmin.safe_remove(tmp_pkl, indent=8)
#         # FileAdmin.safe_remove(target_pkl, indent=8)
#         # FileAdmin.safe_save(self.buffer0, tmp_pkl, indent=8)
#         # FileAdmin.safe_copy(tmp_pkl, target_pkl, indent=8)
#         # FileAdmin.safe_remove(tmp_pkl, indent=8)

#         FileAdmin.safe_remove(target_pkl, indent=8)
#         FileAdmin.safe_save(self.buffer0, target_pkl, indent=8)

#         Logger.indent(indent=4)
#         print(f"saving {target_pkl} completed")

#         if self.split:
#             self.buffer1.save(name, split_id + 1)

#     def load(self, name, split_id=0):
#         target_pkl = f"{name}_{split_id}.pkl"

#         self.buffer0 = FileAdmin.safe_load(self.buffer0, target_pkl, indent=4)

#         if self.split:
#             self.buffer1.load(name, split_id + 1)


def unpackbits(x, num_bits):
    if np.issubdtype(x.dtype, np.floating):
        raise ValueError("numpy data type needs to be int-like")
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2**np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits])
    res = (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])

    del xshape, x, mask
    return torch.from_numpy(res)
