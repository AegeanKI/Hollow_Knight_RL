import numpy as np

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

def unpackbits(x, num_bits):
    if np.issubdtype(x.dtype, np.floating):
        raise ValueError("numpy data type needs to be int-like")
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2**np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits])
    return (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])
