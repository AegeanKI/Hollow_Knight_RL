"""BC 資料集：載入所有 demo episode，提供 (疊幀觀測, 動作) 配對。

- frame stacking 不跨 episode 邊界（episode 開頭以重複首幀補齊）。
- 預先把 96x96 幀降到 NET_SIZE 存記憶體，stacking 時只做索引，省 CPU。
"""
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

import config
from obs import preprocess_frame, stack_to_tensor


class DemoDataset(Dataset):
    def __init__(self, files):
        self.k = config.FRAME_STACK
        self.ep_frames = []   # 每個 episode: (N, C, H, W) uint8（已降到 NET_SIZE）
        self.ep_actions = []  # 每個 episode: (N, 11) uint8
        self.index = []       # 攤平索引: (ep_idx, local_i)
        for f in files:
            d = np.load(f)
            frames, actions = d["frames"], d["actions"]
            proc = np.stack([preprocess_frame(fr) for fr in frames])  # (N,C,H,W)
            ep = len(self.ep_frames)
            self.ep_frames.append(proc)
            self.ep_actions.append(actions.astype(np.float32))
            for i in range(len(frames)):
                self.index.append((ep, i))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ep, i = self.index[idx]
        frames = self.ep_frames[ep]
        lo = i - self.k + 1
        ids = [max(0, j) for j in range(lo, i + 1)]   # 不足以首幀補齊
        stack = np.concatenate([frames[j] for j in ids], axis=0)  # (k*C,H,W)
        obs = torch.from_numpy(stack_to_tensor(stack))
        act = torch.from_numpy(self.ep_actions[ep][i])
        return obs, act

    def action_positive_rate(self):
        """每個鍵的正樣本比例（給 BCE pos_weight 用）。"""
        alla = np.concatenate(self.ep_actions, axis=0)
        return alla.mean(axis=0)


def split_files(val_frac=0.12, seed=0):
    files = sorted(glob.glob(os.path.join(config.DATA_DIR, "*.npz")))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(files))
    n_val = max(1, int(len(files) * val_frac))
    val = [files[i] for i in perm[:n_val]]
    train = [files[i] for i in perm[n_val:]]
    return train, val
