"""策略網路：吃疊幀觀測，輸出 11 個鍵各自的 logit（多標籤二元）。

結構是經典的 Nature-CNN（Atari DQN 同款），對 64x64 輸入夠用又輕，
RTX 3070 上推論延遲遠小於 66ms（15Hz）的預算。
"""
import torch
import torch.nn as nn

import config


class PolicyNet(nn.Module):
    def __init__(self, in_ch=None, n_actions=None):
        super().__init__()
        in_ch = in_ch or config.NET_CHANNELS
        n_actions = n_actions or config.N_ACTIONS
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.conv(torch.zeros(1, in_ch, config.NET_SIZE, config.NET_SIZE)).shape[1]
        self.head = nn.Sequential(
            nn.Linear(n, 512), nn.ReLU(inplace=True),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        return self.head(self.conv(x))   # logits (B, n_actions)

    @torch.no_grad()
    def act(self, obs, threshold=0.5, resolve_opposites=True):
        """obs: (C,H,W) float32 tensor -> MultiBinary 動作向量 (numpy uint8)。"""
        logits = self.forward(obs.unsqueeze(0))
        probs = torch.sigmoid(logits)[0].cpu().numpy()
        vec = (probs > threshold).astype("uint8")
        if resolve_opposites:
            for a, b in (("UP", "DOWN"), ("LEFT", "RIGHT")):
                ia, ib = config.ACTION_NAMES.index(a), config.ACTION_NAMES.index(b)
                if vec[ia] and vec[ib]:                       # 同時按對向鍵 -> 留機率高的
                    drop = ib if probs[ia] >= probs[ib] else ia
                    vec[drop] = 0
        return vec
