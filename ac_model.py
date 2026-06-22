"""Actor-Critic 網路（PPO 用）。共用 CNN，分出策略頭與價值頭。

策略頭沿用 BC 的結構，可直接從 bc.pt 載入權重熱啟動；價值頭從頭學。
動作是 11 個獨立 Bernoulli（每鍵按/不按）。
"""
import torch
import torch.nn as nn
from torch.distributions import Bernoulli

import config


class ActorCritic(nn.Module):
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
        # actor 與 BC 的 head 同結構 -> 可直接載入
        self.actor = nn.Sequential(
            nn.Linear(n, 512), nn.ReLU(inplace=True), nn.Linear(512, n_actions))
        self.critic = nn.Sequential(
            nn.Linear(n, 512), nn.ReLU(inplace=True), nn.Linear(512, 1))

    def forward(self, x):
        h = self.conv(x)
        return self.actor(h), self.critic(h).squeeze(-1)   # logits (B,A), value (B,)

    def init_from_bc(self, bc_state):
        """把 BC 的 conv + head 權重載進來（head -> actor），critic 維持隨機。"""
        own = self.state_dict()
        for k, v in bc_state.items():
            if k.startswith("conv."):
                own[k] = v
            elif k.startswith("head."):
                own["actor." + k[len("head."):]] = v
        self.load_state_dict(own)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        """obs (C,H,W) -> (action uint8[A], logp float, value float)。"""
        logits, value = self.forward(obs.unsqueeze(0))
        dist = Bernoulli(logits=logits[0])
        a = (torch.sigmoid(logits[0]) > 0.5).float() if deterministic else dist.sample()
        logp = dist.log_prob(a).sum().item()
        return a.to(torch.uint8).cpu().numpy(), logp, value.item()

    @torch.no_grad()
    def value(self, obs):
        """obs (C,H,W) -> V(s) float。給 truncation bootstrap 用。"""
        _, value = self.forward(obs.unsqueeze(0))
        return value.item()

    def evaluate(self, obs, actions):
        """給一批 (obs, actions) -> (logp, entropy, value)，PPO 更新用。"""
        logits, value = self.forward(obs)
        dist = Bernoulli(logits=logits)
        logp = dist.log_prob(actions).sum(-1)
        ent = dist.entropy().sum(-1)
        return logp, ent, value
