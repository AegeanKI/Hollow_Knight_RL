"""Actor-Critic 網路（PPO 用）。共用 CNN，分出策略頭與價值頭。

策略頭沿用 BC 的結構，可直接從 bc.pt 載入權重熱啟動；價值頭從頭學。
動作是 11 個獨立 Bernoulli（每鍵按/不按）。
"""
import torch
import torch.nn as nn
from torch.distributions import Bernoulli

import config


class ActorCritic(nn.Module):
    def __init__(self, in_ch=None, n_actions=None, n_extra=None):
        super().__init__()
        in_ch = in_ch or config.NET_CHANNELS
        n_actions = n_actions or config.N_ACTIONS
        # privileged critic：critic 額外吃 n_extra 個特權特徵（actor 不給，維持純看畫面）。
        self.n_extra = config.N_CRITIC_EXTRA if n_extra is None else n_extra
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.conv(torch.zeros(1, in_ch, config.NET_SIZE, config.NET_SIZE)).shape[1]
        # actor 與 BC 的 head 同結構 -> 可直接載入（actor 只吃畫面特徵 h）
        self.actor = nn.Sequential(
            nn.Linear(n, 512), nn.ReLU(inplace=True), nn.Linear(512, n_actions))
        # critic 第一層多吃 n_extra 欄特權特徵；warm-start 舊 ckpt 時新欄 zero-init（見 train_rl）
        self.critic = nn.Sequential(
            nn.Linear(n + self.n_extra, 512), nn.ReLU(inplace=True), nn.Linear(512, 1))

    def forward(self, x, extra=None):
        """extra: (B, n_extra) critic 特權特徵；None 時只算 actor（eval/部署不需 value）。"""
        h = self.conv(x)
        logits = self.actor(h)
        if extra is None:
            return logits, None
        value = self.critic(torch.cat([h, extra], dim=1)).squeeze(-1)
        return logits, value               # logits (B,A), value (B,)

    def init_from_bc(self, bc_state):
        """把 BC 的 conv + head 權重載進來（head -> actor），critic 維持隨機。"""
        own = self.state_dict()
        for k, v in bc_state.items():
            if k.startswith("conv."):
                own[k] = v
            elif k.startswith("head."):
                own["actor." + k[len("head."):]] = v
        self.load_state_dict(own)

    def load_compat(self, state):
        """載入權重，並相容 privileged-critic 之前的舊 checkpoint。
        舊 ckpt 的 critic 第一層少了 n_extra 欄 → 把舊權重放前面、新增的特權欄 zero-init
        （critic 初始輸出與舊 critic 完全一致，不丟 rl_best 行為），其餘照載。
        回傳 True 表示做了此適配（=架構改過 → 呼叫端應放棄載入舊 optimizer 狀態：critic
        第一層形狀已變，Adam 動量 buffer 對不上，改用全新 Adam）。"""
        own = self.state_dict()
        key = "critic.0.weight"
        adapted = False
        if key in state and key in own and state[key].shape != own[key].shape:
            old = state[key]                 # 舊 [512, n]
            new = own[key].clone()           # 新 [512, n+n_extra]（隨機初值）
            ncols = old.shape[1]
            new[:, :ncols] = old             # 舊欄載入
            new[:, ncols:] = 0.0             # 新增特權欄 zero-init → 初始等同舊 critic
            state = dict(state); state[key] = new
            adapted = True
        own.update(state)
        self.load_state_dict(own)
        return adapted

    @torch.no_grad()
    def act(self, obs, extra=None, deterministic=False):
        """obs (C,H,W) -> (action uint8[A], logp float, value float)。
        extra (n_extra,) 給 critic 算 value；eval 取樣不需 value 時可省略（回傳 value=0）。"""
        e = None if extra is None else extra.unsqueeze(0)
        logits, value = self.forward(obs.unsqueeze(0), e)
        dist = Bernoulli(logits=logits[0])
        a = (torch.sigmoid(logits[0]) > 0.5).float() if deterministic else dist.sample()
        logp = dist.log_prob(a).sum().item()
        return a.to(torch.uint8).cpu().numpy(), logp, (0.0 if value is None else value.item())

    @torch.no_grad()
    def value(self, obs, extra):
        """obs (C,H,W), extra (n_extra,) -> V(s) float。給 truncation bootstrap 用。"""
        _, value = self.forward(obs.unsqueeze(0), extra.unsqueeze(0))
        return value.item()

    def evaluate(self, obs, actions, extra):
        """給一批 (obs, actions, extra) -> (logp, entropy, value)，PPO 更新用。"""
        logits, value = self.forward(obs, extra)
        dist = Bernoulli(logits=logits)
        logp = dist.log_prob(actions).sum(-1)
        ent = dist.entropy().sum(-1)
        return logp, ent, value
