"""DreamerV3 模型元件（world model + 想像用 actor-critic），與訓練迴圈解耦。

只看畫面：encoder 吃 env 的疊幀觀測 (NET_CHANNELS×NET_SIZE×NET_SIZE)，RSSM 在潛在空間
學動態；reward/continue/decoder 頭 + 在「想像」中訓練的 actor/critic。動作沿用 11 鍵
獨立 Bernoulli（MultiBinary）。

忠實度（DreamerV3 要點都有）：symlog two-hot 的 reward/value、categorical 潛在 +
straight-through、KL balancing + free bits、EMA slow critic、percentile return 正規化。
**屬第一版骨架，形狀已離線煙霧測試，但需實機長跑驗證與調參。**

對應檔：`ac_model.py`(PPO 的網路) 之於本檔；`ppo.py` 之於 `dreamer.py`；
`train_rl.py` 之於 `train_dreamer.py`。
"""
import dataclasses

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


@dataclasses.dataclass
class DreamerConfig:
    deter: int = 512            # GRU 決定性狀態維度
    stoch: int = 32             # 隨機潛在的 categorical 數
    classes: int = 32           # 每個 categorical 的類別數
    hidden: int = 512           # MLP 寬度
    cnn_depth: int = 48         # encoder 首層通道（之後每層 ×2）
    bins: int = 255             # reward/value two-hot 的 bin 數
    bin_lo: float = -20.0
    bin_hi: float = 20.0
    horizon: int = 15           # 想像 rollout 長度
    gamma: float = 0.997        # 折扣（DreamerV3 預設）
    lam: float = 0.95           # λ-return
    free_bits: float = 1.0      # KL free bits（nat）
    beta_dyn: float = 0.5       # dynamics KL 權重
    beta_rep: float = 0.1       # representation KL 權重
    unimix: float = 0.01        # 1% 均勻混入，避免 categorical 太尖
    actor_ent: float = 3e-4     # actor 熵係數
    slow_tau: float = 0.02      # critic EMA 慢目標更新率
    lr_model: float = 1e-4
    lr_ac: float = 3e-5
    # 批次（VRAM 主要成本=decoder 重建 B×L 張影像）。RTX3070 實測：B12×L32=3.26GB/524ms、
    # B16×L48=6.0GB/780ms。**HK 遊戲同 GPU 渲染也吃 VRAM**，故預設 B12×L32 留 ~4.7GB 給遊戲。
    # OOM 就調小、VRAM 有餘可加大（--batch/--length）。train_step ~0.5s ×train_steps=每場訓練時間。
    batch: int = 12             # 序列批次大小
    length: int = 32            # 序列長度

    @property
    def stoch_dim(self):
        return self.stoch * self.classes

    @property
    def feat_dim(self):
        return self.deter + self.stoch_dim


# ---------- symlog / two-hot（DreamerV3 的尺度穩健化）----------
def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))


class TwoHot:
    """symlog 空間的 two-hot 編碼／解碼。pred = symexp(softmax·bins)。"""
    def __init__(self, cfg, device):
        self.bins = torch.linspace(cfg.bin_lo, cfg.bin_hi, cfg.bins, device=device)

    def encode(self, y):
        """y(...): 真值 -> symlog -> two-hot (..., bins)。"""
        y = symlog(y).clamp(self.bins[0], self.bins[-1])
        idx = torch.searchsorted(self.bins, y, right=True).clamp(1, len(self.bins) - 1)
        lo, hi = self.bins[idx - 1], self.bins[idx]
        w_hi = (y - lo) / (hi - lo + 1e-8)
        oh = torch.zeros(*y.shape, len(self.bins), device=y.device)
        oh.scatter_(-1, idx.unsqueeze(-1), w_hi.unsqueeze(-1))
        oh.scatter_(-1, (idx - 1).unsqueeze(-1), (1 - w_hi).unsqueeze(-1))
        return oh

    def loss(self, logits, y):
        """-log p（two-hot 交叉熵）。logits(...,bins), y(...)。"""
        target = self.encode(y)
        logp = F.log_softmax(logits, -1)
        return -(target * logp).sum(-1)

    def mean(self, logits):
        """期望值（解回真值空間）。"""
        return symexp((F.softmax(logits, -1) * self.bins).sum(-1))


def _mlp(sizes, act=nn.SiLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if i < len(sizes) - 2:
            layers += [nn.LayerNorm(sizes[i + 1]), act()]
    return nn.Sequential(*layers)


# ---------- Encoder / Decoder（CNN，96→6 四層 stride2）----------
class Encoder(nn.Module):
    def __init__(self, cfg, in_ch):
        super().__init__()
        d = cfg.cnn_depth
        chs = [in_ch, d, 2 * d, 4 * d, 8 * d]
        layers = []
        for i in range(4):
            layers += [nn.Conv2d(chs[i], chs[i + 1], 4, stride=2, padding=1),
                       nn.GroupNorm(1, chs[i + 1]), nn.SiLU()]
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            n = self.conv(torch.zeros(1, in_ch, config.NET_SIZE, config.NET_SIZE)).flatten(1).shape[1]
        self.out_dim = n

    def forward(self, x):
        return self.conv(x).flatten(1)


class Decoder(nn.Module):
    def __init__(self, cfg, out_ch, enc_spatial):
        super().__init__()
        d = cfg.cnn_depth
        self.c0, self.s0 = 8 * d, enc_spatial          # 還原成 encoder 末層 (8d, s0, s0)
        self.fc = nn.Linear(cfg.feat_dim, self.c0 * self.s0 * self.s0)
        chs = [8 * d, 4 * d, 2 * d, d]
        layers = []
        for i in range(3):
            layers += [nn.ConvTranspose2d(chs[i], chs[i + 1], 4, stride=2, padding=1),
                       nn.GroupNorm(1, chs[i + 1]), nn.SiLU()]
        layers += [nn.ConvTranspose2d(d, out_ch, 4, stride=2, padding=1)]   # 末層回 out_ch
        self.deconv = nn.Sequential(*layers)

    def forward(self, feat):
        h = self.fc(feat).view(-1, self.c0, self.s0, self.s0)
        return self.deconv(h)                          # (B, out_ch, NET_SIZE, NET_SIZE)


# ---------- RSSM（決定性 GRU + categorical 隨機潛在）----------
class RSSM(nn.Module):
    def __init__(self, cfg, action_dim, embed_dim):
        super().__init__()
        self.cfg = cfg
        s = cfg.stoch_dim
        self.act_in = _mlp([s + action_dim, cfg.hidden])           # [z,a] -> GRU 輸入
        self.gru = nn.GRUCell(cfg.hidden, cfg.deter)
        self.prior_net = nn.Sequential(_mlp([cfg.deter, cfg.hidden]),
                                       nn.SiLU(), nn.Linear(cfg.hidden, s))
        self.post_net = nn.Sequential(_mlp([cfg.deter + embed_dim, cfg.hidden]),
                                      nn.SiLU(), nn.Linear(cfg.hidden, s))

    def initial(self, batch, device):
        return {"deter": torch.zeros(batch, self.cfg.deter, device=device),
                "stoch": torch.zeros(batch, self.cfg.stoch_dim, device=device)}

    def _dist_sample(self, logits):
        c = self.cfg
        logits = logits.reshape(*logits.shape[:-1], c.stoch, c.classes)
        probs = F.softmax(logits, -1)
        probs = (1 - c.unimix) * probs + c.unimix / c.classes
        logits = torch.log(probs)
        sample = torch.distributions.OneHotCategorical(logits=logits).sample()
        sample = sample + (probs - probs.detach())                 # straight-through
        return sample.reshape(*sample.shape[:-2], -1), logits

    def img_step(self, state, action):
        """先驗一步：用 (prev stoch, action) 推進 deter，再從 deter 取先驗 stoch。"""
        x = self.act_in(torch.cat([state["stoch"], action], -1))
        deter = self.gru(x, state["deter"])
        logits = self.prior_net(deter)
        stoch, logits = self._dist_sample(logits)
        return {"deter": deter, "stoch": stoch}, logits

    def obs_step(self, state, action, embed):
        """後驗一步：先 img_step 推 deter，再結合觀測 embed 取後驗 stoch。"""
        prior, prior_logits = self.img_step(state, action)
        logits = self.post_net(torch.cat([prior["deter"], embed], -1))
        stoch, post_logits = self._dist_sample(logits)
        post = {"deter": prior["deter"], "stoch": stoch}
        return post, post_logits, prior_logits

    def observe(self, embed_seq, action_seq, state):
        """掃過整段序列（用後驗）。embed_seq/action_seq: (B,L,*)。
        對齊慣例：index t 的 action 是「導致 obs_t 的動作」(a_{t-1})，第一步為 0。"""
        B, L = embed_seq.shape[:2]
        posts, post_lg, prior_lg = [], [], []
        for t in range(L):
            state, plg, qlg = self.obs_step(state, action_seq[:, t], embed_seq[:, t])
            posts.append(state); post_lg.append(plg); prior_lg.append(qlg)
        stack = lambda key: torch.stack([p[key] for p in posts], 1)
        feats = torch.cat([stack("deter"), stack("stoch")], -1)     # (B,L,feat)
        return feats, torch.stack(post_lg, 1), torch.stack(prior_lg, 1), posts[-1]

    def kl_loss(self, post_lg, prior_lg):
        """dyn/rep KL（含 free bits + balancing）。logits: (...,stoch,classes)。"""
        c = self.cfg
        def kl(a, b):
            pa = torch.distributions.Categorical(logits=a)
            pb = torch.distributions.Categorical(logits=b)
            return torch.distributions.kl_divergence(pa, pb).sum(-1)  # 各 categorical 加總
        sg = lambda x: x.detach()
        dyn = kl(sg(post_lg), prior_lg).clamp(min=c.free_bits)
        rep = kl(post_lg, sg(prior_lg)).clamp(min=c.free_bits)
        return c.beta_dyn * dyn.mean() + c.beta_rep * rep.mean(), dyn.mean(), rep.mean()


def feat_of(state):
    return torch.cat([state["deter"], state["stoch"]], -1)


# ---------- 頭：reward / continue / decoder 由 WorldModel 統整 ----------
class WorldModel(nn.Module):
    def __init__(self, cfg, in_ch, action_dim):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg, in_ch)
        enc_spatial = config.NET_SIZE // 16                          # 四次 stride2
        self.rssm = RSSM(cfg, action_dim, self.encoder.out_dim)
        self.decoder = Decoder(cfg, in_ch, enc_spatial)
        self.reward = nn.Sequential(_mlp([cfg.feat_dim, cfg.hidden]),
                                    nn.SiLU(), nn.Linear(cfg.hidden, cfg.bins))
        self.cont = nn.Sequential(_mlp([cfg.feat_dim, cfg.hidden]),
                                  nn.SiLU(), nn.Linear(cfg.hidden, 1))

    def encode(self, obs):
        return self.encoder(obs)


# ---------- Actor / Critic（潛在空間；動作=11 獨立 Bernoulli）----------
class Actor(nn.Module):
    def __init__(self, cfg, action_dim):
        super().__init__()
        self.net = nn.Sequential(_mlp([cfg.feat_dim, cfg.hidden, cfg.hidden]),
                                 nn.SiLU(), nn.Linear(cfg.hidden, action_dim))

    def dist(self, feat):
        return torch.distributions.Bernoulli(logits=self.net(feat))

    def act(self, feat, deterministic=False):
        logits = self.net(feat)
        if deterministic:
            return (logits > 0).float()
        return torch.distributions.Bernoulli(logits=logits).sample()


class Critic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(_mlp([cfg.feat_dim, cfg.hidden, cfg.hidden]),
                                 nn.SiLU(), nn.Linear(cfg.hidden, cfg.bins))

    def forward(self, feat):
        return self.net(feat)
