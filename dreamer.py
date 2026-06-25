"""DreamerV3 的資料、想像、損失與 learner——與真實環境解耦，可離線測試（見檔末 _selftest）。

對應 `ppo.py` 之於 PPO。`train_dreamer.py` 負責接真實 env。
"""
import numpy as np
import torch
import torch.nn.functional as F

from dreamer_model import (DreamerConfig, WorldModel, Actor, Critic, TwoHot, feat_of)


# ---------------- 序列 replay（存 uint8 省 RAM）----------------
class SequenceReplay:
    """逐 episode 存 (obs, action, reward, cont)；sample 連續 L 幀的序列批次。

    對齊慣例（與 RSSM.observe 一致）：index t 存「obs_t、導致 obs_t 的動作 a_{t-1}、
    抵達 obs_t 得到的 reward、cont_t(=1-done)」。第一步 action=0, reward=0, cont=1。
    """
    def __init__(self, capacity_steps=200_000):
        self.eps = []                 # 每個 ep: dict of np arrays
        self.cap = capacity_steps
        self._n = 0

    def add_episode(self, obs, act, rew, cont):
        ep = {"obs": np.asarray(obs, np.uint8), "act": np.asarray(act, np.float32),
              "rew": np.asarray(rew, np.float32), "cont": np.asarray(cont, np.float32)}
        self.eps.append(ep)
        self._n += len(ep["rew"])
        while self._n > self.cap and len(self.eps) > 1:
            self._n -= len(self.eps.pop(0)["rew"])

    def __len__(self):
        return self._n

    def can_sample(self, length):
        return any(len(e["rew"]) >= length for e in self.eps)

    def sample(self, batch, length, device):
        usable = [e for e in self.eps if len(e["rew"]) >= length]
        obs, act, rew, cont = [], [], [], []
        for _ in range(batch):
            e = usable[np.random.randint(len(usable))]
            s = np.random.randint(0, len(e["rew"]) - length + 1)
            obs.append(e["obs"][s:s + length]); act.append(e["act"][s:s + length])
            rew.append(e["rew"][s:s + length]); cont.append(e["cont"][s:s + length])
        to = lambda x, dt: torch.as_tensor(np.stack(x), dtype=dt, device=device)
        return {"obs": to(obs, torch.uint8).float() / 255.0,     # (B,L,C,H,W)
                "act": to(act, torch.float32),                   # (B,L,A)
                "rew": to(rew, torch.float32),                   # (B,L)
                "cont": to(cont, torch.float32)}                 # (B,L)


# ---------------- λ-return（想像軌跡上）----------------
def lambda_return(reward, value, cont, gamma, lam):
    """reward/cont: (H, B)；value: (H+1, B)。回傳 R: (H, B)。
    R_t = r_t + gamma*cont_t*((1-λ)V_{t+1} + λ R_{t+1})，R_H bootstrap=V_H。"""
    H = reward.shape[0]
    R = [None] * H
    nxt = value[-1]
    for t in reversed(range(H)):
        R[t] = reward[t] + gamma * cont[t] * ((1 - lam) * value[t + 1] + lam * nxt)
        nxt = R[t]
    return torch.stack(R, 0)


# ---------------- Learner（world model + 想像 actor-critic）----------------
class DreamerLearner:
    def __init__(self, cfg, in_ch, action_dim, device):
        self.cfg, self.device = cfg, device
        self.wm = WorldModel(cfg, in_ch, action_dim).to(device)
        self.actor = Actor(cfg, action_dim).to(device)
        self.critic = Critic(cfg).to(device)
        self.slow = Critic(cfg).to(device)               # EMA 慢目標
        self.slow.load_state_dict(self.critic.state_dict())
        for p in self.slow.parameters():
            p.requires_grad_(False)
        self.twohot = TwoHot(cfg, device)
        self.opt_wm = torch.optim.Adam(self.wm.parameters(), cfg.lr_model)
        self.opt_ac = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), cfg.lr_ac)
        self.ret_lo, self.ret_hi = 0.0, 1.0              # return 百分位 EMA（正規化用）

    # ---- 線上行動（收集時用；維護 RSSM 隱狀態）----
    @torch.no_grad()
    def init_state(self):
        st = self.wm.rssm.initial(1, self.device)
        a = torch.zeros(1, self.actor.net[-1].out_features, device=self.device)
        return st, a

    @torch.no_grad()
    def act(self, obs_np, state, prev_action, deterministic=False):
        """obs_np: (C,H,W) float[0,1]。回傳 (action_np uint8, new_state, action_tensor)。"""
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        embed = self.wm.encode(obs)
        post, _, _ = self.wm.rssm.obs_step(state, prev_action, embed)
        feat = feat_of(post)
        a = self.actor.act(feat, deterministic)
        return a.squeeze(0).cpu().numpy().astype(np.uint8), post, a

    # ---- world model 更新 ----
    def _wm_loss(self, batch):
        cfg = self.cfg
        B, L = batch["obs"].shape[:2]
        C = batch["obs"].shape[2]
        obs = batch["obs"].reshape(B * L, C, *batch["obs"].shape[3:])
        embed = self.wm.encode(obs).reshape(B, L, -1)
        state = self.wm.rssm.initial(B, self.device)
        feats, post_lg, prior_lg, last = self.wm.rssm.observe(embed, batch["act"], state)
        flat = feats.reshape(B * L, -1)
        # 重建（MSE 於 [-0.5,0.5]）
        recon = self.wm.decoder(flat)
        rec_loss = F.mse_loss(recon, obs - 0.5, reduction="none").reshape(B * L, -1).sum(-1).mean()
        # reward / continue
        rew_loss = self.twohot.loss(self.wm.reward(flat), batch["rew"].reshape(B * L)).mean()
        cont_logit = self.wm.cont(flat).squeeze(-1)
        cont_loss = F.binary_cross_entropy_with_logits(cont_logit, batch["cont"].reshape(B * L))
        kl, dyn, rep = self.wm.rssm.kl_loss(post_lg, prior_lg)
        loss = rec_loss + rew_loss + cont_loss + kl
        return loss, feats.detach(), {"rec": rec_loss.item(), "rew": rew_loss.item(),
                                      "cont": cont_loss.item(), "dyn": dyn.item(), "rep": rep.item()}

    # ---- 想像 + actor-critic 更新 ----
    def _imagine_ac_loss(self, feats):
        cfg = self.cfg
        B, L, _ = feats.shape
        # 從所有後驗狀態出發（攤平成起點），detach 不回傳到 world model
        deter = feats[..., :cfg.deter].reshape(B * L, cfg.deter)
        stoch = feats[..., cfg.deter:].reshape(B * L, cfg.stoch_dim)
        state = {"deter": deter, "stoch": stoch}
        feat_list = [feat_of(state)]
        actions, rewards, conts = [], [], []
        for _ in range(cfg.horizon):
            f = feat_list[-1].detach()
            a = self.actor.act(f)                         # 取樣（reinforce，用 logprob 回傳）
            state, _ = self.wm.rssm.img_step(state, a)
            nf = feat_of(state)
            feat_list.append(nf)
            actions.append(a)
            rewards.append(self.twohot.mean(self.wm.reward(nf)))
            conts.append(torch.sigmoid(self.wm.cont(nf).squeeze(-1)))
        feats_imag = torch.stack(feat_list, 0)            # (H+1, B*L, feat)
        reward = torch.stack(rewards, 0)                  # (H, B*L)
        cont = torch.stack(conts, 0)                      # (H, B*L) 續存機率；折扣=gamma*cont
        # 用慢 critic 算 bootstrap value
        with torch.no_grad():
            value_slow = self.twohot.mean(self.slow(feats_imag))   # (H+1, B*L)
        ret = lambda_return(reward, value_slow, cont, cfg.gamma, cfg.lam)  # (H, B*L)
        # return 百分位 EMA 正規化
        with torch.no_grad():
            lo = torch.quantile(ret, 0.05).item()
            hi = torch.quantile(ret, 0.95).item()
            self.ret_lo += cfg.slow_tau * (lo - self.ret_lo)
            self.ret_hi += cfg.slow_tau * (hi - self.ret_hi)
            denom = max(1.0, self.ret_hi - self.ret_lo)
        # critic：fast critic 的 two-hot 迴歸到 sg(ret)
        value_logits = self.critic(feats_imag[:-1])       # (H, B*L, bins)
        critic_loss = self.twohot.loss(value_logits, ret.detach()).mean()
        # actor：reinforce + 熵；advantage 正規化
        value_fast = self.twohot.mean(value_logits).detach()
        adv = (ret.detach() - value_fast) / denom
        logp, ent = [], []
        for t in range(cfg.horizon):
            d = self.actor.dist(feat_list[t].detach())
            logp.append(d.log_prob(actions[t]).sum(-1))
            ent.append(d.entropy().sum(-1))
        logp = torch.stack(logp, 0); ent = torch.stack(ent, 0)
        actor_loss = -(logp * adv).mean() - cfg.actor_ent * ent.mean()
        return actor_loss, critic_loss, {"imag_ret": ret.mean().item(),
                                         "ret_denom": denom, "ent": ent.mean().item()}

    def train_step(self, batch):
        wm_loss, feats, wm_stats = self._wm_loss(batch)
        self.opt_wm.zero_grad(); wm_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.wm.parameters(), 100.0)
        self.opt_wm.step()

        actor_loss, critic_loss, ac_stats = self._imagine_ac_loss(feats)
        self.opt_ac.zero_grad(); (actor_loss + critic_loss).backward()
        torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 100.0)
        self.opt_ac.step()
        # 慢 critic EMA
        with torch.no_grad():
            for s, f in zip(self.slow.parameters(), self.critic.parameters()):
                s.mul_(1 - self.cfg.slow_tau).add_(self.cfg.slow_tau * f)
        return {"wm_loss": wm_loss.item(), "actor_loss": actor_loss.item(),
                "critic_loss": critic_loss.item(), **wm_stats, **ac_stats}

    def state_dict(self):
        return {"wm": self.wm.state_dict(), "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(), "slow": self.slow.state_dict(),
                "opt_wm": self.opt_wm.state_dict(), "opt_ac": self.opt_ac.state_dict(),
                "ret_lo": self.ret_lo, "ret_hi": self.ret_hi}

    def load_state_dict(self, s):
        self.wm.load_state_dict(s["wm"]); self.actor.load_state_dict(s["actor"])
        self.critic.load_state_dict(s["critic"]); self.slow.load_state_dict(s["slow"])
        self.opt_wm.load_state_dict(s["opt_wm"]); self.opt_ac.load_state_dict(s["opt_ac"])
        self.ret_lo, self.ret_hi = s["ret_lo"], s["ret_hi"]


def _selftest():
    """離線煙霧：建 learner、塞假序列、跑 train_step，確認形狀與 loss 有限。"""
    import config
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DreamerConfig(batch=2, length=8, horizon=5, deter=64, stoch=8, classes=8,
                        hidden=64, cnn_depth=16)
    C, A = config.NET_CHANNELS, config.N_ACTIONS
    learner = DreamerLearner(cfg, C, A, dev)
    rep = SequenceReplay()
    for _ in range(3):
        T = 20
        rep.add_episode(np.random.randint(0, 255, (T, C, config.NET_SIZE, config.NET_SIZE), np.uint8),
                        np.random.randint(0, 2, (T, A)).astype(np.float32),
                        np.random.randn(T).astype(np.float32),
                        np.ones(T, np.float32))
    batch = rep.sample(cfg.batch, cfg.length, dev)
    st = learner.train_step(batch)
    print("train_step OK:", {k: round(v, 4) for k, v in st.items()})
    # 線上行動
    state, a = learner.init_state()
    obs = np.random.rand(C, config.NET_SIZE, config.NET_SIZE).astype(np.float32)
    act, state, a = learner.act(obs, state, a)
    print("act OK: action shape", act.shape, "sum", int(act.sum()))
    assert np.isfinite(st["wm_loss"]) and np.isfinite(st["actor_loss"])
    print("[OK] dreamer 自測通過")


if __name__ == "__main__":
    _selftest()
