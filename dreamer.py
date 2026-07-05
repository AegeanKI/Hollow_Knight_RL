"""DreamerV3 的資料、想像、損失與 learner——與真實環境解耦，可離線測試（見檔末 _selftest）。

對應 `ppo.py` 之於 PPO。`train_dreamer.py` 負責接真實 env。
"""
import numpy as np
import torch
import torch.nn.functional as F

import config
from dreamer_model import (DreamerConfig, WorldModel, Actor, Critic, TwoHot, feat_of)


# ---------------- 序列 replay（存 uint8 省 RAM）----------------
class SequenceReplay:
    """逐 episode 存 (obs, action, reward, cont)；sample 連續 L 幀的序列批次。

    對齊慣例（與 RSSM.observe 一致）：index t 存「obs_t、導致 obs_t 的動作 a_{t-1}、
    抵達 obs_t 得到的 reward、cont_t(=1-done)」。第一步 action=0, reward=0, cont=1。
    """
    def __init__(self, capacity_steps=200_000, protected_weight=1, tail_bias_prob=0.5):
        self.eps = []                 # FIFO 主體：每個 ep dict of np arrays，超 cap 從頭驅逐
        self.protected = []           # 不驅逐、不計入 cap（PPO 勝場 kickstart：保住「贏的起點」與 reward 刷新）
        self.cap = capacity_steps
        self.protected_weight = protected_weight   # 過抽倍數：sample 時勝場 episode 重複列入 → 想像更常從勝 posterior 起步
        self.tail_bias_prob = tail_bias_prob       # 尾段偏抽機率：抽到勝場時多大機率強取「含終局末段」window（0=關）
        self._n = 0                   # 只計 FIFO 主體步數
        self._protn = 0               # 保護區步數

    def add_episode(self, obs, act, rew, cont, protect=False):
        ep = {"obs": np.asarray(obs, np.uint8), "act": np.asarray(act, np.float32),
              "rew": np.asarray(rew, np.float32), "cont": np.asarray(cont, np.float32)}
        if protect:                                   # 永久保留，不驅逐、不計 cap
            self.protected.append(ep); self._protn += len(ep["rew"]); return
        self.eps.append(ep)
        self._n += len(ep["rew"])
        while self._n > self.cap and len(self.eps) > 1:
            self._n -= len(self.eps.pop(0)["rew"])

    def n_protected(self):
        return len(self.protected)

    def __len__(self):
        return self._n + self._protn

    def can_sample(self, length):
        return any(len(e["rew"]) >= length for e in self.eps) or \
               any(len(e["rew"]) >= length for e in self.protected)

    def save_protected(self, path):
        torch.save(self.protected, path)             # 保護區持久化 → resume 免重跑 live PPO

    def load_protected(self, path):
        self.protected = torch.load(path)
        self._protn = sum(len(e["rew"]) for e in self.protected)

    def sample(self, batch, length, device):
        # 勝場過抽 + 尾段偏抽（見 [[dreamer-zero-win-rootcause]]；兩者都可調/可關）：
        # ①過抽：protected(勝場) 重複 protected_weight 次 → 被抽機率 ×weight（=1 關）。
        # ②尾段偏抽：抽到勝場時 tail_bias_prob 機率強取「含終局末段」window（=0 關，全 uniform）。
        #   加它是為了把 +560 勝場終局塞進 imagination（reward head 才學得到 +28）。**但 reward head 學會後，
        #   過強的過抽/尾段偏抽會讓想像 return 變雙峰（近勝收尾 ~560 vs 滿血全場 ~8）→ 撐大 ret_denom → 稀釋
        #   advantage、且想像偏樂觀（偏練收尾、actor 實戰到不了）。學會 +28 後可調小/關，把訊號拉回全場戰鬥。
        fifo = [e for e in self.eps if len(e["rew"]) >= length]
        prot = [e for e in self.protected if len(e["rew"]) >= length]
        pool = fifo + prot * self.protected_weight
        n_fifo = len(fifo)
        obs, act, rew, cont = [], [], [], []
        for _ in range(batch):
            idx = np.random.randint(len(pool))
            e = pool[idx]; L = len(e["rew"])
            if idx >= n_fifo and self.tail_bias_prob > 0 and np.random.rand() < self.tail_bias_prob:
                s = L - length                                # 勝場、tail_bias_prob 機率取含終局末段
            else:
                s = np.random.randint(0, L - length + 1)      # uniform（非勝場，或勝場另一機率）
            obs.append(e["obs"][s:s + length]); act.append(e["act"][s:s + length])
            rew.append(e["rew"][s:s + length]); cont.append(e["cont"][s:s + length])
        to = lambda x, dt: torch.as_tensor(np.stack(x), dtype=dt, device=device)
        return {"obs": to(obs, torch.uint8).float() / 255.0,     # (B,L,C,H,W)
                "act": to(act, torch.float32),                   # (B,L,A)
                "rew": to(rew, torch.float32),                   # (B,L)
                "cont": to(cont, torch.float32)}                 # (B,L)

    def _sample_obs_act(self, pool, batch, length, device):
        """從 pool（episode list）均勻抽 obs+act 序列窗（BC/DAgger 用，不需 reward/cont）。空回 None。"""
        usable = [e for e in pool if len(e["rew"]) >= length]
        if not usable:
            return None
        obs, act = [], []
        for _ in range(batch):
            e = usable[np.random.randint(len(usable))]; L = len(e["rew"])
            s = np.random.randint(0, L - length + 1)
            obs.append(e["obs"][s:s + length]); act.append(e["act"][s:s + length])
        to = lambda x, dt: torch.as_tensor(np.stack(x), dtype=dt, device=device)
        return {"obs": to(obs, torch.uint8).float() / 255.0, "act": to(act, torch.float32)}

    def sample_protected(self, batch, length, device):
        """只從保護區（勝場 demo）**均勻**抽序列，供 latent-BC（要涵蓋整條軌跡 → 不用尾段偏抽）。"""
        return self._sample_obs_act(self.protected, batch, length, device)

    def sample_fifo(self, batch, length, device):
        """只從 FIFO 主體（**actor 自己走到的狀態**）均勻抽序列，供 DAgger（對這些狀態查專家當 target）。
        FIFO 空（resume 剛開跑、尚未 collect）回 None → 該步略過 DAgger。"""
        return self._sample_obs_act(self.eps, batch, length, device)


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


def _env11_to_model13_t(e):
    """torch 版 11 維 env 動作 → 13 維方向-Cat 模型動作（DAgger 查 PPO 後轉；規則同 train_dreamer
    `_env11_to_model13`：方向衝突→「無」）。e: (..., 11) -> (..., 13)。"""
    up, down, left, right = e[..., 0], e[..., 1], e[..., 2], e[..., 3]
    v_up = up * (1 - down); v_down = down * (1 - up); v_none = 1 - v_up - v_down
    h_left = left * (1 - right); h_right = right * (1 - left); h_none = 1 - h_left - h_right
    return torch.cat([torch.stack([v_none, v_up, v_down], -1),
                      torch.stack([h_none, h_left, h_right], -1), e[..., 4:]], -1)


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
        self.expert = None                               # DAgger 專家（PPO ActorCritic）；--bc-mode dagger 時 set

    def set_expert(self, expert):
        """DAgger 用：掛上可查詢的專家 policy（PPO）。expert.forward(obs,None)->(logits(B,11),_)。"""
        self.expert = expert

    # ---- 線上行動（收集時用；維護 RSSM 隱狀態）----
    @torch.no_grad()
    def init_state(self):
        st = self.wm.rssm.initial(1, self.device)
        a = torch.zeros(1, self.actor.net[-1].out_features, device=self.device)
        return st, a

    @torch.no_grad()
    def act(self, obs_np, state, prev_action, deterministic=False):
        """obs_np: (C,H,W) float[0,1]。
        回傳 (env_action uint8[11] 送 env, model_action float[13] 存 replay/餵 RSSM, new_state, model_tensor)。"""
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        embed = self.wm.encode(obs)
        post, _, _ = self.wm.rssm.obs_step(state, prev_action, embed)
        feat = feat_of(post)
        a = self.actor.act(feat, deterministic)                       # (1,13) 模型動作
        a_model = a.squeeze(0).cpu().numpy().astype(np.float32)
        a_env = config.dir_model_to_env(a_model).astype(np.uint8)     # (11,) 送 actuator
        return a_env, a_model, post, a

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
            logp.append(d.log_prob(actions[t]))      # 混合分布 log_prob/entropy 已對各分量加總
            ent.append(d.entropy())
        logp = torch.stack(logp, 0); ent = torch.stack(ent, 0)
        actor_loss = -(logp * adv).mean() - cfg.actor_ent * ent.mean()
        return actor_loss, critic_loss, {"imag_ret": ret.mean().item(),
                                         "ret_denom": denom, "ent": ent.mean().item(),
                                         # 診斷 actor 崩壞：denom 是否黏在 1.0(=return spread<1 被 clamp)、
                                         # adv 量級是否極小(=熵項相對變強)、想像 reward 是否有信號(std≈0=reward head 恆吐 0)
                                         "adv_absmean": adv.detach().abs().mean().item(),
                                         "adv_std": adv.detach().std().item(),
                                         "imag_rew_std": reward.detach().std().item()}

    def _bc_loss(self, demo_batch):
        """latent-BC：把 demo obs 經(當前)WM 編成潛在、監督 actor 從潛在預測 demo 動作。
        WM 編碼在 no_grad 下（BC 只訓 actor、不回傳 WM；WM 由 _wm_loss 訓）。回傳 -log_prob(demo action)。
        對齊：replay 慣例 act[t]=導致 obs_t 的動作(a_{t-1})；actor 在 latent(obs_t) 要預測「在 obs_t 採取的動作」
        ＝act[t+1]。故用 feat[:, :-1] 配 target act[:, 1:]（也順帶避開 act[0]=zeros 的非法 one-hot）。"""
        B, L = demo_batch["obs"].shape[:2]
        C = demo_batch["obs"].shape[2]
        with torch.no_grad():
            embed = self.wm.encode(demo_batch["obs"].reshape(B * L, C, *demo_batch["obs"].shape[3:])).reshape(B, L, -1)
            feats, _, _, _ = self.wm.rssm.observe(embed, demo_batch["act"], self.wm.rssm.initial(B, self.device))
        feat = feats[:, :-1].reshape(B * (L - 1), -1)            # latent(obs_0..obs_{L-2})
        tgt = demo_batch["act"][:, 1:].reshape(B * (L - 1), -1)  # 在該 obs 採取的動作＝act[1..L-1]（全合法 one-hot）
        return -self.actor.dist(feat).log_prob(tgt).mean()

    def _dagger_loss(self, fifo_batch):
        """DAgger：對 **actor 自己走到的狀態**（FIFO obs）查專家 PPO 當 target → 監督 actor 預測。
        治 covariate shift（demo-BC 只覆蓋 demo 狀態、actor 到不了）。對齊：查 π_PPO(obs_t)＝「在 obs_t 當下
        該做的動作」，與 latent(obs_t)＝feat_t **同格、不位移**（跟 demo-BC 位移一格不同）；PPO 恆吐合法動作、
        無 act[0]=zeros 問題，故用全部 t。RSSM 用 FIFO 自己的 act 捲 latent（真實 transition）。"""
        B, L = fifo_batch["obs"].shape[:2]
        C = fifo_batch["obs"].shape[2]
        obs_flat = fifo_batch["obs"].reshape(B * L, C, *fifo_batch["obs"].shape[3:])
        with torch.no_grad():
            logits, _ = self.expert.forward(obs_flat, None)          # (B*L, 11) PPO logits（extra=None 只算 actor）
            a_env = (torch.sigmoid(logits) > 0.5).float()            # deterministic 專家動作（11 維）
            a_ppo = _env11_to_model13_t(a_env)                       # (B*L, 13) 方向-Cat（衝突→無，同 seed_from_ppo）
            embed = self.wm.encode(obs_flat).reshape(B, L, -1)
            feats, _, _, _ = self.wm.rssm.observe(embed, fifo_batch["act"], self.wm.rssm.initial(B, self.device))
        return -self.actor.dist(feats.reshape(B * L, -1)).log_prob(a_ppo).mean()

    def train_step(self, batch, bc_batch=None, bc_weight=0.0, bc_mode="demo"):
        # Dreamer 內部 reward 放大（全成分同乘 K；見 cfg.reward_scale）。在 world model 入口乘一次即可：
        # reward predictor 學放大後的目標 → 想像用 predictor 輸出 → returns/denom/critic 全在放大空間、自洽。
        # 存進 replay 的仍是原始 reward（可隨時改 K 免重收）；env/config.RW_* 不動、PPO 免疫。
        batch = {**batch, "rew": batch["rew"] * self.cfg.reward_scale}
        wm_loss, feats, wm_stats = self._wm_loss(batch)
        self.opt_wm.zero_grad(); wm_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.wm.parameters(), 100.0)
        self.opt_wm.step()

        actor_loss, critic_loss, ac_stats = self._imagine_ac_loss(feats)
        if bc_batch is not None and bc_weight > 0:            # BC/DAgger：把 actor 拉向專家動作（治 reachability）
            bc = self._dagger_loss(bc_batch) if bc_mode == "dagger" else self._bc_loss(bc_batch)
            actor_loss = actor_loss + bc_weight * bc
            ac_stats["bc_loss"] = bc.item()
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
    C, A = config.NET_CHANNELS, config.N_ACTIONS_MODEL   # 模型動作 13 維（方向-Categorical）
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
    a_env, a_model, state, a = learner.act(obs, state, a)
    print(f"act OK: env action {a_env.shape} sum={int(a_env.sum())}, model action {a_model.shape}")
    assert np.isfinite(st["wm_loss"]) and np.isfinite(st["actor_loss"])
    print("[OK] dreamer 自測通過")


if __name__ == "__main__":
    _selftest()
