"""PPO 的資料緩衝、GAE、與更新步驟。與真實環境解耦，可離線測試。"""
import numpy as np
import torch
import torch.nn as nn


class RunningMeanStd:
    """跑動均值/方差（Welford 批次版）。① 用來估 return 尺度做正規化，狀態存進 checkpoint。"""
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return
        bm, bv, bc = x.mean(), x.var(), x.size
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        m_a, m_b = self.var * self.count, bv * bc
        self.var = (m_a + m_b + delta * delta * self.count * bc / tot) / tot
        self.count = tot

    @property
    def std(self):
        return float(np.sqrt(self.var)) + 1e-8

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, s):
        self.mean, self.var, self.count = s["mean"], s["var"], s["count"]


def mc_returns(rew, done, gamma):
    """每步的折扣蒙地卡羅回報（value-independent，只用來估 return 的尺度 σ）。"""
    rew = np.asarray(rew, dtype=np.float64)
    done = np.asarray(done, dtype=np.float64)
    G = np.zeros_like(rew)
    run = 0.0
    for t in range(len(rew) - 1, -1, -1):
        if done[t]:
            run = 0.0
        run = rew[t] + gamma * run
        G[t] = run
    return G


class RolloutBuffer:
    """收集 transitions；finish() 算出 GAE 優勢與 returns。

    區分 done(episode 邊界=term 或 trunc) 與 term(真正終局=死/勝)：
    超時(truncation)不是真終局，結尾要用最後狀態的 value bootstrap，不能當未來=0。
    """
    def __init__(self):
        self.obs, self.act, self.logp = [], [], []
        self.rew, self.val, self.done = [], [], []
        self.term, self.boot = [], []
        self.extra = []                  # privileged critic 每步特權特徵 (n_extra,)

    def add(self, obs, act, logp, rew, val, done, term, boot, extra):
        self.obs.append(obs); self.act.append(act); self.logp.append(logp)
        self.rew.append(rew); self.val.append(val); self.done.append(done)
        self.term.append(term); self.boot.append(boot); self.extra.append(extra)

    def __len__(self):
        return len(self.rew)

    def finish(self, gamma=0.99, lam=0.95, rew_scale=1.0):
        rew = np.asarray(self.rew, dtype=np.float32) * rew_scale
        val = np.asarray(self.val, dtype=np.float32)
        done = np.asarray(self.done, dtype=np.float32)   # episode 邊界（term 或 trunc）
        term = np.asarray(self.term, dtype=np.float32)   # 真正終局；只有它砍掉未來價值
        boot = np.asarray(self.boot, dtype=np.float32)   # 邊界的 V(s_next)：超時=V、真終局=0
        adv = np.zeros_like(rew)
        last = 0.0
        for t in reversed(range(len(rew))):
            if done[t]:                       # episode 邊界：用存好的 bootstrap value
                next_val = boot[t]
            elif t + 1 < len(rew):
                next_val = val[t + 1]
            else:
                next_val = 0.0                # buffer 末步且非邊界（被 F10 中途切斷）：保守 0
            nonterminal = 1.0 - term[t]       # 只有「真正終局」才不 bootstrap
            delta = rew[t] + gamma * next_val * nonterminal - val[t]
            last = delta + gamma * lam * (1.0 - done[t]) * last  # 跨 episode 邊界重置累積
            adv[t] = last
        ret = adv + val
        return adv, ret


def ppo_update(ac, opt, buffer, device, ret_rms=None, epochs=4, batch_size=256,
               clip=0.2, vf_coef=0.25, ent_coef=0.0005, max_grad=0.5, gamma=0.99, lam=0.95):
    """對緩衝內資料做數個 epoch 的 PPO 更新。回傳統計 dict。

    ① return 正規化：用 raw return 的跑動 std(ret_rms) 縮放 reward → value 目標 ~O(1)、
    且跟 curriculum 改 scale 造成的 reward 量級漂移解耦，穩住 critic。advantage 之後仍會
    逐批標準化，故 policy 梯度尺度不受影響（只動 value 目標尺度）。ret_rms=None 則不縮放。
    """
    if ret_rms is not None:
        ret_rms.update(mc_returns(buffer.rew, buffer.done, gamma))
        rew_scale = 1.0 / ret_rms.std
    else:
        rew_scale = 1.0
    adv, ret = buffer.finish(gamma=gamma, lam=lam, rew_scale=rew_scale)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    obs = torch.as_tensor(np.stack(buffer.obs), dtype=torch.float32)
    act = torch.as_tensor(np.stack(buffer.act), dtype=torch.float32)
    extra = torch.as_tensor(np.stack(buffer.extra), dtype=torch.float32)   # (N, n_extra)
    old_logp = torch.as_tensor(np.asarray(buffer.logp), dtype=torch.float32)
    adv_t = torch.as_tensor(adv, dtype=torch.float32)
    ret_t = torch.as_tensor(ret, dtype=torch.float32)

    n = len(buffer)
    idx = np.arange(n)
    stats = {"pi_loss": 0.0, "vf_loss": 0.0, "entropy": 0.0, "kl": 0.0, "n": 0}
    for _ in range(epochs):
        np.random.shuffle(idx)
        for s in range(0, n, batch_size):
            b = idx[s:s + batch_size]
            ob = obs[b].to(device); ac_b = act[b].to(device); ex_b = extra[b].to(device)
            olp = old_logp[b].to(device); ad = adv_t[b].to(device); rt = ret_t[b].to(device)

            logp, ent, val = ac.evaluate(ob, ac_b, ex_b)
            ratio = torch.exp(logp - olp)
            s1 = ratio * ad
            s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * ad
            pi_loss = -torch.min(s1, s2).mean()
            vf_loss = nn.functional.mse_loss(val, rt)
            ent_mean = ent.mean()
            loss = pi_loss + vf_coef * vf_loss - ent_coef * ent_mean

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), max_grad)
            opt.step()

            with torch.no_grad():
                stats["pi_loss"] += pi_loss.item()
                stats["vf_loss"] += vf_loss.item()
                stats["entropy"] += ent_mean.item()
                stats["kl"] += (olp - logp).mean().item()
                stats["n"] += 1
    m = max(stats["n"], 1)
    out = {k: (v / m if k != "n" else v) for k, v in stats.items()}
    out["ret_std"] = (1.0 / rew_scale) if rew_scale else 0.0   # 目前 return 尺度（給 log/CSV）
    return out
