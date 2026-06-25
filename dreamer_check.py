"""DreamerV3 實作正確性 + inference 速度驗證（離線，不需遊戲）。

跑：python dreamer_check.py
涵蓋：symlog 互逆、two-hot 往返、λ-return 對照解析、world model 過擬合(rec/reward 下降、
reward 頭學出已知值)、梯度流、act() 是否達 15fps。
"""
import sys
import time

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")     # 避免 cp950 console 印 unicode 崩潰
except Exception:
    pass

import config
from dreamer_model import symlog, symexp, TwoHot, DreamerConfig
from dreamer import DreamerLearner, SequenceReplay, lambda_return

dev = "cuda" if torch.cuda.is_available() else "cpu"
ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"[{'OK ' if cond else 'FAIL'}] {name} {extra}")


# A. symlog/symexp 互逆
x = torch.tensor([-1e4, -12.3, -1.0, 0.0, 0.5, 7.0, 900.0, 1e5])
rt = symexp(symlog(x))
check("symlog/symexp 互逆", torch.allclose(rt, x, rtol=1e-4, atol=1e-3),
      f"max_err={ (rt-x).abs().max().item():.2e}")

# B. two-hot 編碼→期望值 往返
cfg = DreamerConfig()
th = TwoHot(cfg, "cpu")
ys = torch.tensor([-500., -3., 0., 0.7, 5., 250., 800.])
target = th.encode(ys)                                   # two-hot 機率
recon = symexp((target * th.bins.cpu()).sum(-1))         # 用 target 當分佈取期望
check("two-hot 編碼往返", torch.allclose(recon, ys, rtol=1e-3, atol=1e-2),
      f"max_err={(recon-ys).abs().max().item():.2e}")
check("two-hot 機率合法(和=1、非負)",
      torch.allclose(target.sum(-1), torch.ones(len(ys))) and (target >= 0).all())

# C. λ-return 對照解析（常數 reward r、cont=1、bootstrap V_H=r/(1-γ) → 整段應=r/(1-γ)）
H, B = 40, 1
g, lam, r = 0.99, 0.95, 1.0
analytic = r / (1 - g)
reward = torch.full((H, B), r)
cont = torch.ones(H, B)
value = torch.full((H + 1, B), analytic)                 # 真值 baseline
R = lambda_return(reward, value, cont, g, lam)
check("λ-return 對照解析(常數獎勵不動點)", torch.allclose(R, torch.full((H, B), analytic), atol=1e-3),
      f"R[0]={R[0,0].item():.3f} vs {analytic:.3f}")
# λ=1 時應退化成蒙地卡羅；λ=0 時應=r+γV
R0 = lambda_return(reward, value, cont, g, 0.0)
check("λ=0 退化成一步 TD", torch.allclose(R0[0], torch.tensor(r + g * analytic), atol=1e-3))

# D+E. world model 過擬合固定 batch（最強的端到端正確性訊號）
print("\n--- world model 過擬合測試（固定 batch 訓練）---")
small = DreamerConfig(batch=4, length=12, horizon=6, deter=128, stoch=16, classes=16,
                      hidden=128, cnn_depth=24, free_bits=0.0)   # free_bits=0 才看得到 KL 真的降
learner = DreamerLearner(small, config.NET_CHANNELS, config.N_ACTIONS, dev)
C = config.NET_CHANNELS
# 固定 batch：用「結構化」觀測(非純噪聲，較可重建)，reward 設已知常數 2.0
torch.manual_seed(0); np.random.seed(0)
base = np.random.rand(small.batch, 1, C, config.NET_SIZE, config.NET_SIZE).astype(np.float32)
obs = np.broadcast_to(base, (small.batch, small.length, C, config.NET_SIZE, config.NET_SIZE)).copy()
KNOWN_R = 2.0
batch = {"obs": torch.tensor(obs, device=dev),
         "act": torch.tensor(np.random.randint(0, 2, (small.batch, small.length, config.N_ACTIONS)).astype(np.float32), device=dev),
         "rew": torch.full((small.batch, small.length), KNOWN_R, device=dev),
         "cont": torch.ones(small.batch, small.length, device=dev)}
rec0 = rew0 = None
for it in range(400):
    st = learner.train_step(batch)
    if it == 0:
        rec0, rew0 = st["rec"], st["rew"]
rec1, rew1 = st["rec"], st["rew"]
check("world model 重建 loss 大幅下降(學得動)", rec1 < rec0 * 0.5, f"rec {rec0:.0f}→{rec1:.0f}")
check("reward 頭 loss 下降", rew1 < rew0 * 0.5, f"rew_loss {rew0:.3f}→{rew1:.3f}")
# reward 頭是否真的學出已知常數 2.0：跑一次後驗取 feat 餵 reward 頭
with torch.no_grad():
    B_, L_ = small.batch, small.length
    embed = learner.wm.encode(batch["obs"].reshape(B_ * L_, C, config.NET_SIZE, config.NET_SIZE)).reshape(B_, L_, -1)
    feats, *_ = learner.wm.rssm.observe(embed, batch["act"], learner.wm.rssm.initial(B_, dev))
    pred_r = learner.twohot.mean(learner.wm.reward(feats.reshape(B_ * L_, -1))).mean().item()
check("reward 頭學出已知常數(≈2.0)", abs(pred_r - KNOWN_R) < 0.3, f"pred={pred_r:.3f}")

# E. 梯度有流到三組網路
learner2 = DreamerLearner(small, C, config.N_ACTIONS, dev)
st = learner2._wm_loss(batch)[0]
learner2.opt_wm.zero_grad(); st.backward()
g_enc = any(p.grad is not None and p.grad.abs().sum() > 0 for p in learner2.wm.encoder.parameters())
g_rssm = any(p.grad is not None and p.grad.abs().sum() > 0 for p in learner2.wm.rssm.parameters())
check("梯度流到 encoder/RSSM", g_enc and g_rssm)
al, cl, _ = learner2._imagine_ac_loss(learner2._wm_loss(batch)[1])
learner2.opt_ac.zero_grad(); (al + cl).backward()
g_actor = any(p.grad is not None and p.grad.abs().sum() > 0 for p in learner2.actor.parameters())
g_critic = any(p.grad is not None and p.grad.abs().sum() > 0 for p in learner2.critic.parameters())
check("梯度流到 actor/critic", g_actor and g_critic)

# F. inference 速度（act() 線上路徑，全尺寸模型）
print("\n--- inference 速度（act 線上路徑，全尺寸）---")
full = DreamerLearner(DreamerConfig(), C, config.N_ACTIONS, dev)
state, prev_a = full.init_state()
obs1 = np.random.rand(C, config.NET_SIZE, config.NET_SIZE).astype(np.float32)
for _ in range(20):                                       # warmup
    _, s, a = full.act(obs1, state, prev_a)
if dev == "cuda":
    torch.cuda.synchronize()
t0 = time.perf_counter()
N = 200
st_, pa_ = state, prev_a
for _ in range(N):
    _, st_, pa_ = full.act(obs1, st_, pa_)
if dev == "cuda":
    torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / N * 1000
check("act() 達 15fps(≤66.7ms/tick)", dt <= 66.7, f"{dt:.2f} ms = {dt/66.7*100:.1f}% of tick")

print("\n==== 總結:", "全部通過 ✅" if ok else "有 FAIL ❌", "====")
