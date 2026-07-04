"""DreamerV3 真實環境訓練迴圈（鏡像 train_rl.py，但 off-policy：收集→replay→想像訓練）。

與 train_rl.py 並存、互不影響：共用 env.py / controls / config，但用
DreamerLearner（world model + 想像 actor-critic）。F10 安全停、F9 暫停。
**不用 curriculum**：Dreamer 一律全難度（啟動時關 HKCurriculum mod、scale=1），
故無 scale 讀取/顯示、無 eval 旗標（見 [[curriculum-difficulty-approach]] 的 Dreamer 決策）。

關鍵差異 vs PPO：
- off-policy：每場存進 replay，梯度更新在 episode 之間（人在雕像大廳）做，不搶 15Hz。
- 線上行動要維護 RSSM 隱狀態（每 tick 一步後驗），故 collect 自己跑迴圈。
- 觀測沿用 env 的疊幀（NET_CHANNELS 通道）；RSSM 另外提供跨步時序。

用法：
  python train_dreamer.py --input gamepad --eval-every 10 --eval-episodes 5
  python train_dreamer.py --seed-demos 20   # 先用 demo 暖機 world model（reward 不計，僅表徵/動態）
"""
import argparse
import datetime
import gc
import glob
import os
import time

import numpy as np
import torch

import config
from controls import ControlKeys
from env import BossDamageTracker, HollowKnightEnv
from dreamer import DreamerLearner, SequenceReplay
from dreamer_model import DreamerConfig

LOG_PATH = os.path.join("logs", "train_dreamer.log")
CSV_PATH = os.path.join("logs", "dreamer_metrics.csv")
LATEST = os.path.join(config.CKPT_DIR, "dreamer_latest.pt")
BEST = os.path.join(config.CKPT_DIR, "dreamer_best.pt")
PPO_WINS = os.path.join(config.CKPT_DIR, "dreamer_ppo_wins.pt")   # seed-ppo 勝場保護區持久化（resume 免重跑 live PPO）
DEMO_BOSS_MAX = 900.0      # Attuned Hornet 滿血：demo boss_hp 是正規化(0-1)，×此還原 raw dmg（reward 用 raw）
DEMO_PLAYER_MAX = 9.0      # 滿血面具數（demo player_hp 是 raw 面具數，late-factor 分母）
OVER_PRINT_MAX = 20        # fps 診斷：第二行最多列幾個 over-tick 超時值（過多只列最大的前 N 個）
CSV_COLS = ["ep", "train_steps", "result", "dmg", "ep_steps",
            "wm_loss", "rec", "rew", "cont", "dyn", "rep",
            "actor_loss", "critic_loss", "imag_ret", "ent",
            "eval_avg_dmg", "eval_max_dmg", "eval_wins", "eval_eps",
            # actor 崩壞判因（放最末：既有 CSV 表頭不重寫，新欄接尾巴不錯位舊資料）
            "ret_denom", "adv_absmean", "adv_std", "imag_rew_std"]

_logf = None


def log(msg):
    print(msg)
    if _logf:
        _logf.write(msg + "\n"); _logf.flush()


def _to_u8(obs):
    return np.clip(obs * 255.0, 0, 255).astype(np.uint8)


def collect_episode(env, learner, should_stop, deterministic=False):
    """跑一場，回傳對齊好的序列 + 統計。index t = (obs_t, 導致 obs_t 的模型動作, 抵達 reward, cont)。
    act_buf 存「模型動作(13 維方向-Categorical)」；送 env 的是轉出來的 11 鍵。
    中途被遮擋 -> 回 {'occluded': True}（drain 後、垃圾畫素不進 replay）；reset 失敗/0 步 -> None。"""
    obs, _ = env.reset(should_stop=should_stop)
    if obs is None:
        return None
    state, prev_a = learner.init_state()
    A = config.N_ACTIONS_MODEL                    # act_buf/RSSM 吃 13 維模型動作
    obs_buf = [_to_u8(obs)]                       # obs_0
    act_buf = [np.zeros(A, np.float32)]           # a_{-1}=0（13 維）
    rew_buf = [0.0]; cont_buf = [1.0]
    boss = BossDamageTracker()
    done, steps, last_hp, info = False, 0, -1, {}
    act_ms_max, act_ms_sum = 0.0, 0.0                  # act() 計時：判尖刺是不是落在 GPU 推論
    while not done and not should_stop():
        _t = time.perf_counter()
        a_env, a_model, state, prev_a = learner.act(obs, state, prev_a, deterministic)
        _dt = (time.perf_counter() - _t) * 1000.0
        act_ms_max = max(act_ms_max, _dt); act_ms_sum += _dt
        obs2, r, term, trunc, info = env.step(a_env)
        if info.get("occluded"):                  # 中途被遮擋：obs 已污染 -> 別進持久 replay
            env.drain_until_terminal(should_stop)   # 等本場自然結束才回得了大廳
            return {"occluded": True}
        done = term or trunc
        obs_buf.append(_to_u8(obs2)); act_buf.append(a_model)
        rew_buf.append(float(r)); cont_buf.append(0.0 if term else 1.0)   # 真終局才 cont=0；截斷仍 1
        obs = obs2; steps += 1; boss.update(info)
        p = info.get("player_hp", -1)
        if p > 0:                                 # last-valid 剩血（給 rl_best tiebreaker）
            last_hp = p
    if steps == 0:
        return None
    return {"obs": obs_buf, "act": act_buf, "rew": rew_buf, "cont": cont_buf,
            "result": info.get("result"), "dmg": boss.dmg,
            "steps": steps, "end_hp": last_hp, "drop": info.get("tele_drop", 0.0),
            "fps": info.get("fps", 0.0), "over_runs": info.get("over_runs", []),
            "act_ms_max": act_ms_max, "act_ms_avg": act_ms_sum / max(1, steps),
            "max_grab": info.get("max_grab", (0.0, -1)), "max_tele": info.get("max_tele", (0.0, -1))}


def eval_dreamer(env, learner, n_eps, should_stop):
    """決定性 eval。回傳 (results, dmgs, end_hps, steps)；遮擋場丟棄重跑。
    Dreamer 假設 HKCurriculum 關閉、boss 恆滿血、scale=1 → 不設 curriculum eval 旗標。"""
    res, dmgs, hps, steps = [], [], [], []
    while len(res) < n_eps:
        if should_stop():
            break
        env.wait_until_unoccluded(should_stop)   # gate：被遮擋就等解除再開 eval 場
        if should_stop():
            break
        ep = collect_episode(env, learner, should_stop, deterministic=True)
        if ep is None:                       # reset 失敗/被中止/0 步 -> 停止評估
            break
        if ep.get("occluded"):               # 中途被遮擋 -> 不計、重跑（gate 已等到解除）
            print("  [EVAL] 中途被遮擋，重跑該場")
            continue
        res.append(ep["result"]); dmgs.append(ep["dmg"])
        hps.append(ep["end_hp"]); steps.append(ep["steps"])
    return res, dmgs, hps, steps


def eval_key(dmgs, res, hps, steps):
    """dreamer_best 比較鍵（tuple，高者勝）：(平均傷害, 勝場數, 勝場平均剩血, -勝場平均用時)。
    與 train_rl 同準則：avg 主鍵、同分依序比勝場多→剩血多→用時少；0 勝退化成只比 avg。"""
    avg_dmg = float(np.mean(dmgs))
    wins = sum(1 for r in res if r == "win")
    win_hp = [h for r, h in zip(res, hps) if r == "win" and h >= 0]
    win_st = [s for r, s in zip(res, steps) if r == "win"]
    avg_hp = float(np.mean(win_hp)) if win_hp else 0.0
    avg_st = float(np.mean(win_st)) if win_st else 0.0
    return (avg_dmg, wins, avg_hp, -avg_st)


def seed_from_demos(replay, n_files, real_reward=False, protect=False):
    """用 demo 暖機。real_reward=False（舊）：reward=0、進 FIFO，僅 world model 表徵/動態。
    real_reward=True：從 demo 遙測(boss_hp 正規化/player_hp raw 面具)重建**真實 reward**（含勝場 +28 終局）、
    在 boss 死那步截斷成終局、勝場進 protect 保護區——既灌 world model reward head 也供 latent-BC。
    對齊：act[t]=導致 obs_t 的動作=demo action[t-1]，act[0]=0；reward[t]=抵達 obs_t 的 reward。"""
    from obs import FrameStacker
    files = sorted(glob.glob(os.path.join("data", "*.npz")))[:n_files]
    n_win = 0
    for f in files:
        d = np.load(f)
        frames, actions = d["frames"], d["actions"]
        boss = d["boss_hp"] if real_reward and "boss_hp" in d else None
        player = d["player_hp"] if real_reward and "player_hp" in d else None
        N = len(frames)
        if boss is not None:                              # 勝場終局＝boss_hp 首次 <0（boss 消失）；截斷後段(boss 已死的閒置幀)
            neg = np.where(boss < 0)[0]
            N = (int(neg[0]) + 1) if len(neg) else N
        fs = FrameStacker()
        obs_buf, act_buf, rew_buf, cont_buf = [], [], [], []
        for i in range(N):
            fs.push(frames[i])
            obs_buf.append(np.concatenate(fs.buf, axis=0).astype(np.uint8))   # (C,H,W) uint8
            act_buf.append((np.zeros(config.N_ACTIONS_MODEL, np.float32) if i == 0
                            else config.dir_env_to_model(actions[i - 1]).astype(np.float32)))
            r = 0.0
            if boss is not None and i > 0:                # 複製 env._reward_and_done（scale=1、raw dmg=正規化Δ×900）
                if boss[i - 1] >= 0 and boss[i] >= 0:
                    r += config.RW_DMG * max(0.0, float(boss[i - 1] - boss[i])) * DEMO_BOSS_MAX
                if player is not None and player[i - 1] >= 0 and player[i] >= 0:
                    delta = float(player[i] - player[i - 1])
                    if delta < 0:
                        frac = max(0.0, min(1.0, float(player[i]) / DEMO_PLAYER_MAX))
                        r -= config.RW_HIT * (-delta) * (1.0 + config.RW_HIT_LATE_K * (1.0 - frac))
                    elif delta > 0:
                        r += config.RW_HEAL * delta
            rew_buf.append(r); cont_buf.append(1.0)
        if not obs_buf:
            continue
        cont_buf[-1] = 0.0                                # 終局
        won = boss is not None                            # demo 全是勝場
        if won:
            rew_buf[-1] += config.RW_WIN                  # 勝場 +28 終局（scale=1）
            n_win += 1
        replay.add_episode(obs_buf, act_buf, rew_buf, cont_buf, protect=(protect and won))
    if real_reward:
        log(f"demo seed（真實 reward）：{len(files)} 檔、{n_win} 勝進{'保護區' if protect else 'FIFO'}（含 +28 終局，供 WM reward head + latent-BC）")
    else:
        log(f"demo 暖機：seeded {len(files)} 場（reward=0，僅 world model 表徵/動態）")


def _env11_to_model13(a_env):
    """PPO 的 11 維 MultiBinary → 13 維模型動作。方向衝突規則（使用者指定）：同按上下→垂直「無」、
    同按左右→水平「無」（比 config.dir_env_to_model 的『上左優先』保守，衝突視為無方向）。其餘 7 鍵原樣。"""
    e = np.asarray(a_env, np.float32)
    up, down, left, right = e[0], e[1], e[2], e[3]
    v_up = up * (1.0 - down); v_down = down * (1.0 - up); v_none = 1.0 - v_up - v_down
    h_left = left * (1.0 - right); h_right = right * (1.0 - left); h_none = 1.0 - h_left - h_right
    return np.concatenate([[v_none, v_up, v_down], [h_none, h_left, h_right],
                           e[4:]]).astype(np.float32)


def seed_from_ppo(replay, ckpt_path, env, ctrl, device, target_wins, max_eps, deterministic=True):
    """用訓練好的 PPO policy 跑 n_eps 場真實對戰灌進 replay，**含真實 reward**（與 seed_from_demos 的
    reward=0 的關鍵差異）。目的＝打破 Dreamer 零勝死結：讓 reward head 見到勝場 +28 終局、world model
    學到勝利軌跡，想像才變得出「贏」。動作 11→13 用 _env11_to_model13；對齊鏡像 collect_episode
    （act[t]=導致 obs_t 的動作、reward/cont 抵達 obs_t）。**不設保護區**（FIFO 自然洗掉＝kickstart 後
    靠自己學，見對話討論）。回傳 (收集場數, 勝場數)。"""
    import curriculum
    from ac_model import ActorCritic
    eff = curriculum.effective_scale()               # guard：scale<1 會把勝場 +28 稀釋成 28*eff → 毀掉 seeding 目的
    if eff < 0.999:
        log(f"  [seed-ppo] ⚠ 中止 seeding：effective_scale={eff:.2f}<1，勝場 +28 會被稀釋成 {28.0 * eff:.1f}。"
            f" 請關 HKCurriculum mod / 刪 {config.CURRICULUM_SCALE_FILE} 後重跑（Dreamer 應全難度 scale=1）。")
        return 0, 0
    if not os.path.exists(ckpt_path):
        log(f"  [seed-ppo] 找不到 checkpoint {ckpt_path}，略過"); return 0, 0
    ac = ActorCritic().to(device)
    ac.load_compat(torch.load(ckpt_path, map_location=device)["model"]); ac.eval()
    A = config.N_ACTIONS_MODEL
    got, wins = 0, 0
    mode = "stochastic" if not deterministic else "deterministic"
    log(f"  [seed-ppo] 用 {ckpt_path} 收「{target_wins} 勝」（{mode}, 上限 {max_eps} 場, scale={eff:.2f}）...")
    while wins < target_wins and got < max_eps and not ctrl.stop:   # 收到目標勝場數才停（用勝數控制、非場數）
        ctrl.wait_while_paused(on_pause=env.act.release_all, log=log)   # seeding 期間也尊重 F9 暫停
        if ctrl.stop:
            break
        env.wait_until_unoccluded(ctrl.should_stop)
        if ctrl.stop:
            break
        obs, _ = env.reset(should_stop=ctrl.should_stop)
        if obs is None:
            break
        obs_buf = [_to_u8(obs)]; act_buf = [np.zeros(A, np.float32)]
        rew_buf = [0.0]; cont_buf = [1.0]
        boss = BossDamageTracker()
        done, occluded, info = False, False, {}
        while not done and not ctrl.stop:
            ot = torch.as_tensor(obs, dtype=torch.float32, device=device)
            a_env, _, _ = ac.act(ot, deterministic=deterministic)   # 11 維 uint8（actor 純看畫面，extra 免）
            obs2, r, term, trunc, info = env.step(a_env)
            if info.get("occluded"):
                env.drain_until_terminal(ctrl.should_stop); occluded = True; break
            done = term or trunc
            obs_buf.append(_to_u8(obs2)); act_buf.append(_env11_to_model13(a_env))
            rew_buf.append(float(r)); cont_buf.append(0.0 if term else 1.0)
            obs = obs2; boss.update(info)
        if occluded or len(rew_buf) <= 1 or info.get("tele_drop", 0.0) > 0.5:
            continue                                     # 遮擋/0 步/遙測太爛 → 不進 replay（同主迴圈品質閘）
        won = info.get("result") == "win"
        replay.add_episode(obs_buf, act_buf, rew_buf, cont_buf, protect=won)   # 勝場進不驅逐保護區
        got += 1
        wins += 1 if won else 0
        log(f"  [seed-ppo] {mode} {wins}/{target_wins}勝 (第{got}場) {str(info.get('result')):>5} "
            f"dmg={boss.dmg:4.0f} replay={len(replay)}{'  ★protected' if won else ''}")
    if wins < target_wins:
        log(f"  [seed-ppo] ⚠ 只收到 {wins}/{target_wins} 勝就達場數上限 {max_eps}（勝率太低？可提高 max_eps）")
    del ac; gc.collect()                                 # 放掉 PPO 模型（GPU/CPU），訓練不需要它
    if replay.n_protected() > 0:
        replay.save_protected(PPO_WINS)                  # 持久化 → 之後 resume 直接載入、免重跑 live PPO
        log(f"  [seed-ppo] 保護區共 {replay.n_protected()} 勝場已存 → {PPO_WINS}")
    log(f"  [seed-ppo] 完成：{mode} {got} 場、{wins} 勝進保護區（含真實 reward+勝場終局）")
    return got, wins


def save_ckpt(path, learner, ep_i, best_dmg, best_key=None):
    ck = {"learner": learner.state_dict(), "ep_i": ep_i, "best_dmg": best_dmg}
    if best_key is not None:
        ck["best_key"] = list(best_key)           # dreamer_best 比較鍵（avg,勝場,剩血,-用時）
    torch.save(ck, path)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=config.INPUT_BACKEND)
    ap.add_argument("--episodes", type=int, default=10_000)
    ap.add_argument("--train-steps", type=int, default=60, help="每場後做幾次想像訓練步")
    ap.add_argument("--replay-steps", type=int, default=80_000,
                    help="replay 容量(transition 數)。RGB 12ch×96²≈108KB/筆 → 80k≈8.6GB；"
                         "32GB 機器上限，過大會換頁(paging)造成系統級 tick 尖刺")
    ap.add_argument("--reward-scale", type=float, default=None,
                    help="Dreamer 內部 reward 放大倍率（全成分同乘；預設用 DreamerConfig.reward_scale=20）。"
                         "治 reward 太小→predictor 塌成常數+denom clamp；不動 config.RW_*")
    ap.add_argument("--actor-ent", type=float, default=None,
                    help="actor 熵係數（退火起點；預設 DreamerConfig.actor_ent=1e-4）。太高→de-commit、太低→過早 commit")
    ap.add_argument("--actor-ent-end", type=float, default=None,
                    help="熵退火終點：設了就從 --actor-ent 幾何退火到此值（早期探索→後期 commit，治固定熵係數不穩）。"
                         "不設=固定不退火。建議 start 3e-4 → end 3e-5")
    ap.add_argument("--actor-ent-anneal-eps", type=int, default=250,
                    help="熵退火跨幾場 ep 從起點降到終點（之後維持終點值，除非 --actor-ent-anneal-continue）")
    ap.add_argument("--actor-ent-anneal-continue", action="store_true",
                    help="過 anneal-eps 後不維持終點值，而是同幾何速率繼續下降（每 anneal-eps 再降 10×，floor 1e-6）")
    ap.add_argument("--protected-weight", type=int, default=2,
                    help="seed-ppo 勝場保護區的過抽倍數：sample 時勝場 episode 重複列入，讓想像更常從勝 posterior 起步"
                         "（re-commit 的 root driver）。1=不過抽；建議 2（>3 易過擬 18 場勝的低多樣性）")
    ap.add_argument("--eval-every", type=int, default=10, help="每幾場 eval 一次（0=關）")
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--seed-demos", type=int, default=0, help="開跑前用幾場 demo 暖機 world model")
    ap.add_argument("--seed-demos-reward", action="store_true",
                    help="demo 用**真實 reward**(從遙測重建、含 +28 勝場終局)灌進**保護區**(不驅逐)——既灌 WM reward head "
                         "也當 latent-BC 來源。取代 reward=0 暖機。demo 全勝場、比 PPO-seed 多樣")
    ap.add_argument("--bc-weight", type=float, default=0.0,
                    help="latent-BC 權重(起點)：>0 開啟，每步把 actor 拉向保護區 demo 勝場動作(治 reachability 牆)。"
                         "隨 --bc-anneal-eps 衰減到 0（早期強拉→後期放手讓 RL 超越 demo）")
    ap.add_argument("--bc-anneal-eps", type=int, default=250,
                    help="BC 權重從 --bc-weight 線性衰減到 0 跨幾場 ep（之後純 RL）")
    ap.add_argument("--bc-warmup-eps", type=int, default=20,
                    help="BC 前先讓 WM 暖機幾場（latent-BC 需 WM latent 有意義；這幾場 BC=0，之後才開始衰減）")
    ap.add_argument("--seed-ppo-wins-det", type=int, default=0,
                    help="用 PPO 灌『幾場 deterministic 勝』進保護區（含真實 reward+勝場終局，破零勝死結）。"
                         "用勝場數控制、非場數 → 保護區組成精準、不驟增驟減")
    ap.add_argument("--seed-ppo-wins-sto", type=int, default=0,
                    help="用 PPO 灌『幾場 stochastic(隨機) 勝』進保護區。隨機取樣→勝軌多樣、posterior 鋪滿"
                         "開局→終局各階段（補狀態可達性斷層）。與 --seed-ppo-wins-det 各自獨立控制")
    ap.add_argument("--seed-ppo-max-eps", type=int, default=150,
                    help="seed-ppo 每階段(det/sto)的場數安全上限（防勝率太低時無限跑）")
    ap.add_argument("--seed-ppo-ckpt", default=os.path.join("checkpoints_ppo", "rl_best_485_842.pt"),
                    help="seed-ppo 用的 PPO checkpoint 路徑")
    ap.add_argument("--seed-ppo-topup", action="store_true",
                    help="即使已載入保護區，也再 live 收（det+sto 目標勝數）併入既有保護區重存（累積更多勝場）")
    ap.add_argument("--resume", action="store_true", help="接續 dreamer_latest.pt")
    ap.add_argument("--ckpt", default=None, help="從指定 checkpoint 接續（檔名或路徑；優先於 --resume）")
    ap.add_argument("--snapshot-every", type=int, default=10,
                    help="每幾場存一個不覆蓋的編號快照 dreamer_eXXXXX.pt（0=關；過峰崩潰的回溯點）")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--length", type=int, default=None)
    return ap.parse_args()


def main():
    global _logf
    args = parse_args()
    # 即時迴圈防 GC 尖刺：關掉自動 GC，改在 episode 之間（大廳 idle 段）手動收（見下方 gc.collect()）。
    # 循環參照仍每場清一次，不漏記憶體；場內 15Hz 熱路徑保證零 GC 停頓。
    gc.disable()
    os.makedirs("logs", exist_ok=True)
    os.makedirs(config.CKPT_DIR, exist_ok=True)
    _logf = open(LOG_PATH, "a", encoding="utf-8")
    import csv
    is_new_csv = not os.path.exists(CSV_PATH)
    csvf = open(CSV_PATH, "a", newline="", encoding="utf-8")
    csvw = csv.DictWriter(csvf, fieldnames=CSV_COLS)
    if is_new_csv:
        csvw.writeheader(); csvf.flush()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 輸入形狀固定(obs 96×96、batch B×L 固定)→ 讓 cuDNN benchmark+快取最佳 conv plan：
    # 常能消掉 cuDNN v8「plan fallback」warning、順帶加速 world model 的 conv。
    torch.backends.cudnn.benchmark = True
    cfg = DreamerConfig()
    if args.batch:
        cfg.batch = args.batch
    if args.length:
        cfg.length = args.length
    if args.reward_scale is not None:
        cfg.reward_scale = args.reward_scale
    if args.actor_ent is not None:
        cfg.actor_ent = args.actor_ent
    ent_start = cfg.actor_ent                          # 熵退火起點（每場依 ep_i 幾何退火到 --actor-ent-end）
    ent_tag = (f"{ent_start}→{args.actor_ent_end}@{args.actor_ent_anneal_eps}ep"
               if args.actor_ent_end is not None else f"{ent_start}(固定)")
    log(f"===== {datetime.datetime.now():%Y-%m-%d %H:%M:%S} DreamerV3 訓練 "
        f"(device={device}, batch={cfg.batch}, length={cfg.length}, "
        f"train_steps/ep={args.train_steps}, replay_cap={args.replay_steps}, "
        f"reward_scale={cfg.reward_scale}, actor_ent={ent_tag}, "
        f"protected_weight={args.protected_weight}, "
        f"seed_ppo_wins=det{args.seed_ppo_wins_det}/sto{args.seed_ppo_wins_sto}) =====")

    seed_ppo_total = args.seed_ppo_wins_det + args.seed_ppo_wins_sto
    learner = DreamerLearner(cfg, config.NET_CHANNELS, config.N_ACTIONS_MODEL, device)
    replay = SequenceReplay(capacity_steps=args.replay_steps, protected_weight=args.protected_weight)
    if seed_ppo_total > 0 and os.path.exists(PPO_WINS):   # 勝場保護區持久化：resume 直接載回、免重跑 live PPO
        replay.load_protected(PPO_WINS)
        log(f"載入勝場保護區 {replay.n_protected()} 場 from {PPO_WINS}（不驅逐、免重跑 live seeding）")
    ep_i, best_dmg, best_key = 0, -1.0, (-1.0, 0, 0.0, 0.0)

    resume_path = None
    if args.ckpt:
        resume_path = args.ckpt if os.path.exists(args.ckpt) \
            else os.path.join(config.CKPT_DIR, args.ckpt)
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"找不到 checkpoint: {args.ckpt}")
    elif args.resume and os.path.exists(LATEST):
        resume_path = LATEST
    if resume_path:
        ck = torch.load(resume_path, map_location=device)
        learner.load_state_dict(ck["learner"]); ep_i = ck["ep_i"]; best_dmg = ck["best_dmg"]
        best_key = tuple(ck["best_key"]) if "best_key" in ck else (best_dmg, 0, 0.0, 0.0)
        log(f"接續 from {resume_path}：ep={ep_i} best_dmg={best_dmg:.0f}")
    if args.seed_demos > 0:
        seed_from_demos(replay, args.seed_demos, real_reward=args.seed_demos_reward,
                        protect=args.seed_demos_reward)

    # 長壽物件（模型、暖機 replay、resume 進來的狀態）都建好了 → 移進永久世代，GC 永不再掃它們。
    gc.freeze()

    ctrl = ControlKeys().start()
    print(f"{config.START_COUNTDOWN_SEC} 秒後開始，請點一下遊戲視窗取得焦點...（F10 停；F9 暫停）")
    for i in range(config.START_COUNTDOWN_SEC, 0, -1):
        print(f"  {i}..."); import time; time.sleep(1)

    env = HollowKnightEnv(backend=args.input)
    log(f"輸入後端：{args.input}")
    try:
        if seed_ppo_total > 0 and not ctrl.stop:      # 灌真勝場進保護區（勝場不驅逐；det + sto 各自目標勝數）
            if replay.n_protected() > 0 and not args.seed_ppo_topup:
                log(f"  [seed-ppo] 已有保護區勝場 {replay.n_protected()} 場（from disk）→ 略過 live seeding"
                    f"（要再補充加 --seed-ppo-topup）")
            else:                                    # 首次(無保護區) 或 --seed-ppo-topup(補充) → live 收 det 再 sto，併入保護區重存
                n0 = replay.n_protected()
                if args.seed_ppo_wins_det > 0 and not ctrl.stop:
                    seed_from_ppo(replay, args.seed_ppo_ckpt, env, ctrl, device,
                                  target_wins=args.seed_ppo_wins_det, max_eps=args.seed_ppo_max_eps, deterministic=True)
                if args.seed_ppo_wins_sto > 0 and not ctrl.stop:
                    seed_from_ppo(replay, args.seed_ppo_ckpt, env, ctrl, device,
                                  target_wins=args.seed_ppo_wins_sto, max_eps=args.seed_ppo_max_eps, deterministic=False)
                if n0 > 0:                           # topup：seed_from_ppo 已 append 新勝場並重存整個保護區
                    log(f"  [seed-ppo] 補充完成：保護區勝場 {n0} → {replay.n_protected()} 場（已併存）")
        while ep_i < args.episodes and not ctrl.stop:
            ctrl.wait_while_paused(on_pause=env.act.release_all, log=log)
            if ctrl.stop:
                break
            env.wait_until_unoccluded(ctrl.should_stop, log=log)   # 開場前 gate：被遮擋就等解除
            if ctrl.stop:
                break
            ep = collect_episode(env, learner, ctrl.should_stop)
            if ep is None:
                continue
            if ep.get("occluded"):                           # 中途被遮擋：不計、不進 replay
                ep_i += 1
                log(f"  EP{ep_i}: 中途被遮擋，不計")
                continue
            ep_i += 1
            if ep["drop"] <= 0.5:                            # 遙測太爛的場不進 replay
                replay.add_episode(ep["obs"], ep["act"], ep["rew"], ep["cont"])
            over = ep.get("over_runs", []); n_over = len(over)
            over_ms = 1000.0 * sum(s for _, s in over)
            fps = ep.get("fps", 0.0)
            # 只在「fps 掉(<14) 或 整場超支 > 一個 tick(66.7ms)」才 ⚠+印明細；忽略幾 ms 的良性 jitter
            slow = fps < config.TICK_HZ - 1 or over_ms > 1000.0 * config.TICK_DT
            fps_tag = f" fps={fps:4.1f}" + (f"⚠(over{n_over},{over_ms:.0f}ms)" if slow else "")
            act_tag = f" act={ep.get('act_ms_avg', 0):.0f}/{ep.get('act_ms_max', 0):.0f}ms"  # 平均/最大
            log(f"  EP{ep_i}: {str(ep['result']):>5} dmg={ep['dmg']:4.0f} "
                f"steps={ep['steps']}{fps_tag}{act_tag}  replay={len(replay)}")
            if slow:                                          # 真慢時才印各 over-tick 超時 ms（大到小）
                secs = ", ".join(f"{s * 1000:.1f}(t{idx}{',warmup' if idx == 0 else ''})"
                                 for idx, s in over[:OVER_PRINT_MAX])
                tail = f"  …(前{OVER_PRINT_MAX}/共{n_over})" if n_over > OVER_PRINT_MAX else ""
                log(f"    :  over ticks (ms)= {secs}{tail}  (合計 {over_ms:.0f}ms)")
                gms, gt = ep.get("max_grab", (0.0, -1)); tms, tt = ep.get("max_tele", (0.0, -1))
                log(f"    :  tick 拆解 max: grab={gms * 1000:.0f}ms(t{gt}) "
                    f"tele={tms * 1000:.0f}ms(t{tt}) act={ep.get('act_ms_max', 0):.0f}ms")

            # ---- 熵退火：actor_ent 從 ent_start 幾何退火到 --actor-ent-end（早期探索→後期 commit）----
            if args.actor_ent_end is not None:
                frac = ep_i / max(1, args.actor_ent_anneal_eps)
                if not args.actor_ent_anneal_continue:
                    frac = min(1.0, frac)                       # 預設：到終點就維持
                cfg.actor_ent = max(1e-6, ent_start * (args.actor_ent_end / ent_start) ** frac)  # learner.cfg 同物件、即時生效

            # ---- latent-BC 權重：WM 暖機 --bc-warmup-eps 場後才啟動，再從 --bc-weight 線性衰減到 0 ----
            #   (latent-BC 需 WM latent 有意義；暖機前 BC=0；暖機後早強拉向 demo→後期放手讓 RL 超越)
            bc_w = 0.0
            if args.bc_weight > 0 and ep_i >= args.bc_warmup_eps:
                frac = (ep_i - args.bc_warmup_eps) / max(1, args.bc_anneal_eps)
                bc_w = args.bc_weight * max(0.0, 1.0 - frac)

            # ---- 想像訓練（episode 之間做，不搶 15Hz）----
            stats = {}
            if replay.can_sample(cfg.length):
                accum = {}
                for _ in range(args.train_steps):
                    if ctrl.should_stop():
                        break
                    demo_b = replay.sample_protected(cfg.batch, cfg.length, device) if bc_w > 0 else None
                    st = learner.train_step(replay.sample(cfg.batch, cfg.length, device),
                                            demo_batch=demo_b, bc_weight=bc_w)
                    for k, v in st.items():
                        accum[k] = accum.get(k, 0.0) + v
                stats = {k: v / max(1, args.train_steps) for k, v in accum.items()}
                log(f"  [TRAIN ep{ep_i}] wm={stats['wm_loss']:.0f}(rec={stats['rec']:.0f}) "
                    f"actor={stats['actor_loss']:.3f} critic={stats['critic_loss']:.3f} "
                    f"dyn={stats['dyn']:.2f} rep={stats['rep']:.2f} imag_ret={stats['imag_ret']:.3f} "
                    f"ent={stats['ent']:.2f} aent={cfg.actor_ent:.1e}"
                    + (f" bc={stats.get('bc_loss', float('nan')):.3f}@{bc_w:.2f}" if bc_w > 0 else ""))
                # actor 崩壞判因：denom 黏 1.0＝return spread 被 clamp；adv 極小＝熵項相對主導；
                # rew_std≈0＝想像 reward 無信號(放大也是放大雜訊)。三者一起讀才判得出病因（見 memory）。
                log(f"    :  [diag] ret_denom={stats.get('ret_denom', float('nan')):.3f} "
                    f"adv|absmean/std|={stats.get('adv_absmean', float('nan')):.4f}/"
                    f"{stats.get('adv_std', float('nan')):.4f} "
                    f"imag_rew_std={stats.get('imag_rew_std', float('nan')):.4f}")

            # 手動 GC：在大廳 idle 段收一次（收集+訓練都做完了），把停頓丟在 15Hz 熱路徑之外。
            # 放這裡也順帶保護接下來 eval 的 collect（剛收完，場內不會再觸發）。
            gc.collect()

            # ---- eval ----
            ev = None
            if args.eval_every > 0 and ep_i % args.eval_every == 0:
                log(f"  --- eval（決定性 100%）{args.eval_episodes} 場 ---")
                res, dmgs, hps, steps_ev = eval_dreamer(env, learner, args.eval_episodes, ctrl.should_stop)
                if dmgs:
                    avg = float(np.mean(dmgs)); mx = float(np.max(dmgs))
                    wins = sum(1 for r in res if r == "win")
                    ev = (avg, mx, wins, len(dmgs))
                    log(f"  [EVAL] avg_dmg={avg:.0f} max={mx:.0f} wins={wins}/{len(dmgs)}")
                    key = eval_key(dmgs, res, hps, steps_ev)   # 字典序：avg>勝場>剩血>-用時
                    if key > best_key:
                        best_key, best_dmg = key, avg
                        save_ckpt(BEST, learner, ep_i, best_dmg, best_key)
                        log(f"  ★ 新最佳(eval) avg={avg:.0f} wins={wins}，存 dreamer_best.pt")

            save_ckpt(LATEST, learner, ep_i, best_dmg, best_key)
            if args.snapshot_every > 0 and ep_i % args.snapshot_every == 0:
                snap = os.path.join(config.CKPT_DIR, f"dreamer_e{ep_i:05d}.pt")
                save_ckpt(snap, learner, ep_i, best_dmg, best_key)
                log(f"  保存快照 {snap}")
            row = {"ep": ep_i, "train_steps": args.train_steps, "result": ep["result"],
                   "dmg": round(ep["dmg"], 1), "ep_steps": ep["steps"],
                   **{k: round(stats.get(k, float("nan")), 4)
                   for k in ["wm_loss", "rec", "rew", "cont", "dyn", "rep",
                             "actor_loss", "critic_loss", "imag_ret", "ent",
                             "ret_denom", "adv_absmean", "adv_std", "imag_rew_std"]}}
            if ev:
                row.update({"eval_avg_dmg": round(ev[0], 1), "eval_max_dmg": round(ev[1], 1),
                            "eval_wins": ev[2], "eval_eps": ev[3]})
            csvw.writerow(row); csvf.flush()
    finally:
        log("訓練結束，已存 dreamer_latest.pt")
        env.close(); ctrl.stop_listening()
        csvf.close(); _logf.close()


if __name__ == "__main__":
    main()
