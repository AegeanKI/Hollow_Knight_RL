"""DreamerV3 真實環境訓練迴圈（鏡像 train_rl.py，但 off-policy：收集→replay→想像訓練）。

與 train_rl.py 並存、互不影響：共用 env.py / controls / curriculum / config，但用
DreamerLearner（world model + 想像 actor-critic）。F10 安全停、F9 暫停。

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
import glob
import os

import numpy as np
import torch

import config
import curriculum
from controls import ControlKeys
from env import BossDamageTracker, HollowKnightEnv
from dreamer import DreamerLearner, SequenceReplay
from dreamer_model import DreamerConfig

LOG_PATH = os.path.join("logs", "train_dreamer.log")
CSV_PATH = os.path.join("logs", "dreamer_metrics.csv")
LATEST = os.path.join(config.CKPT_DIR, "dreamer_latest.pt")
BEST = os.path.join(config.CKPT_DIR, "dreamer_best.pt")
CSV_COLS = ["ep", "train_steps", "result", "dmg", "ep_steps", "scale",
            "wm_loss", "rec", "rew", "cont", "dyn", "rep",
            "actor_loss", "critic_loss", "imag_ret", "ent",
            "eval_avg_dmg", "eval_max_dmg", "eval_wins", "eval_eps"]

_logf = None


def log(msg):
    print(msg)
    if _logf:
        _logf.write(msg + "\n"); _logf.flush()


def _to_u8(obs):
    return np.clip(obs * 255.0, 0, 255).astype(np.uint8)


def collect_episode(env, learner, should_stop, deterministic=False):
    """跑一場，回傳對齊好的序列 + 統計。index t = (obs_t, 導致 obs_t 的動作, 抵達 reward, cont)。"""
    obs, _ = env.reset(should_stop=should_stop)
    if obs is None:
        return None
    scale = curriculum.read_scale()
    state, prev_a = learner.init_state()
    A = config.N_ACTIONS
    obs_buf = [_to_u8(obs)]                       # obs_0
    act_buf = [np.zeros(A, np.float32)]           # a_{-1}=0
    rew_buf = [0.0]; cont_buf = [1.0]
    boss = BossDamageTracker()
    done, steps, info = False, 0, {}
    while not done and not should_stop():
        a_np, state, prev_a = learner.act(obs, state, prev_a, deterministic)
        obs2, r, term, trunc, info = env.step(a_np)
        done = term or trunc
        obs_buf.append(_to_u8(obs2)); act_buf.append(a_np.astype(np.float32))
        rew_buf.append(float(r)); cont_buf.append(0.0 if term else 1.0)   # 真終局才 cont=0；截斷仍 1
        obs = obs2; steps += 1; boss.update(info)
    if steps == 0:
        return None
    return {"obs": obs_buf, "act": act_buf, "rew": rew_buf, "cont": cont_buf,
            "result": info.get("result"), "dmg": boss.dmg, "scale": scale,
            "steps": steps, "drop": info.get("tele_drop", 0.0)}


def eval_dreamer(env, learner, n_eps, should_stop):
    """決定性 eval（curriculum 強制 100%）。回傳 (avg_dmg, max_dmg, wins, n)。"""
    curriculum.begin_eval()
    dmgs, results = [], []
    try:
        for _ in range(n_eps):
            if should_stop():
                break
            ep = collect_episode(env, learner, should_stop, deterministic=True)
            if ep is None:
                continue
            dmgs.append(ep["dmg"]); results.append(ep["result"])
    finally:
        curriculum.end_eval()
    if not dmgs:
        return None
    wins = sum(1 for r in results if r == "win")
    return float(np.mean(dmgs)), float(np.max(dmgs)), wins, len(dmgs)


def seed_from_demos(replay, n_files):
    """用 demo 暖機 world model（僅表徵/動態；reward=0 不計，靠真實場後續校正）。
    對齊：act[t]=導致 obs_t 的動作=demo action[t-1]，act[0]=0。"""
    from obs import FrameStacker, preprocess_frame  # noqa: F401（preprocess 經 FrameStacker.push 用到）
    files = sorted(glob.glob(os.path.join("data", "*.npz")))[:n_files]
    for f in files:
        d = np.load(f)
        frames, actions = d["frames"], d["actions"]
        fs = FrameStacker()
        obs_buf, act_buf, rew_buf, cont_buf = [], [], [], []
        for i in range(len(frames)):
            fs.push(frames[i])
            obs_buf.append(np.concatenate(fs.buf, axis=0).astype(np.uint8))   # (C,H,W) uint8
            act_buf.append((np.zeros(config.N_ACTIONS, np.float32) if i == 0
                            else actions[i - 1].astype(np.float32)))
            rew_buf.append(0.0); cont_buf.append(1.0)
        if obs_buf:
            cont_buf[-1] = 0.0
            replay.add_episode(obs_buf, act_buf, rew_buf, cont_buf)
    log(f"demo 暖機：seeded {len(files)} 場（reward=0，僅 world model 表徵/動態）")


def save_ckpt(path, learner, ep_i, best_dmg):
    torch.save({"learner": learner.state_dict(), "ep_i": ep_i, "best_dmg": best_dmg}, path)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=config.INPUT_BACKEND)
    ap.add_argument("--episodes", type=int, default=10_000)
    ap.add_argument("--train-steps", type=int, default=60, help="每場後做幾次想像訓練步")
    ap.add_argument("--eval-every", type=int, default=10, help="每幾場 eval 一次（0=關）")
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--seed-demos", type=int, default=0, help="開跑前用幾場 demo 暖機 world model")
    ap.add_argument("--resume", action="store_true", help="接續 dreamer_latest.pt")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--length", type=int, default=None)
    return ap.parse_args()


def main():
    global _logf
    args = parse_args()
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
    cfg = DreamerConfig()
    if args.batch:
        cfg.batch = args.batch
    if args.length:
        cfg.length = args.length
    log(f"===== {datetime.datetime.now():%Y-%m-%d %H:%M:%S} DreamerV3 訓練 "
        f"(device={device}, batch={cfg.batch}, length={cfg.length}, "
        f"train_steps/ep={args.train_steps}) =====")

    learner = DreamerLearner(cfg, config.NET_CHANNELS, config.N_ACTIONS, device)
    replay = SequenceReplay()
    ep_i, best_dmg = 0, -1.0

    resume_path = LATEST if (args.resume and os.path.exists(LATEST)) else None
    if resume_path:
        ck = torch.load(resume_path, map_location=device)
        learner.load_state_dict(ck["learner"]); ep_i = ck["ep_i"]; best_dmg = ck["best_dmg"]
        log(f"接續 from {resume_path}：ep={ep_i} best_dmg={best_dmg:.0f}")
    if args.seed_demos > 0:
        seed_from_demos(replay, args.seed_demos)

    ctrl = ControlKeys().start()
    print(f"{config.START_COUNTDOWN_SEC} 秒後開始，請點一下遊戲視窗取得焦點...（F10 停；F9 暫停）")
    for i in range(config.START_COUNTDOWN_SEC, 0, -1):
        print(f"  {i}..."); import time; time.sleep(1)

    env = HollowKnightEnv(backend=args.input)
    log(f"輸入後端：{args.input}")
    try:
        while ep_i < args.episodes and not ctrl.stop:
            ctrl.wait_while_paused(on_pause=env.act.release_all, log=log)
            if ctrl.stop:
                break
            ep = collect_episode(env, learner, ctrl.should_stop)
            if ep is None:
                continue
            ep_i += 1
            if ep["drop"] <= 0.5:                            # 遙測太爛的場不進 replay
                replay.add_episode(ep["obs"], ep["act"], ep["rew"], ep["cont"])
            real = (ep["dmg"] * ep["scale"]) if ep["scale"] is not None else ep["dmg"]
            sc = f" scale={ep['scale']:.2f}" if ep["scale"] is not None else ""
            log(f"  EP{ep_i}: {str(ep['result']):>5} dmg={ep['dmg']:4.0f} ({real:.0f}) "
                f"steps={ep['steps']}{sc}  replay={len(replay)}")

            # ---- 想像訓練（episode 之間做，不搶 15Hz）----
            stats = {}
            if replay.can_sample(cfg.length):
                accum = {}
                for _ in range(args.train_steps):
                    if ctrl.should_stop():
                        break
                    st = learner.train_step(replay.sample(cfg.batch, cfg.length, device))
                    for k, v in st.items():
                        accum[k] = accum.get(k, 0.0) + v
                stats = {k: v / max(1, args.train_steps) for k, v in accum.items()}
                log(f"  [TRAIN ep{ep_i}] wm={stats['wm_loss']:.0f}(rec={stats['rec']:.0f}) "
                    f"actor={stats['actor_loss']:.3f} critic={stats['critic_loss']:.3f} "
                    f"dyn={stats['dyn']:.2f} rep={stats['rep']:.2f} imag_ret={stats['imag_ret']:.3f} "
                    f"ent={stats['ent']:.2f}")

            # ---- eval ----
            ev = None
            if args.eval_every > 0 and ep_i % args.eval_every == 0:
                log(f"  --- eval（決定性 100%）{args.eval_episodes} 場 ---")
                ev = eval_dreamer(env, learner, args.eval_episodes, ctrl.should_stop)
                if ev:
                    avg, mx, wins, n = ev
                    log(f"  [EVAL] avg_dmg={avg:.0f} max={mx:.0f} wins={wins}/{n}")
                    if avg > best_dmg:
                        best_dmg = avg; save_ckpt(BEST, learner, ep_i, best_dmg)
                        log(f"  ★ 新最佳(eval) {avg:.0f}，存 dreamer_best.pt")

            save_ckpt(LATEST, learner, ep_i, best_dmg)
            row = {"ep": ep_i, "train_steps": args.train_steps, "result": ep["result"],
                   "dmg": round(ep["dmg"], 1), "ep_steps": ep["steps"],
                   "scale": ep["scale"], **{k: round(stats.get(k, float("nan")), 4)
                   for k in ["wm_loss", "rec", "rew", "cont", "dyn", "rep",
                             "actor_loss", "critic_loss", "imag_ret", "ent"]}}
            if ev:
                row.update({"eval_avg_dmg": round(ev[0], 1), "eval_max_dmg": round(ev[1], 1),
                            "eval_wins": ev[2], "eval_eps": ev[3]})
            csvw.writerow(row); csvf.flush()
    finally:
        env.close(); ctrl.stop_listening()
        csvf.close(); _logf.close()
        log("訓練結束，已存 dreamer_latest.pt")


if __name__ == "__main__":
    main()
