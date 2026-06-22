"""PPO 微調：從 BC 權重出發，在真實大黃蜂戰鬥中學習。

- 動作隨機取樣（探索），每 EPISODES_PER_UPDATE 場（預設 4）做一次 PPO 更新。
- 更新發生在 episode 之間（人在雕像大廳、不在戰鬥），不搶即時操作的 GPU。
- F10 = 安全停止（會存檔後結束）。不用 Esc，因為 HK 內看設定要按 Esc。
- F9 = 暫停/繼續（在 episode 之間生效，會放開所有輸入，方便檢查映射/難度/護符）。

Checkpoint（都在 checkpoints/，內含 模型+optimizer+更新次數+場數，接續無縫）：
  rl_latest.pt   每次更新滾動覆蓋（最新）
  rl_best.pt     平均傷害創新高才覆蓋（最佳）
  rl_uXXXX.pt    每 --snapshot-every 次更新存一個「不覆蓋」的編號快照（歷史回溯點）

用法：
  python train_rl.py                       # 全新：從 bc.pt 初始化開始
  python train_rl.py --resume              # 接續 rl_latest.pt
  python train_rl.py --ckpt rl_best.pt     # 從最佳那個接續
  python train_rl.py --ckpt rl_u0020.pt    # 從第 20 次更新的快照接續
  python train_rl.py --ckpt D:/some/dir/xxx.pt # 也吃完整路徑（--ckpt 優先於 --resume）
  python train_rl.py --snapshot-every 5    # 每 5 次更新存一個編號快照（0=關閉）
"""
import argparse
import os
import time

import numpy as np
import torch
from pynput import keyboard

import config
from ac_model import ActorCritic
from env import HollowKnightEnv
from ppo import RolloutBuffer, ppo_update

# 固定輸入尺寸 -> 讓 cudnn 選好演算法（也避開 CUDNN_STATUS_NOT_SUPPORTED 的 plan warning）
torch.backends.cudnn.benchmark = True

LATEST = os.path.join(config.CKPT_DIR, "rl_latest.pt")
BEST = os.path.join(config.CKPT_DIR, "rl_best.pt")
LOG_PATH = os.path.join("logs", "train_rl.log")


def save_ckpt(path, ac, opt, update_i, ep_i, best_dmg):
    torch.save({"model": ac.state_dict(), "opt": opt.state_dict(),
                "update_i": update_i, "ep_i": ep_i, "best_dmg": best_dmg}, path)


def run_eval(env, ac, device, n_eps, stop):
    """跑 n_eps 場決定性（不取樣）戰鬥，回傳 (results, damages)。"""
    res, dmgs = [], []
    for _ in range(n_eps):
        if stop["v"]:
            break
        obs = env.reset()
        boss0, last_boss, done = None, -1, False
        while not done and not stop["v"]:
            ot = torch.from_numpy(obs).to(device)
            a, _, _ = ac.act(ot, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            if boss0 is None and info["boss_hp_raw"] >= 0:
                boss0 = info["boss_hp_raw"]
            last_boss = info["boss_hp_raw"]
        res.append(info["result"])
        dmgs.append((boss0 - last_boss) if boss0 is not None else 0)
    return res, dmgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true", help="接續 rl_latest.pt")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="從指定 checkpoint 接續（路徑或 checkpoints/ 下的檔名），優先於 --resume")
    ap.add_argument("--snapshot-every", type=int, default=10,
                    help="每幾次更新存一個不覆蓋的編號快照 rl_uXXXX.pt（0=關閉）")
    ap.add_argument("--updates", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--ent-coef", type=float, default=0.0005, help="熵獎勵係數（太大策略會變隨機）")
    ap.add_argument("--episodes-per-update", type=int, default=config.EPISODES_PER_UPDATE,
                    help="每次更新收集幾場（越大梯度越穩）")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="每幾次更新跑一次決定性評估（0=關閉）")
    ap.add_argument("--eval-episodes", type=int, default=2)
    ap.add_argument("--input", choices=["keyboard", "gamepad"], default=config.INPUT_BACKEND,
                    help="輸入後端：keyboard(需焦點) 或 gamepad(虛擬手把，背景可、解放鍵盤)")
    args = ap.parse_args()

    os.makedirs("logs", exist_ok=True)
    logf = open(LOG_PATH, "a", encoding="utf-8")

    def log(msg):
        print(msg)
        logf.write(msg + "\n")
        logf.flush()

    log(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} 開始/接續訓練 "
        f"(ent_coef={args.ent_coef}, eps/update={args.episodes_per_update}) =====")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ac = ActorCritic().to(device)
    opt = torch.optim.Adam(ac.parameters(), lr=args.lr)
    update_i, ep_i, best_dmg = 0, 0, -1.0

    # 決定要從哪載入：--ckpt 指定 > --resume(latest) > 從 BC 初始化
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
        ac.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        update_i, ep_i, best_dmg = ck["update_i"], ck["ep_i"], ck["best_dmg"]
        log(f"接續訓練 from {resume_path}：update={update_i} ep={ep_i} best_dmg={best_dmg:.0f}")
    else:
        bc = torch.load(os.path.join(config.CKPT_DIR, "bc.pt"), map_location=device)
        ac.init_from_bc(bc["model"])
        log(f"從 BC 初始化 (macroF1 {bc['macroF1']:.3f})")

    stop = {"v": False}
    pause = {"v": False}

    def on_press(k):
        if k == keyboard.Key.f10:
            stop["v"] = True
        elif k == keyboard.Key.f9:
            pause["v"] = not pause["v"]
            print("⏸ 收到暫停請求，本場結束回大廳後暫停。" if pause["v"]
                  else "▶ 取消暫停。")

    keyboard.Listener(on_press=on_press).start()

    print("5 秒後開始，請點一下遊戲視窗取得焦點...（F10 安全停止；F9 暫停/繼續）")
    for i in range(5, 0, -1):
        print(f"  {i}..."); time.sleep(1)

    env = HollowKnightEnv(backend=args.input)
    log(f"輸入後端：{args.input}")
    try:
        while update_i < args.updates and not stop["v"]:
            buf = RolloutBuffer()
            ep_summ = []
            for _ in range(args.episodes_per_update):
                if stop["v"]:
                    break
                # 暫停點（episode 之間）：放開所有輸入，等使用者檢查遊戲設定後再續
                if pause["v"]:
                    env.act.release_all()
                    log("⏸ 訓練已暫停。可在遊戲內檢查映射/難度/護符。再按 F9 繼續，F10 停止。")
                    while pause["v"] and not stop["v"]:
                        time.sleep(0.1)
                    if not stop["v"]:
                        log("▶ 繼續訓練。")
                if stop["v"]:
                    break
                obs = env.reset()
                boss0, last_boss = None, -1
                done, ep_r, steps = False, 0.0, 0
                while not done and not stop["v"]:
                    ot = torch.from_numpy(obs).to(device)
                    action, logp, value = ac.act(ot)
                    obs2, r, term, trunc, info = env.step(action)
                    done = term or trunc
                    # 超時(truncation)非真終局：用最後狀態 value bootstrap，避免低估結尾
                    boot = ac.value(torch.from_numpy(obs2).to(device)) if (trunc and not term) else 0.0
                    buf.add(obs, action, logp, r, value, float(done), float(term), boot)
                    obs = obs2; ep_r += r; steps += 1
                    if boss0 is None and info["boss_hp_raw"] >= 0:
                        boss0 = info["boss_hp_raw"]
                    last_boss = info["boss_hp_raw"]
                ep_i += 1
                dmg = (boss0 - last_boss) if boss0 is not None else 0
                ep_summ.append((info["result"], dmg, ep_r, steps))
                log(f"  EP{ep_i}: {str(info['result']):>5} dmg={dmg:4.0f} "
                    f"reward={ep_r:6.2f} steps={steps}")

            if len(buf) == 0:
                break
            st = ppo_update(ac, opt, buf, device, ent_coef=args.ent_coef)
            update_i += 1
            avg_dmg = float(np.mean([d for _, d, _, _ in ep_summ]))
            wins = sum(1 for r, _, _, _ in ep_summ if r == "win")
            log(f"[UPDATE {update_i}] avg_dmg={avg_dmg:.0f} wins={wins}/{len(ep_summ)} "
                f"pi={st['pi_loss']:.3f} vf={st['vf_loss']:.3f} ent={st['entropy']:.2f} kl={st['kl']:.3f}")

            save_ckpt(LATEST, ac, opt, update_i, ep_i, best_dmg)
            if avg_dmg > best_dmg:
                best_dmg = avg_dmg
                save_ckpt(BEST, ac, opt, update_i, ep_i, best_dmg)
                log(f"  ★ 新最佳平均傷害 {best_dmg:.0f}，存 rl_best.pt")
            if args.snapshot_every > 0 and update_i % args.snapshot_every == 0:
                snap = os.path.join(config.CKPT_DIR, f"rl_u{update_i:04d}.pt")
                save_ckpt(snap, ac, opt, update_i, ep_i, best_dmg)
                log(f"  保存快照 {snap}")
            # 定期決定性評估
            if args.eval_every > 0 and update_i % args.eval_every == 0 and not stop["v"]:
                res, dmgs = run_eval(env, ac, device, args.eval_episodes, stop)
                ew = sum(1 for r in res if r == "win")
                log(f"  [EVAL] avg_dmg={np.mean(dmgs):.0f} wins={ew}/{len(res)} results={res}")
    finally:
        env.close()
        save_ckpt(LATEST, ac, opt, update_i, ep_i, best_dmg)
        log(f"已停止並存檔。update={update_i} ep={ep_i}")
        logf.close()


if __name__ == "__main__":
    main()
