"""評估某個 checkpoint 的真實實力（決定性出招，不取樣）。

用法：
  python eval.py                      # 評估 rl_best.pt，5 場
  python eval.py --latest             # 評估 rl_latest.pt
  python eval.py --ckpt rl_u0020.pt --episodes 8
  F10 = 中止；F9 = 暫停/繼續（episode 之間生效，會放開輸入，可檢查設定）
"""
import argparse
import os
import time

import numpy as np
import torch

import config
from ac_model import ActorCritic
from controls import ControlKeys
from env import BossDamageTracker, HollowKnightEnv


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None, help="checkpoints/ 下檔名或完整路徑")
    ap.add_argument("--latest", action="store_true", help="評估 rl_latest.pt")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--input", choices=["keyboard", "gamepad"], default=config.INPUT_BACKEND)
    return ap.parse_args()


def main():
    args = parse_args()

    if args.ckpt:
        path = args.ckpt if os.path.exists(args.ckpt) else os.path.join(config.CKPT_DIR, args.ckpt)
    else:
        path = os.path.join(config.CKPT_DIR, "rl_latest.pt" if args.latest else "rl_best.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(path, map_location=device)
    ac = ActorCritic().to(device).eval()
    ac.load_state_dict(ck["model"])
    print(f"評估 {path} (update {ck['update_i']}, ep {ck['ep_i']})")

    ctrl = ControlKeys().start()
    print("5 秒後開始，請點一下遊戲視窗取得焦點...（F10 中止；F9 暫停/繼續）")
    for i in range(5, 0, -1):
        print(f"  {i}..."); time.sleep(1)

    env = HollowKnightEnv(backend=args.input)
    dmgs, results = [], []
    try:
        for ep in range(1, args.episodes + 1):
            if ctrl.stop:
                break
            ctrl.wait_while_paused(on_pause=env.act.release_all)
            if ctrl.stop:
                break
            obs, _ = env.reset(should_stop=ctrl.should_stop)
            if obs is None:                      # 自動開場失敗/被中止 -> 跳過本場
                continue
            boss = BossDamageTracker()
            done, steps, info = False, 0, {}
            while not done and not ctrl.stop:
                ot = torch.from_numpy(obs).to(device)
                a, _, _ = ac.act(ot, deterministic=True)
                obs, r, term, trunc, info = env.step(a)
                done = term or trunc; steps += 1
                boss.update(info)
            if steps == 0:
                continue
            dmg = boss.dmg
            dmgs.append(dmg); results.append(info["result"])
            print(f"[EP {ep}] result={info['result']} dmg={dmg:.0f} "
                  f"boss剩={boss.last} steps={steps}")
    finally:
        env.close()

    if dmgs:
        wins = sum(1 for r in results if r == "win")
        print("\n===== 彙總 =====")
        print(f"場數: {len(dmgs)}  勝率: {wins}/{len(dmgs)}")
        print(f"平均傷害: {np.mean(dmgs):.0f}  最高傷害: {np.max(dmgs):.0f}  (boss 滿血約 900)")


if __name__ == "__main__":
    main()
