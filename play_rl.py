"""看 RL 訓練後的模型實際打大黃蜂（不學習，決定性出招）。

用法：
  python play_rl.py                 # 載 rl_best.pt
  python play_rl.py --latest        # 載 rl_latest.pt
  python play_rl.py --episodes 3
  F10 = 中止；F9 = 暫停/繼續（episode 之間生效，會放開輸入，可檢查設定）
"""
import argparse
import os
import time

import torch

import config
from ac_model import ActorCritic
from controls import ControlKeys
from env import HollowKnightEnv


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--latest", action="store_true", help="載 rl_latest 而非 rl_best")
    ap.add_argument("--input", choices=["keyboard", "gamepad"], default=config.INPUT_BACKEND)
    return ap.parse_args()


def main():
    args = parse_args()

    print(f"輸入後端: {args.input}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = os.path.join(config.CKPT_DIR,
                        config.RL_LATEST_CKPT if args.latest else config.RL_BEST_CKPT)
    ck = torch.load(path, map_location=device)
    ac = ActorCritic().to(device).eval()
    ac.load_state_dict(ck["model"])
    print(f"載入 {path} (update {ck['update_i']}, best_dmg {ck['best_dmg']:.0f})")

    ctrl = ControlKeys().start()

    print(f"{config.START_COUNTDOWN_SEC} 秒後開始，請點一下遊戲視窗取得焦點...（F10 中止；F9 暫停/繼續）")
    for i in range(config.START_COUNTDOWN_SEC, 0, -1):
        print(f"  {i}..."); time.sleep(1)

    env = HollowKnightEnv(backend=args.input)
    try:
        for ep in range(1, args.episodes + 1):
            if ctrl.stop:
                break
            ctrl.wait_while_paused(on_pause=env.act.release_all)
            if ctrl.stop:
                break
            obs, _ = env.reset(should_stop=ctrl.should_stop)
            if obs is None:                      # 自動開場失敗/被中止
                continue
            done, steps, info = False, 0, {}
            while not done and not ctrl.stop:
                ot = torch.from_numpy(obs).to(device)
                action, _, _ = ac.act(ot, deterministic=True)
                obs, r, term, trunc, info = env.step(action)
                done = term or trunc; steps += 1
            if steps == 0:                       # 剛 reset 完就被 stop -> 沒有有效資料
                continue
            print(f"[EP {ep}] result={info['result']} steps={steps}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
