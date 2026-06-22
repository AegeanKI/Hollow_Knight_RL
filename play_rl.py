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
from pynput import keyboard

import config
from ac_model import ActorCritic
from env import HollowKnightEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--latest", action="store_true", help="載 rl_latest 而非 rl_best")
    ap.add_argument("--input", choices=["keyboard", "gamepad"], default=config.INPUT_BACKEND)
    args = ap.parse_args()

    print(f"輸入後端: {args.input}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = os.path.join(config.CKPT_DIR, "rl_latest.pt" if args.latest else "rl_best.pt")
    ck = torch.load(path, map_location=device)
    ac = ActorCritic().to(device).eval()
    ac.load_state_dict(ck["model"])
    print(f"載入 {path} (update {ck['update_i']}, best_dmg {ck['best_dmg']:.0f})")

    stop = {"v": False}
    pause = {"v": False}

    def on_press(k):
        if k == keyboard.Key.f10:
            stop["v"] = True
        elif k == keyboard.Key.f9:
            pause["v"] = not pause["v"]
            print("⏸ 收到暫停請求，本場結束後暫停。" if pause["v"] else "▶ 取消暫停。")

    keyboard.Listener(on_press=on_press).start()

    print("5 秒後開始，請點一下遊戲視窗取得焦點...（F10 中止；F9 暫停/繼續）")
    for i in range(5, 0, -1):
        print(f"  {i}..."); time.sleep(1)

    env = HollowKnightEnv(backend=args.input)
    try:
        for ep in range(1, args.episodes + 1):
            if stop["v"]:
                break
            if pause["v"]:
                env.act.release_all()
                print("⏸ 已暫停。可開 HK 選單檢查映射/難度/護符。再按 F9 繼續，F10 停止。")
                while pause["v"] and not stop["v"]:
                    time.sleep(0.1)
                if not stop["v"]:
                    print("▶ 繼續。")
            if stop["v"]:
                break
            obs = env.reset(should_stop=lambda: stop["v"])
            if obs is None:                      # 自動開場失敗/被中止
                continue
            done, steps = False, 0
            while not done and not stop["v"]:
                ot = torch.from_numpy(obs).to(device)
                action, _, _ = ac.act(ot, deterministic=True)
                obs, r, term, trunc, info = env.step(action)
                done = term or trunc; steps += 1
            print(f"[EP {ep}] result={info['result']} steps={steps}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
