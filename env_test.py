"""驗證 RL 環境：用 BC 模型驅動 env 連打數場，檢查自動重開/reward/勝敗判斷。

還沒有任何學習，只是把 env 的「reset→step→終止→reset」整圈在真實遊戲跑穩。

用法：
  python env_test.py --episodes 3
  1. 開遊戲、角色站在大黃蜂雕像前。
  2. 執行後 5 秒內點一下遊戲視窗取得焦點。
  F10 = 中止；F9 = 暫停/繼續（episode 之間生效，會放開輸入，可檢查設定）。
"""
import argparse
import time

import torch

import config
from controls import ControlKeys
from env import HollowKnightEnv
from model import PolicyNet


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--input", choices=["keyboard", "gamepad"], default=config.INPUT_BACKEND)
    return ap.parse_args()


def main():
    args = parse_args()

    print(f"輸入後端: {args.input}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(f"{config.CKPT_DIR}/bc.pt", map_location=device)
    model = PolicyNet().to(device).eval()
    model.load_state_dict(ckpt["model"])
    print(f"載入 BC 模型 (macroF1 {ckpt['macroF1']:.3f})")

    ctrl = ControlKeys().start()

    print("5 秒後開始，請點一下遊戲視窗取得焦點...（F10 中止；F9 暫停/繼續）")
    for i in range(5, 0, -1):
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
            total_r, steps, done, info = 0.0, 0, False, {}
            t0 = time.perf_counter()
            while not done and not ctrl.stop:
                obs_t = torch.from_numpy(obs).to(device)
                vec = model.act(obs_t, threshold=args.threshold)
                obs, r, term, trunc, info = env.step(vec)
                total_r += r; steps += 1
                done = term or trunc
            if steps == 0:                       # 剛 reset 完就被 stop -> 沒有有效資料
                continue
            dur = time.perf_counter() - t0
            print(f"[EP {ep}] result={info['result']} steps={steps} "
                  f"time={dur:.0f}s reward={total_r:.2f} "
                  f"(實測 {steps/max(dur,1e-3):.1f}Hz)")
    finally:
        env.close()
        print("環境已關閉。")


if __name__ == "__main__":
    main()
