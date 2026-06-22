"""讓訓練好的 BC 模型實際操作遊戲（純看畫面，不用遙測）。

復用錄製時的同一套元件：Capturer(擷取) + FrameStacker(疊幀) + Actuator(送鍵)，
以同樣的 15Hz 跑迴圈——訓練與推論的時序/前處理一致。

用法：
  1. 開遊戲，在神居雕像前選好大黃蜂、進競技場、等可以開打。
  2. 執行：  python play_bc.py
  3. 倒數 5 秒內點一下遊戲視窗讓它取得焦點，AI 就會開始打。
  F10 = 停止（會放開所有鍵）；F9 = 暫停/繼續（會放開輸入，可開 HK 選單檢查設定）。
"""
import argparse
import time

import numpy as np
import torch

import config
import autostart
from capture import Capturer
from controls import ControlKeys
from inputs import make_actuator
from model import PolicyNet
from obs import FrameStacker
from keys import vec_to_keys
from telemetry import TelemetryReceiver


def load_model(device):
    ckpt = torch.load(f"{config.CKPT_DIR}/{config.BC_CKPT}", map_location=device)
    model = PolicyNet().to(device).eval()
    model.load_state_dict(ckpt["model"])
    print(f"載入 {config.CKPT_DIR}/{config.BC_CKPT}  (epoch {ckpt['epoch']}, macroF1 {ckpt['macroF1']:.3f})")
    return model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.5, help="按鍵機率門檻")
    ap.add_argument("--no-autostart", action="store_true", help="跳過自動開場，手動進戰鬥")
    ap.add_argument("--input", choices=["keyboard", "gamepad"], default=config.INPUT_BACKEND)
    return ap.parse_args()


def main():
    args = parse_args()

    print(f"輸入後端: {args.input}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    cap = Capturer()
    stacker = FrameStacker()
    act = make_actuator(args.input)
    rx = TelemetryReceiver()
    rx.start()

    ctrl = ControlKeys().start()

    print(f"{config.START_COUNTDOWN_SEC} 秒後開始，請點一下遊戲視窗取得焦點...（F10 停止；F9 暫停/繼續）")
    for i in range(config.START_COUNTDOWN_SEC, 0, -1):
        print(f"  {i}..."); time.sleep(1)

    # 自動開場：開菜單 -> 選調諧級 -> 等開打
    if not args.no_autostart:
        if not autostart.start_challenge(cap, act, rx):
            act.release_all(); rx.stop()
            print("開場失敗，結束。請確認角色站在雕像上後重試。")
            return

    stacker.reset()
    monitor = autostart.EpisodeMonitor()
    dt = config.TICK_DT
    start = time.perf_counter()
    i = 0
    result = None
    print("AI 開打。")
    try:
        while not ctrl.stop:
            if ctrl.wait_while_paused(on_pause=act.release_all):
                if ctrl.stop:
                    break
                start = time.perf_counter() - i * dt  # 重設節拍基準，避免暫停後爆衝
            target = start + i * dt
            now = time.perf_counter()
            if now < target:
                time.sleep(target - now)

            raw = cap.grab_raw()
            stacker.push(raw)
            obs = torch.from_numpy(stacker.get()).to(device)
            vec = model.act(obs, threshold=args.threshold)
            act.apply_vec(vec)

            # 用遙測判斷本場是否結束
            tele, age = rx.sample()
            if tele is not None and age < config.TELE_FRESH_SEC:
                result = monitor.update(tele)
                if result is not None:
                    break

            i += 1
            if i % (config.TICK_HZ * 3) == 0:
                print("  按住:", sorted(vec_to_keys(vec)))
    finally:
        act.release_all()
        rx.stop()
        msg = {"win": "擊敗大黃蜂！🎉", "lose": "AI 死亡，本場結束。",
               "left": "已離開戰鬥場景。"}.get(result, "已停止。")
        print(msg + " 放開所有鍵。")


if __name__ == "__main__":
    main()
