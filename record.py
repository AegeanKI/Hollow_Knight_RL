"""階段 0 的核心：同步錄製 harness。

唯一的時鐘 = 固定 TICK_HZ 的主迴圈。每個 tick 在同一個時間點同時取樣：
  畫面(obs)、按住的鍵(action)、遙測(reward 資訊)
→ 對齊問題自動解決，因為三者共用同一個 tick。

這支程式之後拔掉「寫檔」、把「讀人類按鍵」換成「讀 AI 輸出」，就直接變成訓練/推論環境。

只圈出「戰鬥本身」——選單/選難度/等怪出現都不要錄（那些之後用腳本自動重開處理）。
控制鍵（不屬於動作鍵，不會被當成動作）：
  F7  = 這場戰鬥開始（大黃蜂出現、開打那一刻按）→ 開始錄
  F8  = 這場戰鬥結束（boss 死 或 你死 那一刻按）→ 存成一個 episode，回到待機
  F10 = 結束程式（若正在錄，會把這段也存起來）。不用 Esc，因為 HK 內看設定要按 Esc。

用法：
  python record.py            # 開始錄製，邊玩邊錄
"""
import json
import os
import time

import numpy as np
from pynput import keyboard

import config
from capture import Capturer
from keys import keys_to_vec
from khook import KeyboardHook
from telemetry import TelemetryReceiver


class _Control:
    """獨立的小 listener，處理 F7(開始) / F8(結束) / F10(離開)。"""
    def __init__(self):
        self.start_req = False
        self.stop_req = False
        self.quit = False
        self._l = keyboard.Listener(on_press=self._on)

    def start(self):
        self._l.start()

    def stop(self):
        self._l.stop()

    def _on(self, key):
        if key == keyboard.Key.f7:
            self.start_req = True
        elif key == keyboard.Key.f8:
            self.stop_req = True
        elif key == keyboard.Key.f10:
            self.quit = True


def _next_episode_path():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    n = 0
    while True:
        p = os.path.join(config.DATA_DIR, f"{config.EPISODE_PREFIX}{n:04d}.npz")
        if not os.path.exists(p):
            return p
        n += 1


def _save_episode(buf, events):
    if not buf["action"]:
        return None
    path = _next_episode_path()
    np.savez_compressed(
        path,
        frames=np.stack(buf["frame"]).astype(np.uint8),
        actions=np.stack(buf["action"]).astype(np.uint8),
        t=np.asarray(buf["t"], dtype=np.float32),
        player_hp=np.asarray(buf["player_hp"], dtype=np.float32),
        boss_hp=np.asarray(buf["boss_hp"], dtype=np.float32),
        tele_age=np.asarray(buf["tele_age"], dtype=np.float32),
    )
    # 原始按鍵事件另存 json（給之後柵格化／除錯用）
    with open(path.replace(".npz", "_events.json"), "w") as f:
        json.dump(events, f)
    n = len(buf["action"])
    print(f"  -> 存檔 {path}  ({n} ticks, {n / config.TICK_HZ:.1f}s)")
    return path


def _empty_buf():
    return {k: [] for k in ("frame", "action", "t", "player_hp", "boss_hp", "tele_age")}


def main():
    cap = Capturer()
    hook = KeyboardHook()
    rx = TelemetryReceiver()
    ctrl = _Control()
    hook.start()
    rx.start()
    ctrl.start()

    buf = _empty_buf()
    n_saved = 0
    recording = False   # 是否正在錄一場戰鬥（F7 開、F8 關）
    arm_t = 0.0         # 這場開始的絕對時間（給事件流過濾用）
    print(f"\n待機中 @ {config.TICK_HZ}Hz。F7=戰鬥開始  F8=戰鬥結束存檔  F10=離開")
    print("（先在雕像前選好難度，等大黃蜂出現要開打時再按 F7）\n")

    dt = config.TICK_DT
    start = time.perf_counter()
    i = 0
    slow_ticks = 0
    try:
        while not ctrl.quit:
            target = start + i * dt
            now = time.perf_counter()
            if now < target:
                time.sleep(target - now)
            elif recording and now - target > dt:
                slow_ticks += 1  # 來不及，落後超過一個 tick（只在錄製時在意）

            # ---- 處理開始/結束（邊緣觸發）----
            if ctrl.start_req:
                ctrl.start_req = False
                if not recording:
                    buf = _empty_buf()
                    recording = True
                    arm_t = time.perf_counter()
                    print("● 開始錄這場戰鬥（打完按 F8）")
            if ctrl.stop_req:
                ctrl.stop_req = False
                if recording:
                    recording = False
                    events = [e for e in hook.events if e[0] >= arm_t]
                    if _save_episode(buf, events):
                        n_saved += 1
                    print("○ 已存檔，回到待機（下一場選好難度再按 F7）")

            # 一律抓畫面以維持節拍；只有錄製中才寫進 buffer
            t_tick = time.perf_counter()
            frame = cap.grab_obs()
            pressed = hook.sample()
            tele, age = rx.sample()

            if recording:
                buf["frame"].append(frame)
                buf["action"].append(keys_to_vec(pressed))
                buf["t"].append(t_tick - start)
                buf["player_hp"].append(float(tele.get("player_hp", np.nan)) if tele else np.nan)
                buf["boss_hp"].append(float(tele.get("boss_hp", np.nan)) if tele else np.nan)
                buf["tele_age"].append(float(age) if tele else np.inf)
                n = len(buf["action"])
                if n % (config.TICK_HZ * 5) == 0:
                    print(f"  錄製中... {n / config.TICK_HZ:.0f}s, "
                          f"落後tick={slow_ticks}, tele={'有' if tele else '無'}")

            i += 1
    finally:
        # 收尾：若還在錄就把這段也存起來
        if recording and buf["action"]:
            events = [e for e in hook.events if e[0] >= arm_t]
            if _save_episode(buf, events):
                n_saved += 1
        hook.stop()
        rx.stop()
        ctrl.stop()
        print(f"\n結束。共存了 {n_saved} 個 episode 到 {config.DATA_DIR}/")
        if slow_ticks > i * 0.05:
            print(f"⚠ 有 {slow_ticks}/{i} 個 tick 來不及（擷取太慢）。"
                  f"可考慮調低 TICK_HZ 或縮小擷取區域。")


if __name__ == "__main__":
    main()
