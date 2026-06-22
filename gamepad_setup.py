"""建立虛擬手把並保持存在，讓 HK 偵測得到；並可手動送出單一按鈕以便在 HK 裡綁定控制。

為什麼需要它：虛擬手把只在本程式執行期間存在。先跑這支，HK 的手把設定才看得到手把。

綁定流程：
  1. 跑 `python gamepad_setup.py`（手把就出現了，保持這個視窗開著）。
  2. 進 HK：選項 → 手把設定，點某個動作的綁定格（HK 會等你按按鈕）。
  3. 切回這個視窗，輸入該動作的編號按 Enter → 倒數 3 秒（這段時間切回 HK 那個格子）
     → 它會送出對應按鈕 0.8 秒，HK 就綁到那顆鈕。
  4. 逐一綁完所有動作。輸入 q 離開。

對應（你在 HK 要把每個動作綁成右邊那顆）：
"""
import time

import config
from inputs import GamepadActuator

# (編號, 顯示名, Action實體鍵, 手把按鈕說明)
MENU = [
    ("1", "JUMP", "z", "A"),
    ("2", "ATTACK", "x", "X"),
    ("3", "DASH", "c", "B"),
    ("4", "CAST", "v", "Y"),
    ("5", "SUPER_DASH", "s", "LB 左肩"),
    ("6", "FOCUS", "a", "RB 右肩"),
    ("7", "DREAM_NAIL", "d", "RT 右扳機"),
    ("8", "UP", "up", "D-pad/左搖桿 上"),
    ("9", "DOWN", "down", "D-pad/左搖桿 下"),
    ("10", "LEFT", "left", "D-pad/左搖桿 左"),
    ("11", "RIGHT", "right", "D-pad/左搖桿 右"),
]
BYNUM = {num: (key, name) for num, name, key, _ in MENU}


def main():
    act = GamepadActuator()   # 建立虛擬手把並保持存在
    # 送一下中性狀態，幫助系統/遊戲偵測到手把
    act.release_all()
    print("虛擬手把已建立並保持中。HK 的手把設定現在應該偵測得到。\n")
    print("動作對應（HK 手把設定請綁成右側按鈕）：")
    for num, name, _key, btn in MENU:
        print(f"  {num:>2}. {name:<11} → {btn}")
    print("\n輸入編號 + Enter 送出該按鈕（會倒數 3 秒讓你切回 HK 綁定格），q 離開。")

    try:
        while True:
            s = input("送出哪個動作？編號(或 q)： ").strip().lower()
            if s in ("q", "quit", "exit"):
                break
            if s not in BYNUM:
                print("  無效編號"); continue
            key, name = BYNUM[s]
            print(f"  3 秒後送出 {name}，請切到 HK 並點選要綁定的格子...")
            for i in range(3, 0, -1):
                print(f"   {i}..."); time.sleep(1)
            act.apply_keys({key})
            time.sleep(0.8)
            act.apply_keys(set())
            print(f"  已送出 {name}（{key}）。")
    finally:
        act.release_all()
        print("離開，手把已釋放。")


if __name__ == "__main__":
    main()
