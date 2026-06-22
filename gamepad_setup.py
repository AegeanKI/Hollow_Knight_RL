"""建立虛擬手把並保持存在，讓 HK 偵測得到；並可手動送出單一按鈕以便在 HK 裡綁定控制。

為什麼需要它：虛擬手把只在本程式執行期間存在。先跑這支，HK 的手把設定才看得到手把。

綁定流程：
  1. 跑 `python gamepad_setup.py`（手把就出現了，保持這個視窗開著）。
  2. 進 HK：選項 → 手把設定，點某個動作的綁定格（HK 會等你按按鈕）。
  3. 切回這個視窗，輸入該動作的編號按 Enter → 倒數 3 秒（這段時間切回 HK 那個格子）
     → 它會送出對應按鈕 0.8 秒，HK 就綁到那顆鈕。
  4. 逐一綁完所有動作。輸入 q 離開。

動作清單與對應按鈕都從 config.Action + GamepadActuator 的實際對應自動產生，不再手寫。
"""
import time

import config
from inputs import GamepadActuator

# XUSB 按鈕 -> 人類可讀標籤（純顯示用；按鈕識別碼取自 vgamepad enum 的 .name）
_BUTTON_LABELS = {
    "XUSB_GAMEPAD_A": "A", "XUSB_GAMEPAD_B": "B",
    "XUSB_GAMEPAD_X": "X", "XUSB_GAMEPAD_Y": "Y",
    "XUSB_GAMEPAD_LEFT_SHOULDER": "LB 左肩",
    "XUSB_GAMEPAD_RIGHT_SHOULDER": "RB 右肩",
    "XUSB_GAMEPAD_DPAD_UP": "D-pad/左搖桿 上",
    "XUSB_GAMEPAD_DPAD_DOWN": "D-pad/左搖桿 下",
    "XUSB_GAMEPAD_DPAD_LEFT": "D-pad/左搖桿 左",
    "XUSB_GAMEPAD_DPAD_RIGHT": "D-pad/左搖桿 右",
}
_TRIGGER_LABELS = {"left": "LT 左扳機", "right": "RT 右扳機"}


def _button_label(act, action):
    """依 GamepadActuator 實際的對應，推出某動作要綁到哪顆手把鈕（人類可讀）。"""
    key = action.value
    if key in act.trigger_map:
        return _TRIGGER_LABELS.get(act.trigger_map[key], act.trigger_map[key])
    btn = act.button_map.get(key)
    if btn is None:
        return "(未綁定)"
    return _BUTTON_LABELS.get(btn.name, btn.name)


def main():
    act = GamepadActuator()   # 建立虛擬手把並保持存在
    act.release_all()         # 送中性狀態，幫助系統/遊戲偵測到手把
    print("虛擬手把已建立並保持中。HK 的手把設定現在應該偵測得到。\n")

    actions = config.ACTIONS                       # 依 config 定義順序
    bynum = {str(i + 1): a for i, a in enumerate(actions)}
    print("動作對應（HK 手把設定請綁成右側按鈕）：")
    for i, a in enumerate(actions):
        print(f"  {i + 1:>2}. {a.name:<11} → {_button_label(act, a)}")
    print("\n輸入編號 + Enter 送出該按鈕（會倒數 3 秒讓你切回 HK 綁定格），q 離開。")

    try:
        while True:
            s = input("送出哪個動作？編號(或 q)： ").strip().lower()
            if s in ("q", "quit", "exit"):
                break
            a = bynum.get(s)
            if a is None:
                print("  無效編號"); continue
            print(f"  3 秒後送出 {a.name}，請切到 HK 並點選要綁定的格子...")
            for i in range(3, 0, -1):
                print(f"   {i}..."); time.sleep(1)
            act.apply_keys({a.value})
            time.sleep(0.8)
            act.apply_keys(set())
            print(f"  已送出 {a.name}（{a.value}）。")
    finally:
        act.release_all()
        print("離開，手把已釋放。")


if __name__ == "__main__":
    main()
