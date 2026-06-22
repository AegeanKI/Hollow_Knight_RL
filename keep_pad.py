"""保持虛擬 Xbox 手把連接，方便在 HidHide / 遊戲設定裡選到它。
按 Ctrl+C 結束。"""
import time

import vgamepad as vg

pad = vg.VX360Gamepad()
pad.update()
print("虛擬 Xbox 360 手把已連接。現在可以到 HidHide 的 Devices 分頁選它。")
print("按 Ctrl+C 結束。")
try:
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    print("\n已釋放手把。")
