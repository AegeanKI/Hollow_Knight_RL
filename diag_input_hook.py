"""診斷：到底是誰把虛擬手把翻成鍵盤事件、又是哪個鍵關掉 PowerShell。

原理：裝一個全域低階鍵盤 hook (WH_KEYBOARD_LL)，記錄每個「被注入(injected)」的
按鍵事件——我們的 vgamepad 只送手把、不送鍵盤，所以驅動手把期間若出現 injected
鍵盤事件，就證明中間有轉譯層（Steam Input 等），且能看到它翻成什麼鍵。

事件即時 flush 到 diag_input_hook.log，就算這個視窗被關掉，紀錄也留得住。

用法：
  1. 開一個「可丟棄」的 Notepad 視窗（不要用 PowerShell 當焦點，免得它被關）。
  2. 在 PowerShell 跑：  python diag_input_hook.py
  3. 看到「開始驅動手把」後的 2 秒內，點一下那個 Notepad 讓它取得焦點。
  4. 腳本會輪流壓 右/左/上/下 各 1.5s，全程記錄注入鍵。
  5. 結束後把 diag_input_hook.log 內容貼給我。
"""
import ctypes
import ctypes.wintypes as wt
import threading
import time

user32 = ctypes.windll.user32
LOG = open("diag_input_hook.log", "w", encoding="utf-8")

# 64-bit 指標寬度的型別（LRESULT/LPARAM = LONG_PTR, WPARAM = UINT_PTR）
LRESULT = ctypes.c_ssize_t
LPARAM = ctypes.c_ssize_t
WPARAM = ctypes.c_size_t

WH_KEYBOARD_LL = 13
WM_KEYDOWN, WM_SYSKEYDOWN = 0x0100, 0x0104
WM_KEYUP, WM_SYSKEYUP = 0x0101, 0x0105
LLKHF_INJECTED = 0x10
LLKHF_LOWER_IL_INJECTED = 0x02

VK_NAMES = {
    0x25: "LEFT", 0x26: "UP", 0x27: "RIGHT", 0x28: "DOWN",
    0x1B: "ESC", 0x0D: "ENTER", 0x09: "TAB", 0x20: "SPACE",
    0x12: "ALT", 0x11: "CTRL", 0x10: "SHIFT", 0x5B: "LWIN", 0x5C: "RWIN",
    0x73: "F4", 0x7A: "F11", 0x7B: "F12", 0x08: "BACKSPACE", 0x2E: "DELETE",
}
for c in range(0x41, 0x5B):
    VK_NAMES[c] = chr(c)          # A-Z
for c in range(0x30, 0x3A):
    VK_NAMES[c] = chr(c)          # 0-9


def vk_name(vk):
    return VK_NAMES.get(vk, f"VK_0x{vk:02X}")


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("vkCode", wt.DWORD), ("scanCode", wt.DWORD),
                ("flags", wt.DWORD), ("time", wt.DWORD),
                ("dwExtraInfo", ctypes.POINTER(wt.ULONG))]


HOOKPROC = ctypes.CFUNCTYPE(LRESULT, ctypes.c_int, WPARAM, LPARAM)

# 明確宣告型別，否則 64-bit 下指標被當 c_int 會 OverflowError
user32.SetWindowsHookExW.argtypes = [ctypes.c_int, HOOKPROC, wt.HINSTANCE, wt.DWORD]
user32.SetWindowsHookExW.restype = wt.HHOOK
user32.CallNextHookEx.argtypes = [wt.HHOOK, ctypes.c_int, WPARAM, LPARAM]
user32.CallNextHookEx.restype = LRESULT
user32.UnhookWindowsHookEx.argtypes = [wt.HHOOK]
user32.UnhookWindowsHookEx.restype = wt.BOOL

t0 = time.perf_counter()


def emit(msg):
    print(msg)
    LOG.write(msg + "\n")
    LOG.flush()


def low_level(nCode, wParam, lParam):
    if nCode == 0 and wParam in (WM_KEYDOWN, WM_SYSKEYDOWN, WM_KEYUP, WM_SYSKEYUP):
        kb = KBDLLHOOKSTRUCT.from_address(lParam & 0xFFFFFFFFFFFFFFFF)
        injected = bool(kb.flags & (LLKHF_INJECTED | LLKHF_LOWER_IL_INJECTED))
        down = wParam in (WM_KEYDOWN, WM_SYSKEYDOWN)
        if injected and down:               # 只關心「被注入的按下」
            mods = []
            if user32.GetAsyncKeyState(0x11) & 0x8000: mods.append("CTRL")
            if user32.GetAsyncKeyState(0x12) & 0x8000: mods.append("ALT")
            if user32.GetAsyncKeyState(0x10) & 0x8000: mods.append("SHIFT")
            if user32.GetAsyncKeyState(0x5B) & 0x8000: mods.append("WIN")
            combo = "+".join(mods + [vk_name(kb.vkCode)])
            emit(f"[{time.perf_counter()-t0:6.2f}s] INJECTED KEYDOWN  {combo:20s} "
                 f"(vk=0x{kb.vkCode:02X} flags=0x{kb.flags:02X})")
    return user32.CallNextHookEx(None, nCode, wParam, lParam)


def drive_gamepad():
    try:
        import vgamepad as vg
    except ImportError:
        emit("!! 沒有 vgamepad，無法驅動手把"); return
    pad = vg.VX360Gamepad()
    B = vg.XUSB_BUTTON
    dirs = [("RIGHT", B.XUSB_GAMEPAD_DPAD_RIGHT), ("LEFT", B.XUSB_GAMEPAD_DPAD_LEFT),
            ("UP", B.XUSB_GAMEPAD_DPAD_UP), ("DOWN", B.XUSB_GAMEPAD_DPAD_DOWN)]
    emit(">> 開始驅動手把（2 秒內把焦點切到丟棄用的 Notepad）")
    time.sleep(2)
    for name, d in dirs:
        emit(f">> 壓住 D-pad {name} 1.5s + 左搖桿滿舵")
        pad.reset()
        pad.press_button(button=d)
        sx = 32767 if name == "RIGHT" else (-32767 if name == "LEFT" else 0)
        sy = 32767 if name == "UP" else (-32767 if name == "DOWN" else 0)
        pad.left_joystick(x_value=sx, y_value=sy)
        pad.update()
        time.sleep(1.5)
        pad.reset(); pad.update()
        time.sleep(0.4)
    emit(">> 測試手把『斷線』(模擬程式結束):")
    del pad
    time.sleep(1.0)
    emit(">> 驅動結束")


def main():
    ptr = HOOKPROC(low_level)
    hook = user32.SetWindowsHookExW(WH_KEYBOARD_LL, ptr, None, 0)
    if not hook:
        emit("!! 安裝 hook 失敗"); return
    emit("hook 安裝成功，開始監聽注入鍵盤事件...")

    worker = threading.Thread(target=drive_gamepad, daemon=True)
    worker.start()

    msg = wt.MSG()
    end = time.perf_counter() + 14
    while time.perf_counter() < end:
        if user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
        time.sleep(0.003)

    user32.UnhookWindowsHookEx(hook)
    emit("== 監聽結束 ==")
    emit("若上面完全沒有 INJECTED KEYDOWN：代表沒有鍵盤轉譯層，問題不是鍵盤注入。")
    emit("若有：那些就是兇手翻出來的鍵，看哪個會關視窗。")
    LOG.close()


if __name__ == "__main__":
    main()
