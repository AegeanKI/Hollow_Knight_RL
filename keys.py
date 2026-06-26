"""動作編碼/解碼，以及視窗定位工具。"""
import ctypes
from ctypes import wintypes

import numpy as np

from config import ACTION_KEYS, KEY_INDEX, N_ACTIONS, WINDOW_TITLE


def keys_to_vec(pressed) -> np.ndarray:
    """一組「目前按住的鍵名」-> MultiBinary(11) 向量 (uint8)。"""
    vec = np.zeros(N_ACTIONS, dtype=np.uint8)
    for k in pressed:
        i = KEY_INDEX.get(k)
        if i is not None:
            vec[i] = 1
    return vec


def vec_to_keys(vec) -> set:
    """MultiBinary(11) 向量 -> 按住的鍵名集合。"""
    return {ACTION_KEYS[i] for i in range(N_ACTIONS) if vec[i]}


def format_action_row(vec) -> str:
    """把 MultiBinary(11) 排成定寬欄位診斷字串：按下顯示鍵名、沒按下換成等寬空白。
    欄位順序與標籤 = ACTION_KEYS（up down left right z x c v s a d）。
    例：只按 up 和 x → '[up                   x          ]'。
    用途：每 tick 印「模型實際輸出的向量」，對照畫面分辨「卡鍵」vs「模型真的持續輸出同方向」。"""
    cells = [ACTION_KEYS[i] if (i < len(vec) and vec[i]) else " " * len(ACTION_KEYS[i])
             for i in range(N_ACTIONS)]
    return "[" + " ".join(cells) + "]"


# ---- 視窗定位 (Win32 ctypes，免額外套件) -------------------------------------
user32 = ctypes.windll.user32


class _RECT(ctypes.Structure):
    _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG),
                ("right", wintypes.LONG), ("bottom", wintypes.LONG)]


class _POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


def find_window_region(title: str):
    """依標題找視窗，回傳其 client 區域在螢幕上的座標 (mss 格式)。
    找不到回傳 None。"""
    hwnd = user32.FindWindowW(None, title)
    if not hwnd:
        # 後備：模糊比對列舉所有頂層視窗
        hwnd = _find_window_contains(title)
    if not hwnd:
        return None

    rect = _RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    # client 左上角 (0,0) 轉成螢幕座標
    pt = _POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    w, h = rect.right - rect.left, rect.bottom - rect.top
    if w <= 0 or h <= 0:
        return None
    return {"left": pt.x, "top": pt.y, "width": w, "height": h}


def _find_window_contains(substr: str):
    """列舉視窗，回傳標題包含 substr 的第一個可見視窗 handle。"""
    substr = substr.lower()
    found = []

    EnumProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def cb(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        if substr in buf.value.lower():
            found.append(hwnd)
            return False
        return True

    user32.EnumWindows(EnumProc(cb), 0)
    return found[0] if found else None


if __name__ == "__main__":
    region = find_window_region(WINDOW_TITLE)
    print(f"視窗 '{WINDOW_TITLE}' 區域: {region}")
