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


# ---- 畫面遮擋偵測 (T1 幾何重疊 + T2 secure desktop)，純 Win32 ctypes ----------
# 64-bit 安全：handle 回傳值要設成指標型別，否則會被截成 32-bit int（FindWindowW 的
# hwnd 在回呼裡是完整指標、若這裡被截斷，hwnd==hk 比較會永遠不相等而無法在 HK 處停手）。
user32.FindWindowW.restype = wintypes.HWND
user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
user32.OpenInputDesktop.restype = wintypes.HANDLE
user32.OpenInputDesktop.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(_RECT)]

try:
    _dwmapi = ctypes.windll.dwmapi
except OSError:
    _dwmapi = None

_DWMWA_CLOAKED = 14            # DwmGetWindowAttribute：!=0 表示視窗被 cloak（別的虛擬桌面/隱藏 UWP）
_GWL_EXSTYLE = -20
_WS_EX_TRANSPARENT = 0x00000020  # 點擊穿透：常駐遊戲 overlay（Discord/GeForce/FPS 計數）多為此，不算遮擋
_UOI_NAME = 2
_DESKTOP_READOBJECTS = 0x0001


def _is_cloaked(hwnd) -> bool:
    if _dwmapi is None:
        return False
    val = ctypes.c_int(0)
    hr = _dwmapi.DwmGetWindowAttribute(wintypes.HWND(hwnd), _DWMWA_CLOAKED,
                                       ctypes.byref(val), ctypes.sizeof(val))
    return hr == 0 and val.value != 0


def _window_title(hwnd) -> str:
    if not hwnd:
        return "未知視窗"
    n = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value or "未命名視窗"


def _input_desktop_name():
    """目前接收輸入的 desktop 名稱；開不起來回 None（多半就是 secure desktop）。"""
    h = user32.OpenInputDesktop(0, False, _DESKTOP_READOBJECTS)
    if not h:
        return None
    try:
        buf = ctypes.create_unicode_buffer(256)
        needed = wintypes.DWORD(0)
        ok = user32.GetUserObjectInformationW(h, _UOI_NAME, buf,
                                              ctypes.sizeof(buf), ctypes.byref(needed))
        return buf.value if ok else None
    finally:
        user32.CloseDesktop(h)


def _overlap_frac(region, r) -> float:
    """視窗矩形 r(_RECT, 螢幕座標) 覆蓋 region 的面積比例 [0,1]。"""
    rl, rt = region["left"], region["top"]
    rr, rb = rl + region["width"], rt + region["height"]
    ix = max(0, min(rr, r.right) - max(rl, r.left))
    iy = max(0, min(rb, r.bottom) - max(rt, r.top))
    area = region["width"] * region["height"]
    return (ix * iy) / area if area > 0 else 0.0


def is_region_occluded(region, min_cover_frac=0.02, title=WINDOW_TITLE):
    """擷取區 region 是否被遮擋。回傳 (occluded: bool, cover_frac: float, reason: str)。

    T2：輸入 desktop 名稱 != 'Default'（UAC secure desktop / 鎖定畫面）-> 遮擋。
        EnumWindows 只看得到自身 desktop，抓不到 secure desktop，故另用此檢查補洞。
    T1：z-order 疊在 HK 之上、與 region 重疊「最大的單一視窗」覆蓋率 >= 門檻 -> 遮擋。
        取最大單一視窗（不疊加，免重複計面積）：右鍵選單/分頁拖曳/截圖灰幕皆為單一全屏或大塊視窗。
        跳過：不可見/最小化/cloaked/點擊穿透(WS_EX_TRANSPARENT)/零面積。
    HK 視窗找不到（多半被最小化）也視為遮擋（看不到遊戲 = 該暫停）。
    """
    name = _input_desktop_name()
    if name is None:
        return True, 1.0, "secure desktop（輸入桌面開不起來，UAC?）"
    if name.lower() != "default":
        return True, 1.0, f"secure desktop（{name}）"

    hk = user32.FindWindowW(None, title) or _find_window_contains(title)
    if not hk:
        return True, 1.0, "HK 視窗找不到/最小化"

    best = {"frac": 0.0, "hwnd": None}

    EnumProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def cb(hwnd, _lparam):
        if hwnd == hk:
            return False                       # 列到 HK 為止：其後都在它下面，停止列舉（只看上方視窗）
        if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return True
        if user32.GetWindowLongW(hwnd, _GWL_EXSTYLE) & _WS_EX_TRANSPARENT:
            return True
        if _is_cloaked(hwnd):
            return True
        r = _RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(r)):
            return True
        frac = _overlap_frac(region, r)
        if frac > best["frac"]:
            best["frac"], best["hwnd"] = frac, hwnd
        return True

    user32.EnumWindows(EnumProc(cb), 0)
    if best["frac"] >= min_cover_frac:
        return True, best["frac"], _window_title(best["hwnd"])
    return False, best["frac"], ""


if __name__ == "__main__":
    region = find_window_region(WINDOW_TITLE)
    print(f"視窗 '{WINDOW_TITLE}' 區域: {region}")
    if region:
        occ, frac, reason = is_region_occluded(region)
        print(f"遮擋: {occ}  覆蓋={frac:.1%}  reason={reason!r}")
