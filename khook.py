"""背景鍵盤 hook：事件驅動地維護「目前按住哪些動作鍵」。

主錄製迴圈在每個 tick 來「取樣」目前按住的集合即可——對齊就是這樣達成的：
所有東西都在同一個 tick、同一個時間點取樣，而不是各自為政的時間軸。

額外保險：同時記下原始 press/release 事件（含時間戳），之後可柵格化到 tick 格子，
避免比一個 tick 還短的快速點按被漏掉。
"""
import threading
import time

from pynput import keyboard

from config import ACTION_KEYS, Action
from controls import ControlKeys

# 從 config 的動作鍵自動推導要監聽哪些鍵（新增/換鍵不用動這支）。
# 特殊鍵（方向鍵）走 pynput.Key；單字元鍵走 .char。
_DIRECTION_KEYS = {
    Action.UP.value: keyboard.Key.up,
    Action.DOWN.value: keyboard.Key.down,
    Action.LEFT.value: keyboard.Key.left,
    Action.RIGHT.value: keyboard.Key.right,
}
_SPECIAL = {_DIRECTION_KEYS[k]: k for k in ACTION_KEYS if k in _DIRECTION_KEYS}
_CHARS = {k for k in ACTION_KEYS if len(k) == 1}


def _name(key):
    if key in _SPECIAL:
        return _SPECIAL[key]
    try:
        ch = key.char
    except AttributeError:
        return None
    if ch is None:
        return None
    ch = ch.lower()
    return ch if ch in _CHARS else None


class KeyboardHook:
    def __init__(self):
        self._pressed = set()
        self._lock = threading.Lock()
        self.events = []  # (t, name, is_down) 原始事件流
        self._listener = None

    def start(self):
        self._listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

    def stop(self):
        if self._listener:
            self._listener.stop()

    def _on_press(self, key):
        name = _name(key)
        if name:
            with self._lock:
                self._pressed.add(name)
                self.events.append((time.perf_counter(), name, True))

    def _on_release(self, key):
        name = _name(key)
        if name:
            with self._lock:
                self._pressed.discard(name)
                self.events.append((time.perf_counter(), name, False))

    def sample(self) -> set:
        """回傳目前按住的動作鍵集合（複本）。"""
        with self._lock:
            return set(self._pressed)


if __name__ == "__main__":
    # 簡單自測：按鍵會即時印出目前按住的集合，F10 結束。
    hook = KeyboardHook()
    hook.start()
    ctrl = ControlKeys().start()
    print(f"監聽中，按 {ACTION_KEYS} 看反應，按 F10 結束。")
    try:
        last = None
        while not ctrl.stop:
            s = hook.sample()
            if s != last:
                print("按住:", sorted(s))
                last = s
            time.sleep(0.03)
    finally:
        hook.stop()
