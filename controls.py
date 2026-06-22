"""統一的控制鍵監聽。集中在這一個檔案定義「哪顆鍵代表什麼」，全專案共用，
避免同一顆鍵在不同檔案有不同意思或行為不一致的問題。

內建：
- F10 = 安全停止（.stop / .should_stop()）
- F9  = 暫停/繼續（.pause / .wait_while_paused()）
另可用 bind_edge() 註冊「邊緣觸發」鍵（按一下做一件事），例如 record.py 的
F7=開始錄、F8=存檔——由 take() 取走事件。

用法（訓練/評估/推論）：
    from controls import ControlKeys
    ctrl = ControlKeys().start()
    while not ctrl.stop:
        ctrl.wait_while_paused(on_pause=env.act.release_all)   # episode 之間
        if ctrl.stop:                       # 暫停中可能按了 F10
            break
        obs, _ = env.reset(should_stop=ctrl.should_stop)
        while not done and not ctrl.stop:
            ...

用法（錄製，邊緣觸發鍵）：
    ctrl = ControlKeys().bind_edge("f7", "start").bind_edge("f8", "save").start()
    while not ctrl.stop:                     # F10 = 離開
        if ctrl.take("start"): ...           # F7 按下一次
        if ctrl.take("save"):  ...           # F8 按下一次
"""
import time

from pynput import keyboard


class ControlKeys:
    """監聽 F10/F9 並維護 {"stop", "pause"} 狀態。一個 process 起一個即可。"""

    def __init__(self):
        self._state = {"stop": False, "pause": False}
        self._edges = {}      # keyboard.Key -> 名稱（邊緣觸發鍵）
        self._fired = {}      # 名稱 -> 是否有未取走的按下事件
        self._listener = keyboard.Listener(on_press=self._on_press)

    def _on_press(self, k):
        if k in self._edges:                 # 邊緣觸發鍵：記一筆，等 take() 取走
            self._fired[self._edges[k]] = True
            return
        if k == keyboard.Key.f10:
            self._state["stop"] = True
        elif k == keyboard.Key.f9:
            self._state["pause"] = not self._state["pause"]
            print("⏸ 收到暫停請求，本場結束後暫停。" if self._state["pause"]
                  else "▶ 取消暫停。")

    def start(self):
        """啟動背景監聽，回傳 self 方便鏈式呼叫。"""
        self._listener.start()
        return self

    def stop_listening(self):
        """停止背景監聽（通常不需要；listener 是 daemon，會隨 process 結束）。"""
        self._listener.stop()

    def bind_edge(self, key, name):
        """註冊一顆「邊緣觸發」鍵（按一下算一次事件，由 take(name) 取走）。

        key 可給 pynput 的 keyboard.Key，或字串如 "f7"（自動解析成 keyboard.Key.f7）。
        回傳 self 方便鏈式呼叫。
        """
        if isinstance(key, str):
            key = getattr(keyboard.Key, key)
        self._edges[key] = name
        self._fired.setdefault(name, False)
        return self

    def take(self, name) -> bool:
        """該邊緣鍵自上次取走後是否被按過；是的話回 True 並清掉旗標（消費一次）。"""
        if self._fired.get(name):
            self._fired[name] = False
            return True
        return False

    @property
    def stop(self) -> bool:
        return self._state["stop"]

    @property
    def pause(self) -> bool:
        return self._state["pause"]

    def should_stop(self) -> bool:
        """給 env.reset(should_stop=...) 之類用的 callable。"""
        return self._state["stop"]

    def wait_while_paused(self, on_pause=None, log=print) -> bool:
        """若正在暫停，阻塞到取消暫停或收到停止。

        on_pause: 進入暫停時呼叫一次（通常傳 env.act.release_all，放開輸入）。
        log:      印訊息用（train_rl 傳會同時寫檔的 log；其餘用預設 print）。
        回傳：是否真的暫停過（True 代表剛從暫停恢復，呼叫端可能要重設節拍）。
              回傳後請自行檢查 .stop——暫停中可能按了 F10。
        """
        if not self._state["pause"]:
            return False
        if on_pause is not None:
            on_pause()
        log("⏸ 已暫停。可在遊戲內檢查映射/難度/護符。再按 F9 繼續，F10 停止。")
        while self._state["pause"] and not self._state["stop"]:
            time.sleep(0.1)
        if not self._state["stop"]:
            log("▶ 繼續。")
        return True
