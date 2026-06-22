"""輸入注入：把「目前要按住哪些鍵」送進遊戲。

用 pydirectinput（底層 SendInput + scan code），比一般虛擬鍵更容易被遊戲收到。
Actuator 維護「目前實際按住」的狀態，每個 tick 只送出差異（新按下/放開），
這正是訓練時 env.step(action) 要做的事——錄製和推論共用同一套執行器。
"""
import pydirectinput

from config import Action
from keys import vec_to_keys

# 低延遲設定：拿掉每次操作之間的內建延遲與 failsafe。
pydirectinput.PAUSE = 0.0
pydirectinput.FAILSAFE = False


class Actuator:
    def __init__(self):
        self._held = set()  # 目前實際按住的鍵名

    def apply_keys(self, target: set):
        """讓實際按住的鍵 == target。只送出差異。"""
        to_press = target - self._held
        to_release = self._held - target
        for k in to_release:
            pydirectinput.keyUp(k)
        for k in to_press:
            pydirectinput.keyDown(k)
        self._held = set(target)

    def apply_vec(self, vec):
        """吃 MultiBinary(11) 向量。"""
        self.apply_keys(vec_to_keys(vec))

    def release_all(self):
        for k in list(self._held):
            pydirectinput.keyUp(k)
        self._held.clear()


class GamepadActuator:
    """用虛擬 Xbox 手把送輸入。XInput 是全域輪詢，不需視窗焦點，
    搭配 mod 的 Application.runInBackground 即可在背景操作、解放你的鍵盤。
    介面與 Actuator 相同（apply_keys/apply_vec/release_all），可無痛互換。"""

    def __init__(self):
        try:
            import vgamepad as vg
        except ImportError as e:
            raise ImportError("需要 vgamepad：請先 `pip install vgamepad`"
                              "（會一併安裝 ViGEmBus 驅動，過程會跳 UAC）") from e
        self.pad = vg.VX360Gamepad()
        B = vg.XUSB_BUTTON
        # 動作(Action) -> 手把按鈕。用 Action enum 當 key，確保與 config 的動作定義同步
        # （遊戲內手把控制需把每個動作綁成這裡對應的按鈕）。
        action_to_button = {
            Action.UP: B.XUSB_GAMEPAD_DPAD_UP,
            Action.DOWN: B.XUSB_GAMEPAD_DPAD_DOWN,
            Action.LEFT: B.XUSB_GAMEPAD_DPAD_LEFT,
            Action.RIGHT: B.XUSB_GAMEPAD_DPAD_RIGHT,
            Action.JUMP: B.XUSB_GAMEPAD_A,
            Action.ATTACK: B.XUSB_GAMEPAD_X,
            Action.DASH: B.XUSB_GAMEPAD_B,
            Action.CAST: B.XUSB_GAMEPAD_Y,
            Action.SUPER_DASH: B.XUSB_GAMEPAD_LEFT_SHOULDER,   # LB
            Action.FOCUS: B.XUSB_GAMEPAD_RIGHT_SHOULDER,       # RB
        }
        # apply_keys 進來的是實體鍵字串集合(Action.value)，故轉成字串為 key
        self.button_map = {a.value: btn for a, btn in action_to_button.items()}
        # 扳機是「類比軸」不是按鈕，XUSB_BUTTON 沒有扳機常數；改用 pad.left_trigger()/
        # right_trigger() 驅動，故這裡用 "left"/"right" 字串選哪一邊。
        # （HK 不接受搖桿按下 R3，DREAM_NAIL 改綁 RT 右扳機。）
        self.trigger_map = {Action.DREAM_NAIL.value: "right"}
        self._held = set()

    def apply_keys(self, target):
        self.pad.reset()                       # 每 tick 重設成當前完整狀態
        for k in target:
            btn = self.button_map.get(k)
            if btn is not None:
                self.pad.press_button(button=btn)
        # 扳機（滿舵 255 / 放開 0）
        self.pad.left_trigger(value=255 if any(self.trigger_map.get(k) == "left" for k in target) else 0)
        self.pad.right_trigger(value=255 if any(self.trigger_map.get(k) == "right" for k in target) else 0)
        # 同時驅動左搖桿（遊戲多半讀搖桿；D-pad 給選單用），數位滿舵
        x = (1 if Action.RIGHT.value in target else 0) - (1 if Action.LEFT.value in target else 0)
        y = (1 if Action.UP.value in target else 0) - (1 if Action.DOWN.value in target else 0)
        self.pad.left_joystick(x_value=int(x * 32767), y_value=int(y * 32767))
        self.pad.update()
        self._held = set(target)

    def apply_vec(self, vec):
        self.apply_keys(vec_to_keys(vec))

    def release_all(self):
        self.pad.reset()
        self.pad.update()
        self._held.clear()


def make_actuator(backend="keyboard"):
    """依後端建立執行器。'keyboard'=pydirectinput；'gamepad'=虛擬手把。"""
    if backend == "gamepad":
        return GamepadActuator()
    return Actuator()
