import cv2
import pyautogui
import numpy as np

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.

class Monitor:
    def __init__(self, window_title, window_location, window_size):
        self.width, self.height = window_size
        self.left, self.top = window_location
        self.region = (self.left, self.top, self.width, self.height)

        self.window = self.find_window(window_title)
        self.activate_move_to_desired()

    @staticmethod
    def find_window(window_title):
        window = pyautogui.getWindowsWithTitle(window_title)
        assert len(window) == 1, f"found {len(window)} windows called {window_title}"
        window = window[0]
        return window

    @staticmethod
    def activate_window(window):
        try:
            window.activate()
        except Exception:
            window.minimize()
            window.maximize()
            window.restore()

    def capture(self):
        img = pyautogui.screenshot(region=self.region)
        img = np.array(img, dtype=np.float32)
        return img


    def find(self, target_region, target_image):
        return pyautogui.locateOnScreen(target_image,
                                        region=target_region,
                                        confidence=0.95)

    def is_active(self):
        return self.window.isActive

    
    def activate_move_to_desired(self):
        self.activate_window(self.window)
        self.window.resizeTo(self.width, self.height)
        self.window.moveTo(self.left, self.top)


class Keyboard:
    def __init__(self, keys):
        self.keys = keys

    def execute(self, action):
        assert len(action) == len(self.keys), f"wrong action size"

        for status, key in zip(action, self.keys):
            if status:
                pyautogui.keyDown(key)
            else:
                pyautogui.keyUp(key)
