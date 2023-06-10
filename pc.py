import pyautogui
import cv2
import numpy as np

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.

class Monitor:
    def __init__(self, window_title, window_location, window_size):
        width, height = window_size
        left, top = window_location
        self.region = (left, top, width, height)

        window = self.find_window(window_title)
        self.activate_window(window)
        window.resizeTo(width, height)
        window.moveTo(left, top)

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
        img = np.array(img, dtype=np.int32)
        return img


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
