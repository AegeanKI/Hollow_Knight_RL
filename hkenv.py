import pyautogui
from pc import Monitor, Keyboard
import time

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.

class HKEnv():
    WINDOW_TITLE = "Hollow Knight"
    WINDOW_SIZE = (1280, 720)
    WINDOW_LOCATION = (0, 0)
    KEYS = ('up', 'down', 'left', 'right', 'z', 'x', 'c', 'v')

    def __init__(self):
        self.monitor = Monitor(self.WINDOW_TITLE, self.WINDOW_SIZE, self.WINDOW_LOCATION)
        self.keyboard = Keyboard(self.KEYS)


    def test(self):
        action = (0, 0, 1, 0, 0, 0, 1, 0)
        self.keyboard.execute(action)
        
        time.sleep(0.3)
        action = (0, 0, 0, 0, 0, 0, 0, 0)
        self.keyboard.execute(action)

        time.sleep(0.1)
        action = (0, 0, 1, 0, 1, 0, 0, 0)
        self.keyboard.execute(action)
        
        time.sleep(0.1)
        action = (0, 0, 0, 0, 0, 0, 0, 0)
        self.keyboard.execute(action)
        
        time.sleep(0.1)
        action = (0, 0, 0, 1, 0, 0, 0, 0)
        self.keyboard.execute(action)
        
        time.sleep(0.1)
        action = (0, 0, 0, 1, 1, 0, 0, 0)
        self.keyboard.execute(action)
        
        time.sleep(0.1)
        action = (0, 0, 0, 1, 1, 0, 0, 0)
        self.keyboard.execute(action)
        
        time.sleep(0.1)
        action = (0, 0, 1, 0, 1, 0, 0, 0)
        self.keyboard.execute(action)
        
        time.sleep(0.1)
        action = (0, 0, 1, 0, 1, 0, 0, 0)
        self.keyboard.execute(action)

        time.sleep(0.1)
        
        # shriek pogo start
        time.sleep(0.1)
        action = (1, 0, 0, 0, 0, 0, 0, 1)
        self.keyboard.execute(action)
       
        time.sleep(0.6)
        action = (0, 1, 0, 0, 1, 1, 0, 0)
        self.keyboard.execute(action)

        time.sleep(0.1)
        action = (0, 0, 0, 0, 1, 0, 0, 0)
        self.keyboard.execute(action)
        
        time.sleep(0.2)
        action = (0, 0, 0, 0, 0, 0, 0, 0)
        self.keyboard.execute(action)
        # shriek pogo end

        # shriek pogo start
        time.sleep(0.1)
        action = (1, 0, 0, 0, 0, 0, 0, 1)
        self.keyboard.execute(action)
        
        time.sleep(0.6)
        action = (0, 1, 0, 0, 1, 1, 0, 0)
        self.keyboard.execute(action)

        time.sleep(0.1)
        action = (0, 0, 0, 0, 1, 0, 0, 0)
        self.keyboard.execute(action)
        
        time.sleep(0.2)
        action = (0, 0, 0, 0, 0, 0, 0, 0)
        self.keyboard.execute(action)
        # shriek pogo end

        # shriek pogo start
        time.sleep(0.1)
        action = (1, 0, 0, 0, 0, 0, 0, 1)
        self.keyboard.execute(action)
        
        time.sleep(0.6)
        action = (0, 1, 0, 0, 1, 1, 0, 0)
        self.keyboard.execute(action)

        time.sleep(0.1)
        action = (0, 0, 0, 0, 1, 0, 0, 0)
        self.keyboard.execute(action)
        
        time.sleep(0.2)
        action = (0, 0, 0, 0, 0, 0, 0, 0)
        self.keyboard.execute(action)
        # shriek pogo end

        time.sleep(0.1)
        action = (0, 0, 0, 0, 1, 0, 0, 0)
        self.keyboard.execute(action)

        time.sleep(0.5)
        action = (0, 0, 0, 0, 0, 0, 0, 0)
        self.keyboard.execute(action)
        
        time.sleep(0.1)
        action = (0, 0, 0, 1, 0, 0, 1, 0)
        self.keyboard.execute(action)
        
        time.sleep(0.3)
        action = (0, 0, 0, 0, 0, 0, 0, 0)
        self.keyboard.execute(action)

