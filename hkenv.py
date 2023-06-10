import pyautogui
import cv2
import time
import numpy as np

from pc import Monitor, Keyboard

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.

class HKEnv():
    WINDOW_TITLE = "Hollow Knight"
    WINDOW_SIZE = (1280, 720)
    WINDOW_LOCATION = (0, 0)
    KEYS = ('up', 'down', 'left', 'right', 'z', 'x', 'c', 'v', 's')
    CHARACTER_FULL_HP = 9

    def __init__(self):
        self.monitor = Monitor(self.WINDOW_TITLE, self.WINDOW_LOCATION, self.WINDOW_SIZE)
        self.keyboard = Keyboard(self.KEYS)

        self.initialize()
        self.reset_env()

    def initialize(self):
        enempy_hp_location = (1 / 4 + 1 / 300, 49 / 52)
        enemy_hp_size = (1 / 2 - 2 / 300, 1 / 35)
        enemy_hp_region = (
            int(self.WINDOW_LOCATION[0] + enempy_hp_location[0] * self.WINDOW_SIZE[0]),
            int(self.WINDOW_LOCATION[1] + enempy_hp_location[1] * self.WINDOW_SIZE[1]),
            int(enemy_hp_size[0] * self.WINDOW_SIZE[0]),
            int(enemy_hp_size[1] * self.WINDOW_SIZE[1]),
        )
        enemy_hp_target_row = enemy_hp_region[1] + int(self.WINDOW_SIZE[1] * enemy_hp_size[1] / 2)
        self.enemy_hp_slice = np.s_[enemy_hp_target_row,
                                    enemy_hp_region[0]:enemy_hp_region[0] + enemy_hp_region[2]]

        character_hp_location = (1 / 6 - 1 / 200 + 1 / 140, 1 / 8)
        character_hp_size = (1 / 4, 1 / 128)
        character_hp_region = (
            int(self.WINDOW_LOCATION[0] + character_hp_location[0] * self.WINDOW_SIZE[0]),
            int(self.WINDOW_LOCATION[1] + character_hp_location[1] * self.WINDOW_SIZE[1]),
            int(character_hp_size[0] * self.WINDOW_SIZE[0]),
            int(character_hp_size[1] * self.WINDOW_SIZE[1]),
        )
        character_hp_target_row = character_hp_region[1] + int(self.WINDOW_SIZE[1] * character_hp_size[1] / 2)
        mask_width = int(1 / 20 * self.WINDOW_SIZE[1])
        self.character_hp_slice = np.s_[character_hp_target_row,
                                        character_hp_region[0]:character_hp_region[0] + character_hp_region[2]:mask_width, 0]


    def reset_env(self):
        self.enemy_full_hp = True

    def observe(self):
        frame = self.monitor.capture()

        enemy_remain, enemy_full = self._get_enemy_hp(frame)
        character_remain = self._get_character_hp(frame)
        return enemy_remain, enemy_full, character_remain, self.CHARACTER_FULL_HP

    def _get_enemy_hp(self, frame):
        bar = frame[self.enemy_hp_slice]
        channel_diff = bar[:, 0] - bar[:, 1] - bar[:, 2]

        enemy_full = channel_diff.shape[0]
        enemy_remain = np.sum(channel_diff > 5)

        if self.enemy_full_hp:
            if enemy_remain == 0:
                enemy_remain = enemy_full
            else:
                self.enemy_full_hp = False

        return enemy_remain, enemy_full


    def _get_character_hp(self, frame):
        bar = frame[self.character_hp_slice]
        remain = np.sum(bar > 150)
        return remain


    def test(self):
        action = (0, 0, 1, 0, 0, 0, 1, 0, 0)
        self.keyboard.execute(action)
        
        time.sleep(0.3)
        action = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.keyboard.execute(action)

        time.sleep(0.1)
        action = (0, 0, 1, 0, 1, 0, 0, 0, 0)
        self.keyboard.execute(action)
        
        time.sleep(0.1)
        action = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.keyboard.execute(action)
        
        time.sleep(0.1)
        action = (0, 0, 0, 1, 0, 0, 0, 0, 0)
        self.keyboard.execute(action)
        
        time.sleep(0.1)
        action = (0, 0, 0, 1, 1, 0, 0, 0, 0)
        self.keyboard.execute(action)
        
        time.sleep(0.1)
        action = (0, 0, 0, 1, 1, 0, 0, 0, 0)
        self.keyboard.execute(action)
        
        time.sleep(0.1)
        action = (0, 0, 1, 0, 1, 0, 0, 0, 0)
        self.keyboard.execute(action)
        
        time.sleep(0.1)
        action = (0, 0, 1, 0, 1, 0, 0, 0, 0)
        self.keyboard.execute(action)

        time.sleep(0.1)
        
        # shriek pogo start
        time.sleep(0.1)
        action = (1, 0, 0, 0, 0, 0, 0, 1, 0)
        self.keyboard.execute(action)
        self.observe()
       
        time.sleep(0.6)
        action = (0, 1, 0, 0, 1, 1, 0, 0, 0)
        self.keyboard.execute(action)
        self.observe()

        time.sleep(0.1)
        action = (0, 0, 0, 0, 1, 0, 0, 0, 0)
        self.keyboard.execute(action)
        self.observe()
        
        time.sleep(0.2)
        action = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.keyboard.execute(action)
        self.observe()
        # shriek pogo end

        # shriek pogo start
        time.sleep(0.1)
        action = (1, 0, 0, 0, 0, 0, 0, 1, 0)
        self.keyboard.execute(action)
        self.observe()
        
        time.sleep(0.6)
        action = (0, 1, 0, 0, 1, 1, 0, 0, 0)
        self.keyboard.execute(action)
        self.observe()

        time.sleep(0.1)
        action = (0, 0, 0, 0, 1, 0, 0, 0, 0)
        self.keyboard.execute(action)
        self.observe()
        
        time.sleep(0.2)
        action = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.keyboard.execute(action)
        self.observe()
        # shriek pogo end

        # shriek pogo start
        time.sleep(0.1)
        action = (1, 0, 0, 0, 0, 0, 0, 1, 0)
        self.keyboard.execute(action)
        self.observe()
        
        time.sleep(0.6)
        action = (0, 1, 0, 0, 1, 1, 0, 0, 0)
        self.keyboard.execute(action)
        self.observe()

        time.sleep(0.1)
        action = (0, 0, 0, 0, 1, 0, 0, 0, 0)
        self.keyboard.execute(action)
        self.observe()
        
        time.sleep(0.2)
        action = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.keyboard.execute(action)
        self.observe()
        # shriek pogo end

        time.sleep(0.1)
        action = (0, 0, 0, 0, 1, 0, 0, 0, 0)
        self.keyboard.execute(action)

        time.sleep(0.4)
        action = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.keyboard.execute(action)

        time.sleep(0.1)
        action = (0, 0, 0, 1, 0, 0, 1, 0, 0)
        self.keyboard.execute(action)
        
        time.sleep(0.1)
        action = (0, 0, 0, 1, 0, 0, 0, 0, 0)
        self.keyboard.execute(action)
        
        time.sleep(0.3)
        action = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.keyboard.execute(action)

