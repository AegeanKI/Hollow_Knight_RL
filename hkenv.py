import time

import pyautogui
import cv2
import torch
import numpy as np

from utils import Counter, unpackbits
from pc import Monitor, Keyboard

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.

class Hotkey:
    KEYS = ('up', 'down', 'left', 'right', 'z', 'x', 'c', 'v', 's')
    NUM = 2 ** len(KEYS)
    ALL_POSSIBLE = unpackbits(np.arange(2 ** len(KEYS)), len(KEYS))

    NULL = ALL_POSSIBLE[0]
    UP = ALL_POSSIBLE[1]
    DOWN = ALL_POSSIBLE[2]
    LEFT = ALL_POSSIBLE[4]
    RIGHT = ALL_POSSIBLE[8]
    JUMP = ALL_POSSIBLE[16]
    ATTACK = ALL_POSSIBLE[32]
    DASH = ALL_POSSIBLE[64]
    SPELL = ALL_POSSIBLE[128]
    SDASH = ALL_POSSIBLE[256]

class HKEnv():
    WINDOW_TITLE = "Hollow Knight"
    WINDOW_SIZE = (1280, 720)
    WINDOW_LOCATION = (0, 0)
    CHARACTER_FULL_HP = 9

    def __init__(self, obs_size, device):
        self.monitor = Monitor(self.WINDOW_TITLE, self.WINDOW_LOCATION, self.WINDOW_SIZE)
        self.keyboard = Keyboard(Hotkey.KEYS)
        self.obs_size = obs_size
        self.device = device

        self._initialize()
        self._reset_env()

    def _initialize(self):
        # for observe
        enempy_hp_location = (1 / 4 + 1 / 300, 49 / 52)
        enemy_hp_size = (1 / 2 - 2 / 300, 1 / 35)
        enemy_hp_region = self._location_size_to_region(enempy_hp_location, enemy_hp_size)
        enemy_hp_target_row = enemy_hp_region[1] + int(self.WINDOW_SIZE[1] * enemy_hp_size[1] / 2)
        self.enemy_hp_slice = np.s_[enemy_hp_target_row,
                                    enemy_hp_region[0]:enemy_hp_region[0] + enemy_hp_region[2]]
        self.enemy_full_hp = enemy_hp_region[2]

        character_hp_location = (1 / 6 - 1 / 200 + 1 / 140, 1 / 8)
        character_hp_size = (1 / 4, 1 / 128)
        character_hp_region = self._location_size_to_region(character_hp_location, character_hp_size)
        character_hp_target_row = character_hp_region[1] + int(self.WINDOW_SIZE[1] * character_hp_size[1] / 2)
        mask_width = int(1 / 20 * self.WINDOW_SIZE[1])
        self.character_hp_slice = np.s_[character_hp_target_row,
                                        character_hp_region[0]:character_hp_region[0] + character_hp_region[2]:mask_width, 0]

        menu_location = (5 / 9, 4 / 9)
        menu_size = (1 / 12, 1 / 14)
        self.menu_region = self._location_size_to_region(menu_location, menu_size)
        self.menu_slice = np.s_[self.menu_region[1]:self.menu_region[1] + self.menu_region[3],
                                self.menu_region[0]:self.menu_region[0] + self.menu_region[2]]


        # for reward
        self.enemy_remain_weight_counter = Counter(init=0.01, increase=-0.00001, high=0.01, low=0.005)
        self.character_remain_weight_counter = Counter(init=0.0001, increase=0.000001, high=0.0002, low=0.0001)
        self.win_reward = 1
        self.lose_reward = -1



    def _location_size_to_region(self, location, size):
        region = (
            int(self.WINDOW_LOCATION[0] + location[0] * self.WINDOW_SIZE[0]),
            int(self.WINDOW_LOCATION[1] + location[1] * self.WINDOW_SIZE[1]),
            int(size[0] * self.WINDOW_SIZE[0]),
            int(size[1] * self.WINDOW_SIZE[1]),
        )
        return region

    def _reset_env(self):
        self.is_enemy_full_hp = True
        self.prev_enemy_remain = self.enemy_full_hp
        self.prev_character_remain = self.CHARACTER_FULL_HP
        self._counter_reset()
        self.keyboard.execute(Hotkey.NULL)
        # self.prev_time = time.time()

    def observe(self):
        # cur_time = time.time()
        # print(f"time = {cur_time - self.prev_time}")
        frame = self.monitor.capture()

        enemy_remain = self._get_enemy_hp(frame)
        character_remain = self._get_character_hp(frame)

        obs = cv2.resize(frame, self.obs_size, interpolation=cv2.INTER_AREA)
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = torch.from_numpy(obs).to(self.device)

        # self.prev_time = cur_time
        return obs, enemy_remain, character_remain

    def _start(self):
        while True:
            find = self.monitor.find(self.menu_region, "locator\menu_badge.png")
            if find:
                break

            stop = False
            while not self.monitor.is_active():
                stop = True
                time.sleep(10)

            if stop:
                self.monitor.activate_move_to_desired()

            self.keyboard.execute(Hotkey.UP)
            time.sleep(0.1)
            self.keyboard.execute(Hotkey.NULL)
            # time.sleep(0.1)

        self.keyboard.execute(Hotkey.JUMP)
        self.keyboard.execute(Hotkey.NULL)

    def step(self, action_id):
        self.keyboard.execute(Hotkey.ALL_POSSIBLE[action_id])
        obs, enemy_remain, character_remain = self.observe()

        win = (enemy_remain == 0)
        lose = (character_remain == 0)
        done = (win or lose)

        reward = self._calculate_reward(enemy_remain, character_remain)
        self._counter_step()

        if enemy_remain < self.prev_enemy_remain:
            self.enemy_remain_weight_counter.reset()
        if character_remain < self.prev_character_remain:
            self.character_remain_weight_counter.reset()

        self.prev_character_remain = character_remain
        self.prev_enemy_remain = enemy_remain
        return obs, reward, done, None

    def reset(self):
        self._reset_env()
        self._start()
        time.sleep(4)
        obs0, enemy_remain, character_remain = self.observe()
        time.sleep(0.01)
        obs1, enemy_remain, character_remain = self.observe()
        time.sleep(0.01)
        obs2, enemy_remain, character_remain = self.observe()
        time.sleep(0.01)
        return obs0, obs1, obs2

    def close(self):
        self._reset_env()

    def _calculate_reward(self, enemy_remain, character_remain):
        win = (enemy_remain == 0)
        lose = (character_remain == 0)

        done_reward = 0
        if win:
            done_reward = done_reward + self.win_reward
        if lose:
            done_reward = done_reward + self.lose_reward

        # print(f"{self.prev_enemy_remain = }, {enemy_remain = }, {self.enemy_remain_weight_counter.val = }")
        enemy_hp_reward = (self.prev_enemy_remain - enemy_remain) * self.enemy_remain_weight_counter.val
        character_hp_reward = (character_remain + 1 - self.prev_character_remain) * self.character_remain_weight_counter.val
        reward = done_reward + enemy_hp_reward + character_hp_reward
        # print(f"{done_reward = }, {enemy_hp_reward = }. {character_hp_reward = }, {reward = }")
        return torch.Tensor([reward]).to(self.device)

    def _counter_step(self):
        self.enemy_remain_weight_counter.step()
        self.character_remain_weight_counter.step()

    def _counter_reset(self):
        self.enemy_remain_weight_counter.reset()
        self.character_remain_weight_counter.reset()

    def _get_enemy_hp(self, frame):
        bar = frame[self.enemy_hp_slice]
        channel_diff = bar[:, 0] - bar[:, 1] - bar[:, 2]

        enemy_full = channel_diff.shape[0]
        enemy_remain = np.sum(channel_diff > 5)

        if self.is_enemy_full_hp:
            if enemy_remain == 0:
                enemy_remain = enemy_full
            else:
                self.is_enemy_full_hp = False

        return enemy_remain


    def _get_character_hp(self, frame):
        bar = frame[self.character_hp_slice]
        remain = np.sum(bar > 150)
        return remain


    def test(self):
        self.keyboard.execute(Hotkey.LEFT + Hotkey.DASH)
        self.observe()
        
        time.sleep(0.3)
        self.keyboard.execute(Hotkey.NULL)
        self.observe()

        time.sleep(0.1)
        self.keyboard.execute(Hotkey.LEFT + Hotkey.JUMP)
        self.observe()
        
        time.sleep(0.1)
        self.keyboard.execute(Hotkey.NULL)
        self.observe()
        
        time.sleep(0.1)
        self.keyboard.execute(Hotkey.RIGHT)
        self.observe()
        
        time.sleep(0.1)
        self.keyboard.execute(Hotkey.RIGHT + Hotkey.JUMP)
        self.observe()
        
        time.sleep(0.2)
        self.keyboard.execute(Hotkey.LEFT + Hotkey.JUMP)
        self.observe()
        
        time.sleep(0.2)
        self.keyboard.execute(Hotkey.NULL)
        self.observe()
        
        for _ in range(3):
            # shriek pogo
            time.sleep(0.1)
            self.keyboard.execute(Hotkey.UP + Hotkey.SPELL)
            self.observe()
           
            time.sleep(0.6)
            self.keyboard.execute(Hotkey.DOWN + Hotkey.ATTACK + Hotkey.JUMP)
            self.observe()

            time.sleep(0.11)
            self.keyboard.execute(Hotkey.JUMP)
            self.observe()
            
            time.sleep(0.2)
            self.keyboard.execute(Hotkey.NULL)
            self.observe()

        time.sleep(0.1)
        self.keyboard.execute(Hotkey.JUMP)
        self.observe()

        time.sleep(0.4)
        self.keyboard.execute(Hotkey.NULL)
        self.observe()

        time.sleep(0.1)
        self.keyboard.execute(Hotkey.RIGHT + Hotkey.DASH)
        self.observe()
        
        time.sleep(0.3)
        self.keyboard.execute(Hotkey.NULL)
        self.observe()

