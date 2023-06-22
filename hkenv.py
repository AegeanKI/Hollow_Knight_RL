import time
import math
import enum

import cv2
import torch
import pyautogui
import numpy as np

from pc import Monitor, Keyboard
from utils import Counter, unpackbits

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.

class Keys(enum.IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    JUMP = 4
    ATTACK = 5
    DASH = 6
    SPELL = 7
    SDASH = 8

class Operation:
    KEYS_MAP = ('up', 'down', 'left', 'right', 'z', 'x', 'c', 'v', 's')

    POSSIBLE_ACTION = unpackbits(np.arange(2 ** len(Keys)), len(Keys))

    NULL = POSSIBLE_ACTION[0]
    UP = POSSIBLE_ACTION[2 ** Keys.UP]
    DOWN = POSSIBLE_ACTION[2 ** Keys.DOWN]
    LEFT = POSSIBLE_ACTION[2 ** Keys.LEFT]
    RIGHT = POSSIBLE_ACTION[2 ** Keys.RIGHT]
    JUMP = POSSIBLE_ACTION[2 ** Keys.JUMP]
    ATTACK = POSSIBLE_ACTION[2 ** Keys.ATTACK]
    DASH = POSSIBLE_ACTION[2 ** Keys.DASH]
    SPELL = POSSIBLE_ACTION[2 ** Keys.SPELL]
    SDASH = POSSIBLE_ACTION[2 ** Keys.SDASH]

    @staticmethod
    def key_idx(key):
        return 2 ** key

    @staticmethod
    def get_by_idx(idx):
        return Operation.POSSIBLE_ACTION[idx]

    @staticmethod
    def check_contain(action, key):
        return action[key].item()

    @staticmethod
    def count_conflict(action):
        has_left = Operation.check_contain(action, Keys.LEFT)
        has_right = Operation.check_contain(action, Keys.RIGHT)
        has_jump = Operation.check_contain(action, Keys.JUMP)
        has_attack = Operation.check_contain(action, Keys.ATTACK)
        has_dash = Operation.check_contain(action, Keys.DASH)
        has_spell = Operation.check_contain(action, Keys.SPELL)
        has_sdash = Operation.check_contain(action, Keys.SDASH)

        res = 0
        if has_left and has_right:
            res = res + 1
        if has_sdash:
            res = res + has_jump + has_attack + has_dash + has_spell
        return res

    @staticmethod
    def replace_conflict(action_idx, action):
        if torch.rand(1) < 0.1:
            return action_idx, action

        has_left = Operation.check_contain(action, Keys.LEFT)
        has_right = Operation.check_contain(action, Keys.RIGHT)
        has_jump = Operation.check_contain(action, Keys.JUMP)
        has_attack = Operation.check_contain(action, Keys.ATTACK)
        has_dash = Operation.check_contain(action, Keys.DASH)
        has_spell = Operation.check_contain(action, Keys.SPELL)
        has_sdash = Operation.check_contain(action, Keys.SDASH)

        if has_left and has_right:
            change_key = Keys.LEFT if torch.rand(1) < 0.5 else Keys.RIGHT
            action[change_key] = 0
            action_idx = action_idx - Operation.key_idx(change_key)

        if has_sdash:
            if has_jump or has_attack or has_dash or has_spell:
                action[Keys.SDASH] = 0
                action_idx = action_idx - Operation.key_idx(Keys.SDASH)
        return action_idx, action



class HKEnv():
    WINDOW_TITLE = "Hollow Knight"
    WINDOW_SIZE = (1280, 720)
    WINDOW_LOCATION = (0, 0)

    CHARACTER_FULL_HP = 9
    ENEMY_FULL_HP = 955 - 324
    ENEMY_HP_SLICE = np.s_[688, 324:955]
    CHARACTER_HP_SLICE = np.s_[92, 216:536:36, 0]
    MENU_REGION = (711, 320, 106, 51)
    MENU_SLICE = np.s_[320:371, 711:817]
    OBSERVE_SLICE = np.s_[31:648, 47:1232]

    WIN_REWARD = 10
    LOSE_REWARD = -10
    KEY_CONFLICT_REWARD = -10
    KEY_HOLD_REWARD = -10
    NOTHING_HAPPEN_REWARD = -0.1

    def __init__(self, observe_size, info_size, device):
        self.monitor = Monitor(self.WINDOW_TITLE, self.WINDOW_LOCATION, self.WINDOW_SIZE)
        self.keyboard = Keyboard(Operation.KEYS_MAP)
        self.observe_size = observe_size
        self.info_size = info_size
        self.device = device

        self.enemy_remain_weight_counter = Counter(init=1, increase=-0.02, high=1, low=0.5)
        self.character_remain_weight_counter = Counter(init=1, increase=0.01, high=2, low=1)
        self._reset_env()
        self.observe_base = cv2.imread("locator/base_field.png", cv2.IMREAD_GRAYSCALE)
        self.observe_base = torch.from_numpy(self.observe_base).to(self.device)

    def _location_size_to_region(self, location, size): # may comment after change to CONSTANT
        region = (
            int(self.WINDOW_LOCATION[0] + location[0] * self.WINDOW_SIZE[0]),
            int(self.WINDOW_LOCATION[1] + location[1] * self.WINDOW_SIZE[1]),
            int(size[0] * self.WINDOW_SIZE[0]),
            int(size[1] * self.WINDOW_SIZE[1]),
        )
        return region

    def _reset_env(self):
        self.is_enemy_full_hp = True
        self.prev_enemy_remain = self.ENEMY_FULL_HP
        self.prev_character_remain = self.CHARACTER_FULL_HP
        self.prev_time = time.time()
        self.key_hold = torch.zeros(len(Keys))
        self._counter_reset()
        self.keyboard.execute(Operation.NULL)

    def observe(self):
        # cur_time = time.time()
        # print(f"time = {cur_time - self.prev_time}")
        frame = self.monitor.capture()

        enemy_remain = self._get_enemy_hp(frame)
        character_remain = self._get_character_hp(frame)

        observe = cv2.resize(frame[self.OBSERVE_SLICE], self.observe_size, interpolation=cv2.INTER_AREA)
        observe = cv2.cvtColor(observe, cv2.COLOR_RGB2GRAY)
        observe = torch.from_numpy(observe).to(self.device)
        observe = observe - self.observe_base

        del frame
        # self.prev_time = cur_time
        return observe, enemy_remain, character_remain

    def _start(self):
        while not self.monitor.find(self.MENU_REGION, "locator\menu_badge.png"):
            stop = False
            while not self.monitor.is_active():
                if not stop:
                    print(f"stop")
                    stop = True
                time.sleep(10)
            if stop:
                self.monitor.activate_move_to_desired()

            self.keyboard.execute(Operation.UP)
            time.sleep(0.2)
            self.keyboard.execute(Operation.NULL)
            time.sleep(0.2)

        self.keyboard.execute(Operation.JUMP)
        self.keyboard.execute(Operation.NULL)

        del stop

    def step(self, action):
        self.keyboard.execute(action)
        observe, enemy_remain, character_remain = self.observe()

        win = (enemy_remain == 0)
        lose = (character_remain == 0)
        done = (win or lose)

        reward = self._calculate_reward(enemy_remain, character_remain, action)
        self._counter_step()

        info = torch.zeros(self.info_size)
        if enemy_remain < self.prev_enemy_remain:
            self.enemy_remain_weight_counter.reset()
        if character_remain < self.prev_character_remain:
            self.character_remain_weight_counter.reset()
            info[0] = 1 # get hit info

        self.prev_character_remain = character_remain
        self.prev_enemy_remain = enemy_remain

        del character_remain
        del win, lose
        return observe, info, reward, done, enemy_remain

    def reset(self, n_frames, observe_interval):
        self._reset_env()
        self._start()
        time.sleep(3)
        return self.for_warmup(n_frames, observe_interval)

    def for_warmup(self, n_frames, observe_interval):
        observe_list, action_list, info_list = [], [], []
        for i in range(n_frames):
            observe, enemy_remain, character_remain = self.observe()
            action = Operation.NULL.to(self.device)
            info = torch.zeros(self.info_size)

            observe_list.append(observe.to(self.device))
            action_list.append(action.to(self.device))
            info_list.append(info.to(self.device))
            time.sleep(0.02 + observe_interval - 0.07)

            # observe_cpu_numpy = obs.cpu().numpy()
            # cv2.imwrite(f"images/obs{i}.png", observe_cpu_numpy)

        return observe_list, action_list, info_list

    def close(self):
        self._reset_env()

    def _check_key_hold_too_long(self, action):
        for i in range(len(Keys)):
            if action[i]:
                self.key_hold[i] = self.key_hold[i] + 1
            else:
                self.key_hold[i] = 0

        if self.key_hold[Keys.JUMP] > 5:
            return True
        if self.key_hold[Keys.DASH] > 3:
            return True
        if self.key_hold[Keys.SPELL] > 3:
            return True
        return False

    def _calculate_reward(self, enemy_remain, character_remain, action):
        win = (enemy_remain == 0)
        lose = (character_remain == 0)

        done_reward = 0
        if win:
            done_reward = self.WIN_REWARD + math.log(character_remain + 1)
        elif lose:
            done_reward = self.LOSE_REWARD

        enemy_hp_reward = (self.prev_enemy_remain - enemy_remain) * self.enemy_remain_weight_counter.val
        character_hp_reward = (character_remain - self.prev_character_remain) * self.character_remain_weight_counter.val
        # key_conflict_reward = 0 if not Operation.count_conflict(action) else self.KEY_CONFLICT_REWARD
        key_conflict_reward = Operation.count_conflict(action) * self.KEY_CONFLICT_REWARD
        key_hold_reward = 0 if not self._check_key_hold_too_long(action) else self.KEY_HOLD_REWARD

        nothing_happen_reward = 0
        if self.prev_character_remain == character_remain and self.prev_enemy_remain == enemy_remain:
            nothing_happen_reward = self.NOTHING_HAPPEN_REWARD

        reward = done_reward + enemy_hp_reward + character_hp_reward
        reward = reward + key_conflict_reward + key_hold_reward + nothing_happen_reward
        # print(f"{done_reward = }, {enemy_hp_reward = }. {character_hp_reward = }, {key_reward = }, {reward = }")
        del win, lose
        del done_reward, enemy_hp_reward, character_hp_reward
        del key_conflict_reward, key_hold_reward, nothing_happen_reward
        return torch.tensor(reward, dtype=torch.float32).to(self.device)

    def _counter_step(self):
        self.enemy_remain_weight_counter.step()
        self.character_remain_weight_counter.step()

    def _counter_reset(self):
        self.enemy_remain_weight_counter.reset()
        self.character_remain_weight_counter.reset()

    def _get_enemy_hp(self, frame):
        bar = frame[self.ENEMY_HP_SLICE]
        channel_diff = bar[:, 0] - bar[:, 1] - bar[:, 2]

        enemy_full = channel_diff.shape[0]
        enemy_remain = np.sum(channel_diff > 5)

        if self.is_enemy_full_hp:
            if enemy_remain == 0:
                enemy_remain = enemy_full
            else:
                self.is_enemy_full_hp = False

        del bar, channel_diff
        del enemy_full
        return enemy_remain

    def _get_character_hp(self, frame):
        bar = frame[self.CHARACTER_HP_SLICE]
        remain = np.sum(bar > 150)
        del bar
        return remain

    def test(self):
        for _ in range(3):
            # shriek pogo
            time.sleep(0.1)
            self.keyboard.execute(Operation.UP + Operation.SPELL)
            self.observe()
           
            time.sleep(0.6)
            self.keyboard.execute(Operation.DOWN + Operation.ATTACK + Operation.JUMP)
            self.observe()

            time.sleep(0.11)
            self.keyboard.execute(Operation.JUMP)
            self.observe()
            
            time.sleep(0.2)
            self.keyboard.execute(Operation.NULL)
            self.observe()

