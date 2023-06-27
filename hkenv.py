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

class Action:
    KEYS_MAP = ('up', 'down', 'left', 'right', 'z', 'x', 'c', 'v', 's')
    ALL_POSSIBLE = unpackbits(np.arange(2 ** len(Keys)), len(Keys))

    ATTACK_HOLD_TIME = 1.35 + 0.1
    SPELL_HOLD_TIME = 0.62 + 0.1
    DASH_HOLD_TIME = 0.4 + 0.1
    JUMP_HOLD_TIME = 0.5 + 0.1
    SDASH_HOLD_TIME = 1.0 + 0.1

    def __init__(self, idx):
        self.idx = idx
        self.device = "cpu"

    @staticmethod
    def key_idx(key):
        return 2 ** key

    def has(self, key):
        return Action.ALL_POSSIBLE[self.idx][key].item()

    def _add(self, key):
        if self.has(key):
            return
        self.idx = self.idx + Action.key_idx(key)

    def _remove(self, key):
        if not self.has(key):
            return
        self.idx = self.idx - Action.key_idx(key)

    def to(self, device):
        self.device = device
        return self

    @property
    def keys(self):
        return torch.clone(Action.ALL_POSSIBLE[self.idx]).to(self.device)

    @property
    def conflict_num(self):
        res = 0
        if self.has(Keys.LEFT) and self.has(Keys.RIGHT):
            res = res + 1
        if self.has(Keys.SDASH):
            res = res + self.has(Keys.JUMP) + self.has(Keys.ATTACK) + self.has(Keys.DASH) + self.has(Keys.SPELL)
        return res

    def mutate(self, key_hold, observe_interval):
        if torch.rand(1) < 0.9:
            self._remove_conflict()
        if torch.rand(1) < 0.1:
            self._simplify(key_hold, observe_interval)
        if torch.rand(1) < 0.1:
            self._complicate(key_hold, observe_interval)

    def _remove_conflict(self):
        if self.has(Keys.LEFT) and self.has(Keys.RIGHT):
            candidates = [Keys.LEFT, Keys.RIGHT]
            change_key = candidates[torch.randint(len(candidates), (1,))]
            self._remove(change_key)

        if self.has(Keys.JUMP) or self.has(Keys.ATTACK) or self.has(Keys.DASH) or self.has(Keys.SDASH):
            if self.has(Keys.SDASH):
                self._remove(Keys.SDASH)

    def _complicate(self, key_hold, observe_interval):
        if (key_hold[Keys.JUMP] > 0 and key_hold[Keys.JUMP] < Action.JUMP_HOLD_TIME / observe_interval and
            not self.has(Keys.JUMP)):
            self._add(Keys.JUMP)

        if not self.has(Keys.UP) and not self.has(Keys.DOWN) and not self.has(Keys.LEFT) and not self.has(Keys.DOWN):
            if self.has(Keys.SPELL):
                candidates = [Keys.UP, Keys.DOWN, Keys.LEFT, Keys.RIGHT]
                change_key = candidates[torch.randint(len(candidates), (1,))]
                self._add(change_key)

        if not self.has(Keys.LEFT) and not self.has(Keys.RIGHT):
            if self.has(Keys.DASH):
                candidates = [Keys.LEFT, Keys.RIGHT]
                change_key = candidates[torch.randint(len(candidates), (1,))]
                self._add(change_key)

        if not self.has(Keys.UP) and not self.has(Keys.DOWN) and not self.has(Keys.LEFT) and not self.has(Keys.DOWN):
            if key_hold[Keys.ATTACK] > Action.ATTACK_HOLD_TIME / observe_interval and not self.has(Keys.ATTACK):
                candidates = [Keys.UP, Keys.DOWN, Keys.LEFT, Keys.RIGHT]
                change_key = candidates[torch.randint(len(candidates), (1,))]
                self._add(change_key)

                if (change_key == Keys.LEFT or change_key == Keys.RIGHT) and torch.rand(1) < 0.5:
                    self._add(Keys.DASH)


    def _simplify(self, key_hold, observe_interval):
        if key_hold[Keys.ATTACK] and self.has(Keys.ATTACK):
            self._remove(Keys.ATTACK)

        if key_hold[Keys.JUMP] > Action.JUMP_HOLD_TIME / observe_interval and self.has(Keys.JUMP):
            self._remove(Keys.JUMP)

        if key_hold[Keys.DASH] > Action.DASH_HOLD_TIME / observe_interval and self.has(Keys.DASH):
            self._remove(Keys.DASH)

        if key_hold[Keys.SPELL] > Action.SPELL_HOLD_TIME / observe_interval and self.has(Keys.SPELL):
            self._remove(Keys.SPELL)

        if key_hold[Keys.SDASH] > Action.SDASH_HOLD_TIME / observe_interval and self.has(Keys.SDASH):
            self._remove(Keys.SDASH)

        if self.has(Keys.UP) and self.has(Keys.DOWN):
            candidates = [Keys.UP, Keys.DOWN]
            change_key = candidates[torch.randint(len(candidates), (1,))]
            self._add(change_key)

    @staticmethod
    def check_key_hold_too_long(key_hold, observe_interval):
        if key_hold[Keys.JUMP] > Action.JUMP_HOLD_TIME / observe_interval:
            return True
        if key_hold[Keys.DASH] > Action.DASH_HOLD_TIME / observe_interval:
            return True
        if key_hold[Keys.SPELL] > Action.SPELL_HOLD_TIME / observe_interval:
            return True
        return False

class BasicAction:
    NULL = Action(0)
    UP = Action(2 ** Keys.UP)
    DOWN = Action(2 ** Keys.DOWN)
    LEFT = Action(2 ** Keys.LEFT)
    RIGHT = Action(2 ** Keys.RIGHT)
    JUMP = Action(2 ** Keys.JUMP)
    ATTACK = Action(2 ** Keys.ATTACK)
    DASH = Action(2 ** Keys.DASH)
    SPELL = Action(2 ** Keys.SPELL)
    SDASH = Action(2 ** Keys.SDASH)

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
    NOTHING_HAPPEN_REWARD = -0.01

    def __init__(self, observe_size, observe_interval, info_size, device):
        self.monitor = Monitor(self.WINDOW_TITLE, self.WINDOW_LOCATION, self.WINDOW_SIZE)
        self.keyboard = Keyboard(Action.KEYS_MAP)
        self.observe_size = observe_size
        self.observe_interval = observe_interval
        self.info_size = info_size
        self.device = device

        self.enemy_remain_weight_counter = Counter(init=1, increase=-0.01, high=1, low=0.5)
        self.character_remain_weight_counter = Counter(init=-2, increase=0.01, high=-1, low=-2)
        self._reset_env()
        self.observe_base = cv2.imread("locator/base_field.png", cv2.IMREAD_GRAYSCALE)
        self.observe_base = torch.from_numpy(self.observe_base).to(self.device)

    def _reset_env(self):
        time.sleep(0.2)
        self.keyboard.execute(BasicAction.NULL)
        time.sleep(0.2)

        self.is_enemy_full_hp = True
        self.prev_enemy_remain = self.ENEMY_FULL_HP
        self.prev_character_remain = self.CHARACTER_FULL_HP
        self.prev_time = time.time()
        self.key_hold = torch.zeros(len(Keys))
        self.enemy_remain_weight_counter.reset()
        self.character_remain_weight_counter.reset()

    def close(self):
        self._reset_env()

    def observe(self):
        pass_time = time.time() - self.prev_time
        self.prev_time = time.time()
        if pass_time < self.observe_interval:
            time.sleep(self.observe_interval - pass_time)
        frame = self.monitor.capture()

        enemy_remain = self._get_enemy_hp(frame)
        character_remain = self._get_character_hp(frame)

        observe = cv2.resize(frame[self.OBSERVE_SLICE], self.observe_size, interpolation=cv2.INTER_AREA)
        observe = cv2.cvtColor(observe, cv2.COLOR_RGB2GRAY)
        observe = torch.from_numpy(observe).to(self.device)
        observe = observe - self.observe_base

        del frame
        return observe, enemy_remain, character_remain

    def _prepare(self):
        while not self.monitor.find(self.MENU_REGION, "locator\menu_badge.png"):
            stop = False
            while not self.monitor.is_active():
                if not stop:
                    print(f"stop", end='\r')
                    stop = True
                time.sleep(10)
            if stop:
                self.monitor.activate_move_to_desired()
                del stop

            self.keyboard.execute(BasicAction.UP)
            time.sleep(0.2)
            self.keyboard.execute(BasicAction.NULL)
            time.sleep(0.2)

        self.keyboard.execute(BasicAction.JUMP)
        time.sleep(0.2)
        self.keyboard.execute(BasicAction.NULL)
        time.sleep(0.2)

    def _update_key_hold(self, action, beaten):
        for key in Keys:
            self.key_hold[key] = self.key_hold[key] + 1 if action.has(key) else 0

        if beaten:
            for i in range(4, len(Keys)):
                self.key_hold[i] = 0

    def step(self, action):
        self.keyboard.execute(action)
        observe, enemy_remain, character_remain = self.observe()

        win = (enemy_remain == 0)
        lose = (character_remain == 0)
        done = (win or lose)

        hit = enemy_remain < self.prev_enemy_remain
        beaten = character_remain < self.prev_character_remain
        self._update_key_hold(action, beaten)

        reward = self._calculate_reward(enemy_remain, character_remain, action)
        self.enemy_remain_weight_counter.step()
        self.character_remain_weight_counter.step()

        info = torch.zeros(self.info_size).to(self.device)
        info[:len(Keys)] = self.key_hold

        if hit:
            self.enemy_remain_weight_counter.reset()
        if beaten:
            self.character_remain_weight_counter.reset()

        self.prev_character_remain = character_remain
        self.prev_enemy_remain = enemy_remain

        del character_remain
        del win, lose
        return observe, info, reward, done, enemy_remain

    def reset(self, n_frames):
        self._prepare()
        time.sleep(4)
        self._reset_env()
        return self.init_observe(n_frames)

    def init_observe(self, n_frames):
        observe_list, action_list, info_list = [], [], []
        for i in range(n_frames):
            observe, enemy_remain, character_remain = self.observe()
            action = BasicAction.NULL
            info = torch.zeros(self.info_size)

            observe_list.append(observe.to(self.device))
            action_list.append(action.keys.to(self.device))
            info_list.append(info.to(self.device))

            # observe_cpu_numpy = observe.cpu().numpy()
            # cv2.imwrite(f"images/obs{i}.png", observe_cpu_numpy)

        return observe_list, action_list, info_list


    def _calculate_reward(self, enemy_remain, character_remain, action):
        win = (enemy_remain == 0)
        lose = (character_remain == 0)

        done_reward = 0
        if win:
            done_reward = self.WIN_REWARD * (character_remain + 1)
        elif lose:
            done_reward = self.LOSE_REWARD

        enemy_hp_reward = (self.prev_enemy_remain - enemy_remain) * self.enemy_remain_weight_counter.val
        character_hp_reward = (self.prev_character_remain - character_remain) * self.character_remain_weight_counter.val
        key_conflict_reward = action.conflict_num * self.KEY_CONFLICT_REWARD
        key_hold_reward = 0 if not Action.check_key_hold_too_long(self.key_hold, self.observe_interval) else self.KEY_HOLD_REWARD

        nothing_happen_reward = 0
        if self.prev_character_remain == character_remain and self.prev_enemy_remain == enemy_remain:
            nothing_happen_reward = self.NOTHING_HAPPEN_REWARD

        reward = done_reward + enemy_hp_reward + character_hp_reward
        reward = reward + key_conflict_reward + key_hold_reward + nothing_happen_reward
        # print(f"{done_reward = }, {enemy_hp_reward = }. {character_hp_reward = }", end='')
        # print(f", {key_conflict_reward = }, {key_hold_reward = }, {nothing_happen_reward = }, {reward = }")
        del win, lose
        del done_reward, enemy_hp_reward, character_hp_reward
        del key_conflict_reward, key_hold_reward, nothing_happen_reward
        return torch.tensor(reward, dtype=torch.float32).to(self.device)

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
            self.keyboard.execute(BasicAction.UP + BasicAction.SPELL)
           
            time.sleep(0.6)
            self.keyboard.execute(BasicAction.DOWN + BasicAction.ATTACK + BasicAction.JUMP)

            time.sleep(0.11)
            self.keyboard.execute(BasicAction.JUMP)
            
            time.sleep(0.2)
            self.keyboard.execute(BasicAction.NULL)

