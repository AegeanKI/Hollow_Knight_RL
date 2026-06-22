"""真實遊戲的 RL 環境（Gym 風格 reset/step）。

復用階段 0/1 的元件：Capturer + FrameStacker(觀測)、Actuator(動作)、
TelemetryReceiver(算 reward)、autostart(自動開場/重開) + EpisodeMonitor(判勝敗)。

- 觀測：純畫面疊幀（float32 [0,1]）——AI 只看畫面。
- reward：用遙測的血量變化算（訓練訊號，不是 AI 輸入）。
- 一個 episode = 一場大黃蜂戰鬥；reset() 會自動重開下一場。
"""
import time

import numpy as np

import config
from autostart import EpisodeMonitor, start_challenge
from capture import Capturer
from inputs import make_actuator
from obs import FrameStacker
from telemetry import TelemetryReceiver


class HollowKnightEnv:
    def __init__(self, backend=config.INPUT_BACKEND):
        self.cap = Capturer()
        self.act = make_actuator(backend)
        self.rx = TelemetryReceiver()
        self.rx.start()
        self.stacker = FrameStacker()
        self.monitor = None
        self._next_t = 0.0
        self._steps = 0
        self._prev_boss = None
        self._prev_player = None

    # ---- 工具 ----
    def _read_tele(self):
        tele, age = self.rx.sample()
        if tele is not None and age < 1.0:
            return tele
        return None

    def _wait_tick(self):
        now = time.perf_counter()
        if now < self._next_t:
            time.sleep(self._next_t - now)
        self._next_t += config.TICK_DT

    # ---- Gym API ----
    def reset(self, max_retries=3):
        """自動開場進入新一場戰鬥，回傳初始觀測。"""
        self.act.release_all()
        ok = False
        for _ in range(max_retries):
            if start_challenge(self.cap, self.act, self.rx):
                ok = True
                break
            print("  reset 重試...")
            time.sleep(1.0)
        if not ok:
            raise RuntimeError("無法自動開場進入戰鬥，請確認角色在雕像前、mod 有開。")

        self.monitor = EpisodeMonitor()
        self.stacker.reset()
        self.stacker.push(self.cap.grab_raw())
        tele = self._read_tele()
        self._prev_boss = tele.get("boss_hp_raw", -1) if tele else -1
        self._prev_player = tele.get("player_hp", -1) if tele else -1
        self._steps = 0
        self._next_t = time.perf_counter() + config.TICK_DT
        return self.stacker.get()

    def step(self, action_vec):
        """action_vec: MultiBinary(11)。回傳 (obs, reward, terminated, truncated, info)。"""
        self.act.apply_vec(action_vec)
        self._wait_tick()                       # 維持 15Hz

        self.stacker.push(self.cap.grab_raw())
        obs = self.stacker.get()
        tele = self._read_tele()
        self._steps += 1

        reward, terminated, info = self._reward_and_done(tele)
        truncated = self._steps >= config.MAX_EPISODE_STEPS
        if terminated or truncated:
            self.act.release_all()
        return obs, reward, terminated, truncated, info

    def _reward_and_done(self, tele):
        r = config.RW_TIME
        info = {"result": None, "boss_hp_raw": self._prev_boss, "player_hp": self._prev_player}
        if tele is None:
            return r, False, info

        boss = tele.get("boss_hp_raw", -1)
        player = tele.get("player_hp", -1)
        if boss >= 0:
            info["boss_hp_raw"] = boss
        if player >= 0:
            info["player_hp"] = player

        # 造成傷害（兩端都有效時才算，避免 -1 sentinel 造成爆衝）
        if boss >= 0 and self._prev_boss is not None and self._prev_boss >= 0:
            dmg = max(0, self._prev_boss - boss)
            r += config.RW_DMG * dmg
        # 自己掉血
        if player >= 0 and self._prev_player is not None and self._prev_player >= 0:
            hit = max(0, self._prev_player - player)
            r -= config.RW_HIT * hit

        if boss >= 0:
            self._prev_boss = boss
        if player >= 0:
            self._prev_player = player

        result = self.monitor.update(tele)
        terminated = result in ("win", "lose", "left")
        if result == "win":
            r += config.RW_WIN
        elif result == "lose":
            r -= config.RW_LOSE
        info["result"] = result
        return r, terminated, info

    def close(self):
        self.act.release_all()
        self.rx.stop()
