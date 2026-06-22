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


class BossDamageTracker:
    """累積一場戰鬥對 boss 造成的總傷害。

    記下第一個有效血量(boss0)與最新血量(last)，dmg = boss0 - last。
    boss_hp_raw 取自 env 的 step info（<0 表示這 tick 還沒看到 boss）。
    """
    def __init__(self):
        self.boss0 = None       # 第一個有效(>=0)的 boss 血量
        self.last = -1          # 最新一次的 boss 血量

    def update(self, info):
        hp = info.get("boss_hp_raw", -1)
        if self.boss0 is None and hp >= 0:
            self.boss0 = hp
        self.last = hp
        return self

    @property
    def dmg(self):
        return (self.boss0 - self.last) if self.boss0 is not None else 0


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
        # B4 遙測健康：每場統計「拿到新鮮遙測的 tick 比例」
        self._tele_ok = 0
        self._tele_total = 0
        # B3 復原：連續 reset 失敗次數（給訓練端記錄/監看）
        self.reset_fail_count = 0

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

    def _sleep_interruptible(self, secs, should_stop):
        """睡 secs 秒，但能被 should_stop 提早打斷。"""
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < secs:
            if should_stop and should_stop():
                return
            time.sleep(0.1)

    # ---- Gym API ----
    def reset(self, max_retries=8, should_stop=None):
        """自動開場進入新一場戰鬥。回傳 (obs, info)（Gymnasium 慣例）。

        B3：永不硬炸。自動開場失敗就退避重試（次數遞增、間隔遞增），
        全程可被 should_stop 打斷。連續失敗到上限仍不成功時，回傳 (None, {})
        （讓訓練端優雅跳過本場/停止，而不是讓整個 session 崩潰）。
        """
        self.act.release_all()
        for attempt in range(max_retries):
            if should_stop and should_stop():
                return None, {}
            if start_challenge(self.cap, self.act, self.rx):
                self.reset_fail_count = 0
                self.monitor = EpisodeMonitor()
                self.stacker.reset()
                self.stacker.push(self.cap.grab_raw())
                tele = self._read_tele()
                self._prev_boss = tele.get("boss_hp_raw", -1) if tele else -1
                self._prev_player = tele.get("player_hp", -1) if tele else -1
                self._steps = 0
                self._tele_ok = 0
                self._tele_total = 0
                self._next_t = time.perf_counter() + config.TICK_DT
                return self.stacker.get(), {}
            wait = min(1.0 * (attempt + 1), 5.0)     # 退避：1,2,3,4,5,5...
            print(f"  reset 第 {attempt + 1}/{max_retries} 次未成功，{wait:.0f}s 後重試...")
            self._sleep_interruptible(wait, should_stop)
        self.reset_fail_count += 1
        print(f"  ⚠ reset 連續 {max_retries} 次失敗（累計失敗 {self.reset_fail_count} 場）："
              f"請確認角色在雕像前、mod/遊戲正常。本場跳過，不中斷訓練。")
        return None, {}

    def tele_drop_rate(self):
        """本場目前為止的遙測掉包率（0=全程有新鮮遙測，1=完全收不到）。"""
        if self._tele_total == 0:
            return 0.0
        return 1.0 - self._tele_ok / self._tele_total

    def step(self, action_vec):
        """action_vec: MultiBinary(11)。回傳 (obs, reward, terminated, truncated, info)。"""
        self.act.apply_vec(action_vec)
        self._wait_tick()                       # 維持 15Hz

        self.stacker.push(self.cap.grab_raw())
        obs = self.stacker.get()
        tele = self._read_tele()
        self._steps += 1
        self._tele_total += 1                  # B4：統計遙測健康度
        if tele is not None:
            self._tele_ok += 1

        reward, terminated, info = self._reward_and_done(tele)
        info["tele_drop"] = self.tele_drop_rate()
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
