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
import curriculum
from autostart import EpisodeMonitor, start_challenge
from capture import Capturer
from inputs import make_actuator
from keys import format_action_row
from obs import FrameStacker
from telemetry import TelemetryReceiver


class BossDamageTracker:
    """累積一場戰鬥對 boss 造成的總傷害。

    記下第一個有效血量(boss0)與最新血量(last)，dmg = boss0 - last。
    boss_hp_raw 取自 env 的 step info（<0 表示這 tick 還沒看到 boss）。

    win 例外：尾刀常在兩次 15Hz 取樣間擊殺、boss 隨即消失，last 停在死前
    剩血(>0) → 少算最後一塊（甚至 lose 的 dmg 看起來比 win 還高）。win 代表
    boss 血量全清，真實傷害就是 boss0，故獲勝時直接回傳 boss0。
    """
    def __init__(self):
        self.boss0 = None       # 第一個有效(>=0)的 boss 血量
        self.last = -1          # 最新一次的 boss 血量
        self.won = False        # 本場是否獲勝（boss 血量全清）

    def update(self, info):
        hp = info.get("boss_hp_raw", -1)
        if self.boss0 is None and hp >= 0:
            self.boss0 = hp
        self.last = hp
        if info.get("result") == "win":
            self.won = True
        return self

    @property
    def dmg(self):
        if self.boss0 is None:
            return 0
        if self.won:                       # 獲勝 = boss 滿血全被打掉
            return self.boss0
        return self.boss0 - self.last


class HollowKnightEnv:
    def __init__(self, backend=config.INPUT_BACKEND):
        self.cap = Capturer()
        self.act = make_actuator(backend)
        self.rx = TelemetryReceiver()
        self.rx.start()
        self.stacker = FrameStacker()
        self.monitor = None
        self._next_t = 0.0
        self._ep_wall0 = None   # fps 診斷：本場牆鐘起點
        self._over_runs = []    # fps 診斷：本場各 over-tick 的 (tick_idx, 超出 66.7ms 預算的秒數)
        self._steps = 0
        self._prev_boss = None
        self._prev_player = None
        self._scale = 1.0       # 作法A：本場 reward 正規化 scale（reset 時依 curriculum 設定）
        # B4 遙測健康：每場統計「拿到新鮮遙測的 tick 比例」
        self._tele_ok = 0
        self._tele_total = 0
        # B3 復原：連續 reset 失敗次數（給訓練端記錄/監看）
        self.reset_fail_count = 0

    # ---- 工具 ----
    def _read_tele(self):
        tele, age = self.rx.sample()
        if tele is not None and age < config.TELE_FRESH_SEC:
            return tele
        return None

    def _wait_tick(self):
        now = time.perf_counter()
        margin = self._next_t - now             # >0=還有餘裕(會睡)；<=0=這 tick 的工作已爆 66.7ms 預算
        if margin > 0:
            time.sleep(margin)
        else:
            # 來不及維持 15Hz：記下 (tick_idx, 超出 66.7ms 多少秒)。此時 _steps 尚未 +1，即當前 0-based tick。
            self._over_runs.append((self._steps, -margin))
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
                # 作法A：本場 reward 正規化用的 scale（開場已定、整場固定）。
                # eval 場回 1.0（mod 不放大）；訓練場 = 目前難度比例。
                self._scale = curriculum.effective_scale()
                self._next_t = time.perf_counter() + config.TICK_DT
                # fps 診斷：本場第一個 step 進來時記牆鐘起點；累計各 over-tick (idx, 超時秒數)
                self._ep_wall0 = None
                self._over_runs = []
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
        if self._ep_wall0 is None:              # fps 診斷：本場第一個 step 進來才起算牆鐘
            self._ep_wall0 = time.perf_counter()
        if config.TRACE_ACTIONS:
            print(format_action_row(action_vec), flush=True)
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
            # fps 診斷：實測 fps = 步數 / 牆鐘秒數（涵蓋抓圖/推論/送鍵全部開銷）；
            # over_runs = 各爆預算 tick 的 (idx, 超時秒數)，依超時由大到小。實測 < TICK_HZ 代表某環節吃掉 tick 預算。
            elapsed = time.perf_counter() - self._ep_wall0 if self._ep_wall0 else 0.0
            info["fps"] = self._steps / elapsed if elapsed > 0 else 0.0
            info["over_runs"] = sorted(self._over_runs, key=lambda t: t[1], reverse=True)
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

        # 造成傷害（兩端都有效時才算，避免 -1 sentinel 造成爆衝）。
        # 作法A：×_scale 把「被 mod 放大的 boss 掉血」還原成真實傷害，低難度不再多領。
        if boss >= 0 and self._prev_boss is not None and self._prev_boss >= 0:
            dmg = max(0, self._prev_boss - boss)
            r += config.RW_DMG * dmg * self._scale
        # 自己血量變化（不乘 scale：被打/補血的難度與 boss 血量放大無關）。
        if player >= 0 and self._prev_player is not None and self._prev_player >= 0:
            delta = player - self._prev_player
            if delta < 0:
                # 掉血：漸進——剩血比例越低，挨刀越貴（早期兇得起、末段才怕死）。
                pmax = tele.get("player_max", -1)
                frac = (player / pmax) if pmax and pmax > 0 else 1.0
                late = 1.0 + config.RW_HIT_LATE_K * (1.0 - max(0.0, min(1.0, frac)))
                r -= config.RW_HIT * (-delta) * late
            elif delta > 0:
                # 補血：補滿一格才給分（半截停掉=整數不變=不給分，自然罰浪費魂）。
                r += config.RW_HEAL * delta

        if boss >= 0:
            self._prev_boss = boss
        if player >= 0:
            self._prev_player = player

        result = self.monitor.update(tele)
        terminated = result in ("win", "lose", "left")
        # 作法A：贏弱化版不值錢(×scale)、輸弱化版更痛(/scale，設上限防龜縮)
        if result == "win":
            r += config.RW_WIN * self._scale
        elif result == "lose":
            r -= min(config.RW_LOSE / self._scale, config.RW_LOSE_CAP)
        info["result"] = result
        return r, terminated, info

    def close(self):
        self.act.release_all()
        self.rx.stop()
