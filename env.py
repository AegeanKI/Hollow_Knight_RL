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
    """累積一場戰鬥對 boss 造成的總傷害。dmg = 滿血 - 最新血量；獲勝直接回滿血。

    **滿血基準用 mod 的 `boss_max`（其內部 running-max ＝真實滿血，boss 生成那刻就抓到、
    早於 agent 揮第一刀）**，而非首次觀測值 boss0——後者若 Python 第一幀晚抓到（已挨一刀）
    會偏低、且每場零頭不一，使「全勝時 avg 傷害無法精確打平、tiebreaker 失效」。boss_max
    每場恆定 → 全勝 avg 精確相等 → rl_best 的剩血/用時 tiebreaker 可靠接手。boss0 留作
    boss_max 沒拿到時的後備。

    win 例外：尾刀常在兩次 15Hz 取樣間擊殺、boss 隨即消失，last 停在死前剩血(>0) → 少算
    最後一塊（甚至 lose 的 dmg 看起來比 win 還高）。win 代表血量全清，故獲勝直接回滿血。
    """
    def __init__(self):
        self.boss0 = None       # 第一個有效(>=0)的 boss 血量（後備基準）
        self.boss_max = -1      # mod 的 running-max 滿血（主基準，每場恆定）
        self.last = -1          # 最新一次的有效 boss 血量
        self.won = False        # 本場是否獲勝（boss 血量全清）

    def update(self, info):
        hp = info.get("boss_hp_raw", -1)
        if self.boss0 is None and hp >= 0:
            self.boss0 = hp
        if hp >= 0:
            self.last = hp                          # 只記有效值，避免 -1 污染
        bm = info.get("boss_max", -1)
        if bm > 0:
            self.boss_max = max(self.boss_max, bm)  # running-max（boss_max 本就恆定，取 max 保險）
        if info.get("result") == "win":
            self.won = True
        return self

    @property
    def _full(self):
        # 優先用 mod 的 boss_max（真實滿血、每場一致）；沒拿到才退回首次觀測 boss0
        if self.boss_max > 0:
            return self.boss_max
        return self.boss0

    @property
    def dmg(self):
        full = self._full
        if not full or full <= 0:
            return 0
        if self.won:                       # 獲勝 = boss 滿血全被打掉
            return full
        if self.last < 0:                  # 整場沒拿到有效 boss 血量 → 無法估
            return 0
        return max(0, full - self.last)


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
        # privileged critic：特權特徵的「上一步」基準（掉包/未見到時 backfill 用），每場 reset
        self._prev_boss_frac = 1.0
        self._prev_player_frac = 1.0
        self._prev_soul_frac = 0.0
        # B4 遙測健康：每場統計「拿到新鮮遙測的 tick 比例」
        self._tele_ok = 0
        self._tele_total = 0
        # B3 復原：連續 reset 失敗次數（給訓練端記錄/監看）
        self.reset_fail_count = 0

    # ---- 工具 ----
    def _critic_extra(self, tele):
        """privileged critic 的特權特徵 (float32[N_CRITIC_EXTRA])，全部正規化到 [0,1]。
        順序固定 [scale, boss_hp_frac, player_hp_frac, soul_frac]；只給 critic、不給 actor。
        -1/無效時用上一步、整場沒看過用語意預設（boss/player 滿、soul 0）。
        boss 直接用 mod 已正規化的 'boss_hp'（=raw/running-max，沒 boss 時送 -1）。
        soul 正規化分母 = soul_max + soul_reserve_max（soul_total 含主槽+儲備）。"""
        if tele is not None:
            bh = tele.get("boss_hp", -1.0)
            if bh is not None and bh >= 0:
                self._prev_boss_frac = bh
            ph, pmax = tele.get("player_hp", -1), tele.get("player_max", -1)
            if ph >= 0 and pmax and pmax > 0:
                self._prev_player_frac = ph / pmax
            st = tele.get("soul_total", -1)
            smax = tele.get("soul_max", -1) + tele.get("soul_reserve_max", -1)
            if st is not None and st >= 0 and smax > 0:
                self._prev_soul_frac = st / smax
        def clip01(v):
            return max(0.0, min(1.0, float(v)))
        return np.array([clip01(self._scale), clip01(self._prev_boss_frac),
                         clip01(self._prev_player_frac), clip01(self._prev_soul_frac)],
                        dtype=np.float32)

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
        # 凍結後重同步：推進一個 tick 後仍落後 now 超過一整個 tick = 真凍結，
        # 丟棄積欠、從 now 重新起算，不靠連發(>15Hz)追趕、也免 catch-up 把 log 二次方灌大。
        if now - self._next_t > config.TICK_DT:
            self._next_t = now + config.TICK_DT

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
        curriculum.end_drop()     # 清掉上一場可能留下的遮擋旗標，確保新一場正常計入勝負
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
                # privileged critic：每場重置特權特徵的 backfill 基準（_critic_extra 需 _scale 已設好）
                self._prev_boss_frac = 1.0     # 開場 boss 未生成 → 視為滿血
                self._prev_player_frac = 1.0   # 開場滿血
                self._prev_soul_frac = 0.0     # 開場魂預設 0（首個有效遙測即覆蓋）
                # fps 診斷：時鐘改在第一個 step 才錨定（讓 reset→首次冷推論落在計時外）；此處只清狀態
                self._ep_wall0 = None
                self._over_runs = []
                return self.stacker.get(), {"critic_extra": self._critic_extra(tele)}
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

    def wait_until_unoccluded(self, should_stop=None, log=print):
        """若擷取區被遮擋（其他視窗 / secure desktop），放開輸入並阻塞到解除或收到停止。
        用在 episode 之間（reset 前）當 gate，避免一蓋就連環 drop。回傳是否真的等過。"""
        occ, frac, reason = self.cap.is_occluded()
        if not occ:
            return False
        self.act.release_all()
        log(f"⏸ 偵測到遮擋（{reason}，覆蓋 {frac:.0%}），暫停等待解除...")
        while occ and not (should_stop and should_stop()):
            time.sleep(0.2)
            occ, frac, reason = self.cap.is_occluded()
        if not (should_stop and should_stop()):
            log("▶ 遮擋解除，繼續。")
        return True

    def drain_until_terminal(self, should_stop=None):
        """中途遮擋後：放開輸入、空跑到本場自然結束（角色站著被打死）才回得了大廳。
        不收集 transition、不送輸入；以遙測(monitor)判終止——與畫面遮擋無關。有步數上限保險。"""
        self.act.release_all()
        for _ in range(config.MAX_EPISODE_STEPS):
            if should_stop and should_stop():
                return
            self._wait_tick()
            tele = self._read_tele()
            if self.monitor and self.monitor.update(tele) in ("win", "lose", "left"):
                return

    def step(self, action_vec):
        """action_vec: MultiBinary(11)。回傳 (obs, reward, terminated, truncated, info)。"""
        if self._ep_wall0 is None:              # 本場第一個 step：起算牆鐘 + 錨定 15Hz 時鐘
            self._ep_wall0 = time.perf_counter()
            self._next_t = self._ep_wall0 + config.TICK_DT   # 錨在此；reset→首次冷推論不計入 t0
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
        info["critic_extra"] = self._critic_extra(tele)   # privileged critic 特權特徵（本 tick）
        # 畫面遮擋偵測（T1 幾何重疊 + T2 secure desktop）：本 tick 的 obs 已被覆蓋視窗污染。
        # 放開輸入、設旗標讓 mod 本場不計勝負，回報 occluded 讓呼叫端丟棄整場（並「等輸」回大廳）。
        occ, ofrac, oreason = self.cap.is_occluded()
        if occ:
            self.act.release_all()
            curriculum.begin_drop()
            info["occluded"] = True
            info["occ_frac"] = ofrac
            info["occ_reason"] = oreason
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
        info = {"result": None, "boss_hp_raw": self._prev_boss, "player_hp": self._prev_player,
                "boss_max": -1}
        if tele is None:
            return r, False, info

        boss = tele.get("boss_hp_raw", -1)
        player = tele.get("player_hp", -1)
        if boss >= 0:
            info["boss_hp_raw"] = boss
        if player >= 0:
            info["player_hp"] = player
        info["boss_max"] = tele.get("boss_max", -1)   # mod 的 running-max 滿血；給 BossDamageTracker 當基準

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
