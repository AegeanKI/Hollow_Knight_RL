"""與 HKCurriculum mod 的檔案交握。

- 評估時要求 boss 用 100% 滿血（否則 EVAL 會被自適應難度污染，分不清變強還是變簡單）。
- 讀 mod 寫出的目前難度比例，記進訓練指標。
路徑見 config.CURRICULUM_*；mod 不在時這些操作都安全 no-op。
"""
import os

import config


def begin_eval():
    """建立 eval 旗標：mod 看到後該場用 100% 滿血且不計入自適應。"""
    try:
        open(config.CURRICULUM_EVAL_FLAG, "w").close()
    except OSError:
        pass


def end_eval():
    """移除 eval 旗標，恢復自適應難度（沒有旗標也安全）。"""
    try:
        os.remove(config.CURRICULUM_EVAL_FLAG)
    except OSError:
        pass


def begin_drop():
    """偵測到畫面遮擋時建立旗標：mod 看到後該場 win/lose 不計入自適應勝率。
    與 begin_eval 同機制（檔案交握），但不強制 100% 血量——只「不計入」。"""
    try:
        open(config.CURRICULUM_DROP_FLAG, "w").close()
    except OSError:
        pass


def end_drop():
    """移除遮擋旗標（沒有也安全）。每場 reset 前清掉，確保新一場正常計入。"""
    try:
        os.remove(config.CURRICULUM_DROP_FLAG)
    except OSError:
        pass


def _read_finale():
    """讀 mod 寫的殘局握手檔，回 {finale:bool, armed:bool, true_max:int, boss_frac:float}；讀不到回 None。

    boss_frac＝殘局 boss 起始血占真實滿血的比例（mod 降血時抽的值，armed 時才有效；正常/未 armed=-1）。
    """
    try:
        d = {}
        with open(config.CURRICULUM_FINALE_FILE) as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    d[k] = v
        return {"finale": d.get("finale") == "1",
                "armed": d.get("armed") == "1",
                "true_max": int(d.get("true_max", "-1")),
                "boss_frac": float(d.get("boss_frac", "-1"))}
    except (OSError, ValueError):
        return None


def clear_finale():
    """reset 前清掉上一場的殘局握手檔（避免讀到舊狀態）。沒有也安全。"""
    try:
        os.remove(config.CURRICULUM_FINALE_FILE)
    except OSError:
        pass


def wait_for_armed(should_stop=None, timeout=None):
    """殘局握手（只在 config.FINALE_ENABLED 時由 env.reset 呼叫）。回 (is_finale, true_max, boss_frac)。

    流程（mod 在 fight-detect 才寫檔，故有 race，要等）：
      - 等 mod 寫出握手檔。
      - finale=0 → 正常場，立即回 (False, -1, -1)。
      - finale=1 → 等 armed=1（mod 設好殘局 HP/位置/過完開場）才回 (True, true_max, boss_frac)。
      - 逾時/讀不到（mod 舊版未寫、未載入）→ 當正常場 (False, -1, -1)，安全降級。

    boss_frac＝殘局 boss 起始血占真實滿血比例；目前僅供 env 的 ARMED 遙測 log（曾用來縮殘局贏分，
    已於 4c0de8b 還原成全額，不再進 reward）。舊版 mod 未送此欄 → -1。
    """
    import time
    if timeout is None:
        timeout = config.FINALE_ARM_TIMEOUT
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if should_stop and should_stop():
            return False, -1, -1.0
        st = _read_finale()
        if st is not None:
            if not st["finale"]:
                return False, -1, -1.0
            if st["armed"]:
                return True, st["true_max"], st["boss_frac"]
        time.sleep(0.05)
    return False, -1, -1.0   # 逾時：安全當正常場跑


def read_scale():
    """讀目前難度比例(0.5~1.0)；mod 沒載入/讀不到回 None。"""
    try:
        with open(config.CURRICULUM_SCALE_FILE) as f:
            return float(f.read().strip())
    except (OSError, ValueError):
        return None


def effective_scale():
    """本場 reward 正規化要用的 scale（作法 A）。

    eval 場 mod 固定 100% 不放大 → 因子 1.0（不可誤用訓練 scale，否則真實傷害會被
    再乘一次 scale 而低估）。訓練場用 read_scale()；mod 沒載入/讀不到也回 1.0
    （= 不放大、reward 不變，安全降級）。
    """
    if os.path.exists(config.CURRICULUM_EVAL_FLAG):
        return 1.0
    s = read_scale()
    return s if s is not None else 1.0
