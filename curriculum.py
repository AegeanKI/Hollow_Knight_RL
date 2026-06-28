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
