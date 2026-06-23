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


def read_scale():
    """讀目前難度比例(0.6~1.0)；mod 沒載入/讀不到回 None。"""
    try:
        with open(config.CURRICULUM_SCALE_FILE) as f:
            return float(f.read().strip())
    except (OSError, ValueError):
        return None
