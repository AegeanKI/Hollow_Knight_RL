"""觀測前處理——訓練與推論共用，確保一致。

單張流程：錄製存的 96x96x3 uint8 -> 降到 NET_SIZE、(可選)灰階 -> CHW uint8。
FrameStacker：推論時維護最近 FRAME_STACK 張，疊成網路輸入。
"""
import cv2
import numpy as np

import config


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """(H,W,3) uint8 RGB -> (C, NET_SIZE, NET_SIZE) uint8。C=1或3。"""
    s = config.NET_SIZE
    img = cv2.resize(frame, (s, s), interpolation=cv2.INTER_AREA)
    if config.OBS_GRAYSCALE and img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[..., None]
    return np.ascontiguousarray(img.transpose(2, 0, 1))  # HWC -> CHW


def stack_to_tensor(stack_chw_uint8: np.ndarray) -> np.ndarray:
    """把疊好的 (FRAME_STACK*C, H, W) uint8 轉成 float32 [0,1]。"""
    return stack_chw_uint8.astype(np.float32) / 255.0


class FrameStacker:
    """推論用：push 最新一張原始幀，get 取得疊好的網路輸入 (float32)。"""
    def __init__(self):
        self.k = config.FRAME_STACK
        self.buf = None

    def reset(self):
        self.buf = None

    def push(self, raw_frame: np.ndarray):
        f = preprocess_frame(raw_frame)
        if self.buf is None:
            self.buf = [f] * self.k          # 第一張：重複填滿
        else:
            self.buf.pop(0)
            self.buf.append(f)

    def get(self) -> np.ndarray:
        return stack_to_tensor(np.concatenate(self.buf, axis=0))
