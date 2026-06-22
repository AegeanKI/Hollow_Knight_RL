"""畫面擷取。用 mss 抓遊戲視窗區域，輸出降採樣後的觀測 (uint8)。"""
import cv2
import numpy as np
from mss import mss

import config
from keys import find_window_region


class Capturer:
    def __init__(self, region=None):
        self._sct = mss()
        self.region = region or self._resolve_region()
        print(f"擷取區域: {self.region}")

    def _resolve_region(self):
        r = find_window_region(config.WINDOW_TITLE)
        if r:
            return r
        if config.CAPTURE_REGION:
            return config.CAPTURE_REGION
        # 後備：主螢幕全畫面
        mon = self._sct.monitors[1]
        return {"left": mon["left"], "top": mon["top"],
                "width": mon["width"], "height": mon["height"]}

    def grab_raw(self) -> np.ndarray:
        """抓一張原始畫面，回傳 RGB (H, W, 3) uint8。"""
        img = np.asarray(self._sct.grab(self.region))  # BGRA
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    def grab_obs(self) -> np.ndarray:
        """抓一張並降採樣成觀測尺寸。回傳 (H, W, C) uint8。"""
        rgb = self.grab_raw()
        h, w = config.OBS_SIZE
        obs = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        if config.OBS_GRAYSCALE:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)[..., None]
        return obs


if __name__ == "__main__":
    # 抓一張存檔，讓你確認擷取區域對不對。
    cap = Capturer()
    raw = cap.grab_raw()
    obs = cap.grab_obs()
    cv2.imwrite("capture_raw.png", cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
    out = obs if obs.shape[-1] == 3 else np.repeat(obs, 3, axis=-1)
    cv2.imwrite("capture_obs.png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print(f"raw={raw.shape}  obs={obs.shape}  -> 已存 capture_raw.png / capture_obs.png")
