"""檢視錄好的 episode：印統計 + 把畫面+動作疊字播成影片視窗。

用法：
  python inspect_ep.py data/ep0000.npz      # 播放
  python inspect_ep.py data/ep0000.npz stat # 只印統計
"""
import sys

import cv2
import numpy as np

from config import ACTION_NAMES, TICK_HZ


def main():
    if len(sys.argv) < 2:
        print("用法: python inspect_ep.py <檔案.npz> [stat]")
        return
    d = np.load(sys.argv[1])
    frames, actions = d["frames"], d["actions"]
    n = len(frames)
    print(f"檔案: {sys.argv[1]}")
    print(f"ticks={n}  時長≈{n / TICK_HZ:.1f}s  frames={frames.shape}  actions={actions.shape}")
    # 每個鍵被按住的比例
    usage = actions.mean(axis=0)
    print("各動作按住比例:")
    for k, u in zip(ACTION_NAMES, usage):
        bar = "█" * int(u * 30)
        print(f"  {k:>10}: {u * 100:5.1f}% {bar}")
    tele_ok = np.isfinite(d["boss_hp"]).mean()
    print(f"遙測涵蓋率: {tele_ok * 100:.0f}%  "
          f"(boss_hp 範圍 {np.nanmin(d['boss_hp']):.2f}~{np.nanmax(d['boss_hp']):.2f})"
          if tele_ok > 0 else "遙測涵蓋率: 0% (錄製時 mod 沒開)")

    if len(sys.argv) > 2 and sys.argv[2] == "stat":
        return

    print("\n播放中：空白鍵暫停，q 離開")
    for i in range(n):
        img = frames[i]
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (480, 480), interpolation=cv2.INTER_NEAREST)
        pressed = [k for k, on in zip(ACTION_NAMES, actions[i]) if on]
        cv2.putText(img, " ".join(pressed) or "-", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f"{i}/{n}", (8, 464),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("episode", img)
        key = cv2.waitKey(int(1000 / TICK_HZ)) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
