"""從 demo npz 讀畫面，以 M×M 解析度顯示連續 N 幀，用來「目視判斷表徵夠不夠分辨 boss 招式」。

忠實重現網路所見：用 obs.py 同款 cv2.resize(INTER_AREA) 降到 M×M、可選灰階，
並用 interpolation='nearest' 顯示，看得到真實的方塊感（不被顯示端平滑騙過去）。

用法：
  python view_obs.py --ep data/ep0000.npz --size 96 --nframes 4
  python view_obs.py --ep 3 --size 64 --nframes 4      # 模擬現在網路看到的 64x64x4
  python view_obs.py --size 96 --nframes 8             # 評估 ③ 的候選 96x96x8

鍵盤：
  → / ←   視窗滑動 1 幀（ABCD →右→ BCDE，←左→ 回去）
  ↑ / ↓   滑動 N 幀（翻頁）
  Home/End 跳到最前/最後
  g        切換灰階（模擬 OBS_GRAYSCALE）
  q / Esc  離開
"""
import argparse
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use(os.environ.get("MPLBACKEND", "TkAgg"))   # 預設互動；煙霧測試可用 MPLBACKEND=Agg 覆蓋
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "JUMP", "ATTACK",
                "DASH", "CAST", "SUPER_DASH", "FOCUS", "DREAM_NAIL"]


def resolve_ep(ep):
    """--ep 可給完整路徑或編號（3 -> data/ep0003.npz）。"""
    if ep is None:
        return os.path.join("data", "ep0000.npz")
    if os.path.exists(ep):
        return ep
    if ep.isdigit():
        return os.path.join("data", f"ep{int(ep):04d}.npz")
    raise FileNotFoundError(f"找不到 {ep}")


def resize_frame(frame, M, gray):
    """(96,96,3) uint8 -> (M,M[,3])。降採樣用 INTER_AREA（同 obs.py），放大用 LINEAR。"""
    interp = cv2.INTER_AREA if M <= frame.shape[0] else cv2.INTER_LINEAR
    img = cv2.resize(frame, (M, M), interpolation=interp)
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


class Viewer:
    def __init__(self, epname, frames, actions, t, M, N, start, gray):
        self.epname = epname
        self.frames, self.actions, self.t = frames, actions, t
        self.M, self.N, self.start, self.gray = M, N, start, gray
        self.T = len(frames)
        self.fig, axes = plt.subplots(1, N, figsize=(2.2 * N, 3.0))
        self.axes = [axes] if N == 1 else list(axes)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.draw()

    def on_key(self, e):
        k = e.key
        if k == "right":
            self.start = min(self.start + 1, self.T - self.N)
        elif k == "left":
            self.start = max(self.start - 1, 0)
        elif k == "up":
            self.start = min(self.start + self.N, self.T - self.N)
        elif k == "down":
            self.start = max(self.start - self.N, 0)
        elif k == "home":
            self.start = 0
        elif k == "end":
            self.start = max(self.T - self.N, 0)
        elif k == "g":
            self.gray = not self.gray
        elif k in ("q", "escape"):
            plt.close(self.fig)
            return
        else:
            return
        self.draw()

    def draw(self):
        for i, ax in enumerate(self.axes):
            ax.clear()
            ax.axis("off")
            fi = self.start + i
            if fi >= self.T:
                continue
            img = resize_frame(self.frames[fi], self.M, self.gray)
            ax.imshow(img, cmap="gray" if self.gray else None,
                      interpolation="nearest", vmin=0, vmax=255)
            acts = [ACTION_NAMES[j] for j in range(len(ACTION_NAMES)) if self.actions[fi][j]]
            ax.set_title(f"{chr(ord('A') + i)}  #{fi}  t={self.t[fi]:.2f}s\n{'+'.join(acts) or '—'}",
                         fontsize=8)
        end = min(self.start + self.N - 1, self.T - 1)
        self.fig.suptitle(
            f"{os.path.basename(self.epname)}   {self.M}x{self.M}x{self.N}"
            f"{'(灰)' if self.gray else '(RGB)'}   幀 {self.start}..{end}/{self.T - 1}"
            f"    [←→滑1  ↑↓滑{self.N}  g灰階  q離開]", fontsize=10)
        self.fig.canvas.draw_idle()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", default=None, help="npz 路徑或編號（如 3）")
    ap.add_argument("--size", "-M", type=int, default=96, help="顯示解析度 M（M×M）")
    ap.add_argument("--nframes", "-N", type=int, default=4, help="同時顯示幾幀 N")
    ap.add_argument("--start", type=int, default=0, help="起始幀")
    ap.add_argument("--gray", action="store_true", help="灰階（模擬 OBS_GRAYSCALE）")
    args = ap.parse_args()

    path = resolve_ep(args.ep)
    d = np.load(path)
    frames, actions, t = d["frames"], d["actions"], d["t"]
    if args.size > frames.shape[1]:
        print(f"⚠ M={args.size} > demo 原生 {frames.shape[1]}，會放大（無新資訊）。要更高解析度需重錄。")
    print(f"{os.path.basename(path)}: {len(frames)} 幀，原生 {frames.shape[1]}x{frames.shape[2]} RGB")
    print(f"顯示 {args.size}x{args.size}x{args.nframes}。←→滑1 ↑↓滑{args.nframes} g灰階 q離開")
    # 清掉會跟我們按鍵衝突的預設 keymap（g=grid、左右=瀏覽歷史、home/s/f 等）；保留 quit 讓 q 仍能關
    for km in ("keymap.back", "keymap.forward", "keymap.grid", "keymap.grid_minor",
               "keymap.home", "keymap.save", "keymap.fullscreen", "keymap.yscale", "keymap.xscale"):
        try:
            plt.rcParams[km] = []
        except KeyError:
            pass
    # 必須保住強引用：mpl_connect 對 bound method 存弱引用，不接住 viewer 會被 GC → 鍵盤失效
    viewer = Viewer(path, frames, actions, t, args.size, args.nframes,
                    max(0, min(args.start, len(frames) - args.nframes)), args.gray)
    plt.show()
    return viewer


if __name__ == "__main__":
    main()
