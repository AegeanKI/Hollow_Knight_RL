"""Behavior Cloning 訓練：從 demo 學 (畫面 -> 按哪些鍵)。

多標籤二元分類：11 個鍵各自一個 sigmoid。用 pos_weight 補償稀有鍵的不平衡。
評估用每鍵的 precision/recall/F1（比 accuracy 有意義，因為大多時候鍵是放開的）。

用法：
  python train_bc.py                 # 預設 30 epoch
  python train_bc.py --epochs 50
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from dataset import DemoDataset, split_files
from model import PolicyNet


def make_pos_weight(pos_rate, cap=20.0):
    p = np.clip(pos_rate, 1e-4, 1 - 1e-4)
    w = np.minimum(cap, (1 - p) / p)   # 正樣本越稀有，權重越大
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device, thr=0.5):
    model.eval()
    n = config.N_ACTIONS
    tp = np.zeros(n); fp = np.zeros(n); fn = np.zeros(n)
    loss_sum, nb = 0.0, 0
    bce = nn.BCEWithLogitsLoss()
    for obs, act in loader:
        obs, act = obs.to(device), act.to(device)
        logits = model(obs)
        loss_sum += bce(logits, act).item(); nb += 1
        pred = (torch.sigmoid(logits) > thr).float()
        tp += (pred * act).sum(0).cpu().numpy()
        fp += (pred * (1 - act)).sum(0).cpu().numpy()
        fn += ((1 - pred) * act).sum(0).cpu().numpy()
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / np.maximum(tp + fn, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-8)
    return loss_sum / max(nb, 1), prec, rec, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_files, val_files = split_files()
    print(f"train episodes: {len(train_files)}  val episodes: {len(val_files)}")
    train_ds, val_ds = DemoDataset(train_files), DemoDataset(val_files)
    print(f"train ticks: {len(train_ds)}  val ticks: {len(val_ds)}")

    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=0, pin_memory=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)

    model = PolicyNet().to(device)
    pos_w = make_pos_weight(train_ds.action_positive_rate()).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(config.CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(config.CKPT_DIR, "bc.pt")
    best_f1 = -1.0

    for ep in range(1, args.epochs + 1):
        model.train()
        tl, nb = 0.0, 0
        for obs, act in train_ld:
            obs, act = obs.to(device, non_blocking=True), act.to(device, non_blocking=True)
            logits = model(obs)
            loss = bce(logits, act)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        vloss, prec, rec, f1 = evaluate(model, val_ld, device)
        # 只對「有出現過的鍵」算 macro-F1
        seen = train_ds.action_positive_rate() > 0
        mf1 = f1[seen].mean()
        print(f"ep {ep:2d} | train_loss {tl/nb:.3f} | val_loss {vloss:.3f} | macroF1 {mf1:.3f}")
        if mf1 > best_f1:
            best_f1 = mf1
            torch.save({
                "model": model.state_dict(),
                "config": {"NET_SIZE": config.NET_SIZE, "FRAME_STACK": config.FRAME_STACK,
                           "NET_CHANNELS": config.NET_CHANNELS, "N_ACTIONS": config.N_ACTIONS,
                           "ACTION_KEYS": config.ACTION_KEYS, "ACTION_NAMES": config.ACTION_NAMES},
                "macroF1": best_f1, "epoch": ep,
            }, ckpt_path)

    # 收尾：印最佳模型的每鍵指標
    print(f"\n最佳 macroF1 = {best_f1:.3f}，存於 {ckpt_path}")
    print("各鍵 (val) precision / recall / F1:")
    for i, name in enumerate(config.ACTION_NAMES):
        print(f"  {name:>11}: P={prec[i]:.2f} R={rec[i]:.2f} F1={f1[i]:.2f}")


if __name__ == "__main__":
    main()
