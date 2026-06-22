# Hollow Knight AI — 階段 0：系統閉環

目標：用視覺（只看畫面）訓練 AI 打大黃蜂。階段 0 先把「擷取 → 動作 → reward → 重開」的
系統迴圈打通，**還不做學習**。

## 模組

| 檔案 | 作用 |
|---|---|
| `config.py` | 集中設定（動作鍵、tick 頻率、擷取區域、遙測 port…） |
| `keys.py` | 動作編碼/解碼 + 視窗定位 |
| `inputs.py` | 輸入注入（讓程式按鍵進遊戲） |
| `capture.py` | 畫面擷取 + 降採樣成觀測 |
| `khook.py` | 背景鍵盤 hook，維護「目前按住的鍵」 |
| `telemetry.py` | UDP 接收 reward mod 的 boss/玩家血量（含假資料 sender） |
| `record.py` | **同步錄製主迴圈**（階段 0 核心），之後會演化成訓練/推論環境 |
| `inspect_ep.py` | 檢視/播放錄好的 episode |
| `test_input.py` | 驗證程式能不能讓角色動 |

## 建議執行順序

```bash
# 1) 【最關鍵】驗證輸入注入：開遊戲站在空地，跑這個看角色會不會動
python test_input.py

# 2) 驗證擷取區域：開遊戲，跑這個，看 capture_raw.png 是不是剛好框住遊戲畫面
python capture.py

# 3) 驗證鍵盤 hook：跑這個然後亂按 10 個動作鍵，看有沒有即時印出
python khook.py

# 4) 驗證遙測 pipeline（mod 還沒寫好，先用假資料）：
python telemetry.py send      # 視窗 A：灌假資料
python telemetry.py           # 視窗 B：看有沒有收到

# 5) 同步錄製（mod 還沒好也能錄，只是 reward 欄位會是空的）：
#    只圈「戰鬥本身」：選單/選難度/等怪都不要錄。
python record.py             # F7=開打時按(開始錄), F8=分勝負時按(存檔), Esc=離開
python inspect_ep.py data/ep0000.npz   # 檢查錄到的畫面+動作對不對

## 階段 1：Behavior Cloning

```bash
python train_bc.py --epochs 30     # 從 data/ 的 demo 學策略，存到 checkpoints/bc.pt
python play_bc.py                  # 載入模型實際操作遊戲（純看畫面）。Esc 停止
python play_bc.py --threshold 0.4  # 門檻調低 -> AI 更願意按鍵（太被動時用）
```

模組：`obs.py`(前處理) `dataset.py`(資料集) `model.py`(CNN策略) `train_bc.py` `play_bc.py`

## 階段 2：RL（PPO 從 BC 微調）

```bash
python env_test.py --episodes 3    # 先驗證 env 自動重開那圈穩定（用 BC 驅動）
python train_rl.py                 # 從 bc.pt 初始化開始 PPO 訓練。Esc 安全停止
python train_rl.py --resume        # 接續 checkpoints/rl_latest.pt
python play_rl.py                  # 看 rl_best.pt 實際打（--latest 看最新）
```

模組：`env.py`(環境) `ac_model.py`(Actor-Critic) `ppo.py`(PPO) `train_rl.py` `play_rl.py`
reward 與訓練參數在 `config.py` 的 RL 區段（RW_*、EPISODES_PER_UPDATE）。

## 下一步
- 寫 reward 遙測 mod（C#），把 telemetry 的 boss_hp/player_hp 換成真實值
- 自動重開（偵測死亡/勝利 → 走回雕像重新挑戰）
- 把 record.py 改造成 Gymnasium 環境
```
