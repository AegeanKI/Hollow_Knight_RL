# Hollow Knight AI — 純看畫面打神居大黃蜂

目標：訓練 AI **只看遊戲畫面**（不讀記憶體）打贏神居的大黃蜂（調諧級 Attuned）。
管線分三階段：**同步錄製 → 行為複製(BC，可行性驗證) → 強化學習(PPO 從零訓練)**。

> BC 不是 RL 的起點，是**可行性驗證**：證明「同一條 96×96 疊幀觀測 + 11 鍵動作空間 + 15Hz 決策」這套介面真的能讓模型操作角色。驗證過了，戰鬥策略就交給 PPO **從隨機初始化的權重**、靠 reward 自己學起。

> reward 來自一個 C# mod 透過 UDP 回傳的血量遙測——這只當訓練訊號，**不進模型輸入**，所以「只看畫面」的設定不被破壞。

## Demo

PPO agent 純看畫面打調諧級大黃蜂（2026-09-21 錄製），兩種輸入後端：

| 手把後端 (`--input gamepad`) | 鍵盤後端 (`--input keyboard`) |
|:---:|:---:|
| [![PPO - gamepad](https://i.ytimg.com/vi/oYOyt0e61fc/hqdefault.jpg)](https://www.youtube.com/watch?v=oYOyt0e61fc) | [![PPO - keyboard](https://i.ytimg.com/vi/4JnAc4u-Je0/hqdefault.jpg)](https://www.youtube.com/watch?v=4JnAc4u-Je0) |
| [YouTube](https://www.youtube.com/watch?v=oYOyt0e61fc) | [YouTube](https://www.youtube.com/watch?v=4JnAc4u-Je0) |

---

## 環境

- Python：conda 環境 `hk`（`conda create -n hk python=3.11` 後 `conda activate hk`；以下指令都在這個環境下跑）。
- 套件：`pip install -r requirements.txt`（torch / opencv / mss / pynput / pydirectinput / vgamepad …）。
- GPU：RTX 3070（CNN 在 15Hz 預算內推論綽綽有餘）。
- 遊戲：HK 實際生效安裝在 **D 槽**（`D:\...\Hollow Knight`）；mod 要 build/部署到 D 那份才會載入。

---

## 管線與執行順序

### 階段 0：系統閉環（錄製）
先把「擷取 → 動作 → reward → 重開」跑通，並錄人類示範。
```bash
python test_input.py        # 1) 最關鍵：開遊戲站空地，確認程式能讓角色動
python capture.py           # 2) 確認擷取區域：看 capture_raw.png 是否剛好框住畫面
python khook.py             # 3) 確認鍵盤 hook：亂按動作鍵看有沒有即時印出
python telemetry.py send    #    視窗A：灌假遙測；   python telemetry.py 視窗B：看有沒有收到
python record.py            # 4) 同步錄製。F7=開打按(開錄) F8=分勝負按(存檔) F10=離開
python inspect_ep.py data/ep0000.npz   #    檢查錄到的畫面+動作對不對
```

### 階段 1：行為複製 (BC) — 可行性驗證
目的不是產出要拿去打王的策略，而是確認「畫面 → 動作」這條路學得起來、介面沒問題。
```bash
python train_bc.py --epochs 30      # 從 data/ 的 demo 學策略 -> checkpoints/bc.pt（存 macroF1 最佳）
python play_bc.py                   # 載入 bc.pt 實際操作（純看畫面）。F10 停止；F9 暫停
python play_bc.py --threshold 0.4   # 門檻調低 = AI 更願意按鍵（太被動時用）
```

### 階段 2：強化學習 (PPO 從零訓練)
```bash
python env_test.py --episodes 3     # 先驗證 env 自動重開那圈穩定（用 BC 驅動）
python train_rl.py                  # 隨機初始化開始 PPO。F10 安全停止；F9 暫停
python train_rl.py --init-bc        # 選配：改用 bc.pt 熱啟動策略頭（預設不載）
python train_rl.py --resume         # 接續 checkpoints/rl_latest.pt
python train_rl.py --ckpt rl_best.pt        # 從最佳接續（--ckpt 也吃完整路徑）
python eval.py                      # 評估 rl_best.pt（決定性出招）。--latest 看最新
python play_rl.py                   # 看 rl_best.pt 實際打（--latest 看最新）
```
RL 重要行為：
- **預設不載 BC 權重**，策略從隨機初始化學起；`--init-bc` 才會拿 `bc.pt` 熱啟動策略頭。
- **梯度更新在 episode 之間做**（人在雕像大廳、非戰鬥），不搶即時 GPU。
- **rl_best 由決定性 eval 的平均傷害選出**（非訓練取樣 avg_dmg）；`--eval-every` 預設 5、`--eval-episodes` 預設 5。
- **遙測掉太兇的場會被丟棄**（>50%）不納入更新；reset 失敗不會崩，會跳過該場。
- 每次 update 寫一列指標到 `logs/metrics.csv`，方便畫曲線/比較。

---

## 檔案速查

### 設定與共用基礎
| 檔案 | 用途 |
|---|---|
| `config.py` | 集中設定：動作鍵(Action Enum)、tick 頻率(15Hz)、觀測尺寸、reward 權重(RW_*)、遙測 port、路徑等 |
| `keys.py` | 動作編碼/解碼（鍵集合 ↔ MultiBinary 向量）+ 用 Win32 定位遊戲視窗區域 |
| `obs.py` | 觀測前處理（96→`NET_SIZE`，現 **96**＝不再降採樣、可選灰階、CHW、float[0,1]）+ `FrameStacker` 疊幀。**訓練/推論共用，確保一致** |

### 擷取與輸入
| 檔案 | 用途 |
|---|---|
| `capture.py` | 用 mss 抓遊戲視窗，`grab_raw()` 原圖 / `grab_obs()` 降採樣觀測 |
| `inputs.py` | 輸入執行器：`Actuator`(pydirectinput 鍵盤) 與 `GamepadActuator`(vgamepad 虛擬手把)，介面相同可互換；`make_actuator(backend)` |
| `khook.py` | 背景鍵盤 hook，維護「目前按住哪些動作鍵」（錄製時取樣人類輸入） |

### 遙測（reward 來源）
| 檔案 | 用途 |
|---|---|
| `telemetry.py` | UDP 接收 mod 送來的 boss/玩家血量；附 `send` 假資料 sender 供離線測 pipeline |
| `mod/HKReward/` | **C# reward mod**（`RewardMod.cs` + `HKReward.csproj`）：遊戲內每 1/30s 用 UDP 送血量 JSON。用 dotnet SDK 8 編 net472，build 後自動 Copy 到 D 槽的 `Mods/`（覆蓋前要先關遊戲） |

### 模型
| 檔案 | 用途 |
|---|---|
| `model.py` | `PolicyNet`：BC 用的 Nature-CNN，輸出 11 鍵各自 logit（多標籤二元）；`act()` 含對向鍵互斥處理 |
| `ac_model.py` | `ActorCritic`：PPO 用，共用 CNN 分策略/價值頭；動作為 11 個獨立 Bernoulli。`init_from_bc()` 是選配熱啟動（`--init-bc`），預設不用 |
| `dataset.py` | `DemoDataset` 載入 demo 配對 (疊幀觀測, 動作)；`split_files()` 切 train/val；疊幀不跨 episode 邊界 |

### 階段 0 — 錄製
| 檔案 | 用途 |
|---|---|
| `record.py` | **同步錄製主迴圈**：單一 15Hz 時鐘同時取樣 畫面/按鍵/遙測，存 `data/*.npz`。F7 開錄、F8 存檔、F10 離開 |
| `inspect_ep.py` | 檢視錄好的 episode：印統計或把畫面+動作疊字播成影片 |
| `test_input.py` | 驗證輸入注入：開遊戲站空地，跑這個看角色會不會動 |

### 階段 1 — BC
| 檔案 | 用途 |
|---|---|
| `train_bc.py` | BC 訓練（可行性驗證）：多標籤二元(11 sigmoid)，pos_weight 補稀有鍵，每鍵 P/R/F1 評估；存 macroF1 最佳到 `checkpoints/bc.pt` |
| `play_bc.py` | 載入 bc.pt 實際操作遊戲（純看畫面）。F10 停止；F9 暫停 |

### 階段 2 — RL
| 檔案 | 用途 |
|---|---|
| `env.py` | `HollowKnightEnv`：Gym 風格 reset/step。觀測=疊幀畫面、reward=遙測血量變化；reset 自動重開且**失敗不崩**(回 None)，每場統計遙測健康度 |
| `ppo.py` | `RolloutBuffer`(GAE，正確區分 truncate/terminal) + `ppo_update()` + `RunningMeanStd`（① return-std 正規化：用 raw return 的跑動 std 縮放 reward，穩住 critic、與 curriculum scale 解耦）。`vf_coef=0.25`。與環境解耦，可離線測 |
| `train_rl.py` | PPO 主訓練：隨機初始化起步（`--init-bc` 可選配 BC 熱啟動）、episode 間更新、eval 選 best、CSV 指標、遙測丟棄、F10 停 F9 暫停 |
| `eval.py` | 評估某 checkpoint 的真實實力（決定性出招），印勝率/平均/最高傷害 |
| `play_rl.py` | 載入 RL checkpoint 實際打給你看（決定性） |
| `env_test.py` | 用 BC 驅動驗證 env 自動重開那圈是否穩定（含實測 Hz） |

### 階段 2 — 自適應難度（A1 curriculum，進行中）
讓卡在 plateau 的 agent 先在弱化版打贏、收集勝利訊號(`RW_WIN`)，再隨勝率漸進調回 100%。
> **reward 與難度耦合（作法A）**：低 scale 時 boss 掉血被放大，為免 agent 鑽「弱化版照領滿額獎賞」的漏洞，reward 端把 boss 傷害 `×scale` 還原真實傷害、勝利 bonus `×scale`、敗北 penalty `÷scale`。詳見 `config.py` 的 `RW_*` 與 `env.py`。
| 檔案 | 用途 |
|---|---|
| `mod/HKCurriculum/` | **C# 難度 mod**：依滑動勝率(視窗 30 場)自動調 scale **0.5~1.0**。**作法D**＝hook `HealthManager.TakeDamage` 把對 boss 的傷害 ×1/scale → 整場等比壓縮(所有 phase 都在、各自縮短)、boss 在打出 scale×滿血的真實傷害時死。勝負偵測/自適應/持久化(state+log)內建 |
| `curriculum.py` | Python↔mod 檔案交握：`begin_eval/end_eval`(eval 強制 100%、不放大、不計入)、`read_scale`(讀目前難度寫進 log/CSV)、`effective_scale`(本場 reward 正規化用的 scale；eval 固定 1.0) |

### 自動開場
| 檔案 / 資料 | 用途 |
|---|---|
| `autostart.py` | 自動開場/重開：等回大廳→叫醒→開菜單→**難度校正(調諧級)**→確認 boss 出現；含 `EpisodeMonitor` 判勝敗。`python autostart.py` 有離線難度偵測自我測試 |
| `data/target_difficulty.png` | 難度校正的參考圖（調諧級被選時的全畫面截圖，1084×605）。`autostart` 用它比對目前選中哪一級 |

### 手把 / 診斷工具
| 檔案 | 用途 |
|---|---|
| `gamepad_setup.py` | 建立虛擬手把並互動送出單一按鈕，方便在 HK 手把設定裡逐一綁定動作 |
| `keep_pad.py` | 只是保持虛擬 Xbox 手把連接（給 HidHide / 遊戲設定能選到它）。Ctrl+C 結束 |
| `diag_input_hook.py` | 診斷：低階鍵盤 hook 抓「被注入的按鍵」，用來查 keymapper 把手把翻成鍵盤的問題（即時寫 `diag_input_hook.log`） |
| `gae_truncate_test.py` | 離線單元測試：驗證 `RolloutBuffer.finish()` 的 truncate/terminal/中途切斷三種 GAE 收尾正確 |
| `view_obs.py` | 表徵診斷：從 demo npz 以 `M×M×N`(參數可調)顯示連續幀，目視判斷某解析度/幀數下分不分得出 boss 招式。`←→`滑幀、`g`灰階。（用它定出 NET_SIZE 64→96） |

---

## 輸出檔案
| 路徑 | 內容 |
|---|---|
| `data/epXXXX.npz` (+ `_events.json`) | 錄製的 demo：畫面、動作、遙測、原始按鍵事件 |
| `checkpoints/bc.pt` | BC 最佳模型（可行性驗證產物；RL 預設不載，僅 `env_test.py` / `--init-bc` 會用） |
| `checkpoints/rl_latest.pt` / `rl_best.pt` / `rl_uXXXX.pt` | RL 最新 / 最佳(由 eval 選) / 編號快照 |
| `logs/train_rl.log` | RL 訓練文字 log |
| `logs/metrics.csv` | 每次 update 一列指標（train/eval 傷害、勝場、loss、entropy、kl、遙測掉包率、難度 scale），用來畫曲線 |
| `.claude/progress.md` | 當前進度快照（在做什麼、改了哪些檔、待辦、訓練現況） |

> 大型/可重生產物（`data/*.npz`、`checkpoints/`、`logs/`、除錯圖）已在 `.gitignore` 排除，不進版控。

---

## 操作鍵備忘
- **錄製 (`record.py`)**：F7 開錄、F8 存檔、F10 離開。
- **訓練/評估/推論 (`train_rl`/`eval`/`play_rl`/`play_bc`)**：**F10 安全停止**、**F9 暫停/繼續**（暫停在 episode 之間生效，會放開輸入，可開 HK 選單檢查設定）。
- **Esc 留給 HK 選單**，腳本一律不用 Esc。
- 用 gamepad 後端時：訓練前景請保持在遊戲視窗或中性視窗，**勿 focus 在 PowerShell / 檔案總管**（會觸發某 keymapper 亂注入）。
