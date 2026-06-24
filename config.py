"""集中設定檔。所有模組共用，避免魔術數字散落各處。"""
import os
import tempfile
from enum import Enum

# ---- 動作空間 ----------------------------------------------------------------
# 用 Enum 把「動作語意」與「實體按鍵」解耦：
#   成員名稱 = 動作的意義（JUMP、DASH…），方便閱讀
#   成員的值 = 真正送進遊戲的實體鍵（換鍵只改這裡的 value）
# 定義順序 = MultiBinary 向量的 index 順序。千萬不要改順序或插入中間，否則舊資料會對不上。
class Action(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    JUMP = "z"
    ATTACK = "x"
    DASH = "c"
    CAST = "v"
    SUPER_DASH = "s"
    FOCUS = "a"
    DREAM_NAIL = "d"


# 輸入後端預設："keyboard"(需視窗焦點) 或 "gamepad"(虛擬手把，背景也可、解放鍵盤)
INPUT_BACKEND = "gamepad"

ACTIONS = list(Action)                       # 依定義順序，決定 MultiBinary index
ACTION_KEYS = [a.value for a in ACTIONS]     # 對應的實體鍵（同順序）
ACTION_NAMES = [a.name for a in ACTIONS]     # 語意名稱（給 log/檢視顯示）
KEY_INDEX = {k: i for i, k in enumerate(ACTION_KEYS)}  # 實體鍵 -> MultiBinary index
N_ACTIONS = len(ACTIONS)

# 方向鍵集合（用 Action 取值，與動作定義同源，勿硬寫字串）。
# 鍵盤後端送鍵時「方向先就位、再動其他動作鍵」，確保如旋風斬/蓄力斬這類
# 「放開攻擊瞬間讀方向」的劍技，attack edge 時讀到的是本 tick 的目標方向。
DIRECTION_KEYS = frozenset(a.value for a in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT))

# ---- 時序 --------------------------------------------------------------------
# 控制 tick 頻率（Hz）。這是整個系統唯一的時鐘：擷取、按鍵取樣、reward 取樣都跟它對齊。
# 15Hz = 每 66.7ms 一個決策。先用 15，之後可調。
TICK_HZ = 15
TICK_DT = 1.0 / TICK_HZ
# 互動腳本（訓練/評估/推論/測試）開始前的倒數秒數，讓你切換到遊戲視窗取得焦點
START_COUNTDOWN_SEC = 5

# ---- 畫面擷取 ----------------------------------------------------------------
# 遊戲視窗標題（用來自動定位擷取區域）。找不到就用 CAPTURE_REGION 後備。
WINDOW_TITLE = "Hollow Knight"
# 後備擷取區域（找不到視窗時用）。None 表示抓主螢幕全畫面。
# 格式: {"left": int, "top": int, "width": int, "height": int}
CAPTURE_REGION = None

# 存進資料集的觀測尺寸 (H, W)。保留一點餘裕，訓練時再降到 64x64。
OBS_SIZE = (96, 96)
OBS_GRAYSCALE = False  # False=RGB(3ch)，大黃蜂的紅色對辨識有幫助，先留彩色

# ---- RL reward ---------------------------------------------------------------
# 每 tick reward = 造成傷害*RW_DMG - 掉血*RW_HIT；終局再加勝/敗 bonus。
# 量級設計：打掉全部 900 血 ≈ +9，掉光 9 面具 ≈ -9，勝 +10，敗 -5。可調。
#
# curriculum 難度耦合（作法 A，2026-06-24）：mod 在 scale<1 時把對 boss 的傷害放大
# 1/scale，使 boss_hp_raw 的跌幅(=這裡的 dmg)被等比放大。為避免「低難度照領滿額傷害
# reward / 用滿額勝利 bonus 鑽弱化版漏洞」，把 **boss 傷害相關項乘上 scale 還原成真實
# 傷害**、勝利 bonus 乘 scale、敗北 penalty 除以 scale（贏弱化版不值錢、輸弱化版更痛）。
# scale 由 env 每場開場讀 curriculum.effective_scale()（eval 場固定 1.0，mod 不放大）。
# 掉血 penalty(RW_HIT) 不乘 scale：被打的難度與 boss 血量放大無關。
RW_DMG = 0.02          # 每點 boss 真實血（×scale 還原；打掉真實 900 血 = +18）
RW_HIT = 0.5           # 每個面具（掉 1 面具 = -0.5）
RW_WIN = 15.0          # 勝利 bonus（實得 = RW_WIN×scale：贏滿血 +15、贏 0.6 +9）
RW_LOSE = 3.0          # 敗北 penalty（實扣 = RW_LOSE/scale，取負；輸滿血 -3、輸 0.6 -5）
RW_LOSE_CAP = 6.0      # 敗北 penalty 上限（防 scale 過低時懲罰爆炸/變龜縮；0.5→6 剛好不觸頂）
RW_TIME = 0.0          # 每步時間懲罰（先 0，需要時設小負值催它快點打）
MAX_EPISODE_STEPS = 1500   # 截斷上限（~100s），避免卡住

# ---- RL 訓練 ----
EPISODES_PER_UPDATE = 8    # 收集幾場才做一次梯度更新（在 episode 之間更新，不搶即時 GPU）

# ---- 自動開場 ----------------------------------------------------------------
# 大黃蜂在神居的場景名（守護者 / 哨衛）。用來判斷「戰鬥場景是否已載入」。
HORNET_SCENES = ("GG_Hornet_1", "GG_Hornet_2")
# 神居雕像大廳場景名。死亡/勝利後會切回這裡。
HALL_SCENE = "GG_Workshop"

# ---- 遙測 (reward mod) --------------------------------------------------------
# 之後寫的 C# mod 會用 UDP 把 boss/玩家血量打到這個 port。
TELEMETRY_HOST = "127.0.0.1"
# 注意：51789 落在 Windows UDP 排除範圍(Hyper-V/WSL 動態保留 49152–65535 內)會導致
# bind 出現 WinError 10013。改用 < 49152 的固定埠，永遠不會被動態保留。
TELEMETRY_PORT = 48789
# 遙測新鮮度門檻：sample 距今超過這秒數就視為「沒有有效遙測」（reward/狀態判斷共用）
TELE_FRESH_SEC = 1.0

# ---- 自適應難度 mod (HKCurriculum) 的檔案交握 -------------------------------
# eval 前 Python 建立此旗標 -> mod 該場 boss 用 100% 滿血且不計入自適應；eval 後刪除。
CURRICULUM_EVAL_FLAG = os.path.join(tempfile.gettempdir(), "hk_curriculum_eval.flag")
# mod 每場寫出目前難度比例(0.6~1.0)，Python 讀來記進 metrics.csv（分辨「變強」vs「變簡單」）。
CURRICULUM_SCALE_FILE = os.path.join(tempfile.gettempdir(), "hk_curriculum_scale.txt")

# ---- 資料存放 ----------------------------------------------------------------
DATA_DIR = "data"          # 錄製的 demo 存這裡
EPISODE_PREFIX = "ep"      # 檔名前綴
CKPT_DIR = "checkpoints"   # 模型權重存這裡
# checkpoint 檔名（都在 CKPT_DIR 底下；多個腳本共用，避免各自硬寫）
BC_CKPT = "bc.pt"              # 階段1 BC 最佳模型
RL_LATEST_CKPT = "rl_latest.pt"   # 階段2 RL 最新（每次更新覆蓋）
RL_BEST_CKPT = "rl_best.pt"       # 階段2 RL 最佳（由決定性 eval 選）

# ---- 網路輸入 ----------------------------------------------------------------
# 進網路前把 96x96 觀測再降到 NET_SIZE，並疊 FRAME_STACK 張連續幀（給速度/方向資訊）。
# 推論與訓練必須用同一組設定（共用 obs.py 的前處理）。
NET_SIZE = 64        # 網路輸入邊長
FRAME_STACK = 4      # 疊幾張連續幀
# 輸入通道數：RGB=3 -> 3*FRAME_STACK；灰階=1 -> FRAME_STACK
NET_CHANNELS = (1 if OBS_GRAYSCALE else 3) * FRAME_STACK
