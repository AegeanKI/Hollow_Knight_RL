"""自動開場：站在大黃蜂雕像前 -> 開挑戰菜單 -> 選調諧級 -> 等戰鬥載入。

流程：
  1. 連續點 UP，直到畫面出現大變化（挑戰菜單疊上來）就停手——避免多按 UP 改到難度選項。
  2. 校正難度：用畫面偵測目前選中哪一級，若不是調諧級就按 UP（調諧級在最上）直到選中它。
  3. 點 Z 確認難度。
  4. 等遙測回報「已進入大黃蜂場景且 boss 出現」才算開打（比盯白畫面可靠）。
     若沒有遙測（mod 沒開），退而用白畫面消失當作載入完成。

這段是「流程控制」，用遙測不影響 AI 純看畫面的設定。之後 RL 自動重開會復用它。
"""
import os
import time

import cv2
import numpy as np

import config
from config import Action

# --- 可調參數 ---
MENU_OPEN_THRESH = 7.0    # 按 UP 後畫面平均像素差 > 此值 => 菜單開始出現（一偵測到就停按 UP）
SETTLE_DIFF = 2.5         # 連續兩幀差異 < 此值 => 畫面已靜止（菜單動畫跑完）
WHITE_THRESH = 225.0      # 畫面平均亮度 > 此值 => 視為白色轉場中
TAP_HOLD = 0.07           # 點按持續時間
START_TIMEOUT = 8.0       # 單次等開打的最長秒數（縮短以便快速重試）
LYING_WAIT = 2.5          # 回大廳後再等這麼久，讓地上躺臥期過、角色可操控


def _tap(act, key, hold=TAP_HOLD):
    act.apply_keys({key})
    time.sleep(hold)
    act.apply_keys(set())


def _framediff(a, b):
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def _wait_settle(cap, timeout=2.0):
    """等畫面靜止（菜單動畫跑完）。"""
    prev = cap.grab_obs()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        time.sleep(0.15)
        cur = cap.grab_obs()
        if _framediff(cur, prev) < SETTLE_DIFF:
            return
        prev = cur


# --- 難度校正（畫面偵測選中的難度，必要時按 UP 移到調諧級）-------------------
# 原理：挑戰菜單裡選中的那一列，左右各有一個亮的裝飾游標。沒選中的列沒有游標。
# 我們用參考圖（data/target_difficulty.png，調諧級被選時的全畫面截圖）裁出「右側游標」
# 當模板，在右側直條（涵蓋三個難度列）裡 matchTemplate 找游標，由其「垂直位置」判斷
# 目前選中哪一級。座標都換算到參考圖尺寸，故對視窗大小變化有韌性。
DIFFICULTY_REF = os.path.join(config.DATA_DIR, "target_difficulty.png")
_REF_W, _REF_H = 1084, 605           # 參考圖尺寸（座標基準）
_CURSOR_TPL_BOX = (262, 292, 935, 985)   # 右側游標模板 (y0,y1,x0,x1)
_CURSOR_STRIP_BOX = (240, 420, 920, 1000)  # 三列右側搜尋帶 (y0,y1,x0,x1)
_ATTUNED_CURSOR_Y = 277              # 調諧級游標中心 y（參考座標）
_SELECT_Y_TOL = 26                  # 容差（約半個列距）：|cy-ATTUNED_Y|<=此值 視為調諧級
_CURSOR_MATCH_THRESH = 0.5          # 模板比對分數門檻；低於此視為「找不到游標」
_cursor_tpl = None                  # 延遲載入的模板快取


def _load_cursor_template():
    global _cursor_tpl
    if _cursor_tpl is None:
        ref = cv2.imread(DIFFICULTY_REF)  # BGR
        if ref is None:
            raise FileNotFoundError(
                f"找不到難度參考圖：{DIFFICULTY_REF}（難度校正需要它）")
        if ref.shape[:2] != (_REF_H, _REF_W):
            ref = cv2.resize(ref, (_REF_W, _REF_H))
        g = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        y0, y1, x0, x1 = _CURSOR_TPL_BOX
        _cursor_tpl = g[y0:y1, x0:x1]
    return _cursor_tpl


def detect_difficulty(cap):
    """偵測挑戰菜單目前選中的難度。

    回傳 (label, score)：
      label = "attuned" 選中調諧級 / "lower" 選中較高難度 / None 找不到游標。
      score = 模板比對分數（debug 用）。
    """
    tpl = _load_cursor_template()
    frame = cv2.resize(cap.grab_raw(), (_REF_W, _REF_H))  # 對齊參考座標
    g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    y0, y1, x0, x1 = _CURSOR_STRIP_BOX
    strip = g[y0:y1, x0:x1]
    res = cv2.matchTemplate(strip, tpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    if score < _CURSOR_MATCH_THRESH:
        return None, score
    cy = y0 + loc[1] + tpl.shape[0] // 2           # 游標中心 y（參考座標）
    label = "attuned" if abs(cy - _ATTUNED_CURSOR_Y) <= _SELECT_Y_TOL else "lower"
    return label, score


def ensure_attuned(cap, act, max_up=4):
    """確保選中調諧級：偵測目前難度，不是就按 UP（調諧級在最上）直到選中它。

    迴圈以「偵測結果」為閘門：一旦確認調諧級就立刻停手，所以即使選單會循環
    （頂端再按 UP 跳到底）也不會選過頭。回傳是否確認停在調諧級。
    """
    for i in range(max_up + 1):
        label, score = detect_difficulty(cap)
        if label == "attuned":
            tag = f"（按了 {i} 次 UP 校正）" if i else f"（score={score:.2f}）"
            print(f"  [難度] 確認調諧級{tag}")
            return True
        if label is None:
            print(f"  [難度] ⚠ 偵測不到難度游標（score={score:.2f}），保險按 UP 上移")
        else:
            print(f"  [難度] 目前非調諧級，按 UP 上移（score={score:.2f}）")
        if i < max_up:
            _tap(act, Action.UP.value)
            time.sleep(0.2)
            _wait_settle(cap, timeout=1.5)
    print("  ⚠ 多次 UP 後仍未確認調諧級，仍以目前難度嘗試開打。")
    return False


def wait_for_hall(cap, rx, timeout=15.0):
    """等死亡/勝利轉場整個結束：切回雕像大廳(GG_Workshop)、畫面非白、且已靜止。
    這一步避開「白畫面/場景切換」被 open_menu 誤判成菜單的根因。"""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        tele, age = rx.sample()
        scene = tele.get("scene") if (tele is not None and age < config.TELE_FRESH_SEC) else None
        bright = float(cap.grab_obs().mean())
        if scene == config.HALL_SCENE and bright < WHITE_THRESH:
            _wait_settle(cap, timeout=3.0)
            return True
        time.sleep(0.1)
    print("  ⚠ 等待回到雕像大廳逾時，仍嘗試開場。")
    return False


def wake_up(cap, act):
    """按 Z 叫醒角色（死亡重生後會躺地，需先操控一下才站起）。
    Z 會喚醒但不會開難度菜單；站著時只是原地跳，無害。叫醒後等畫面靜止，
    讓後續 open_menu 的基準畫面是站姿，避免把「站起來」誤判成「菜單開了」。"""
    _tap(act, Action.JUMP.value)
    time.sleep(0.5)
    _wait_settle(cap, timeout=3.0)


def open_menu(cap, act, max_attempts=4):
    """按 UP 開菜單。可能第一下被「死亡重生後起身」吃掉，故多試幾次。

    沒開出菜單的那次 UP 不會動到難度（畫面沒變=還沒進菜單），所以多按也安全；
    一旦真的開出菜單就立刻停手。並用「靜止後差異仍在高檔」過濾起身動畫的閃動誤判。
    """
    base = cap.grab_obs()
    for attempt in range(max_attempts):
        _tap(act, Action.UP.value)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 1.8:
            diff = _framediff(cap.grab_obs(), base)
            if diff > MENU_OPEN_THRESH:
                _wait_settle(cap)                                  # 等動畫/菜單靜止
                if _framediff(cap.grab_obs(), base) > MENU_OPEN_THRESH * 0.7:
                    print(f"  [開菜單] 菜單已開（第{attempt+1}次嘗試，差異 {diff:.1f}）")
                    return True
                break  # 只是起身動畫的閃動，重來一次
            time.sleep(0.1)
        print(f"  [開菜單] 第{attempt+1}次 UP 未開出菜單（角色可能剛起身），重試...")
        time.sleep(0.6)                                            # 給起身一點時間
    print("  ⚠ 多次嘗試仍沒偵測到菜單。請確認角色在大黃蜂雕像前。")
    return False


def wait_fight_start(cap, rx, timeout=START_TIMEOUT):
    """等戰鬥真正開始。優先用遙測，否則退回白畫面偵測。回傳是否成功。"""
    t0 = time.perf_counter()
    saw_white = False
    while time.perf_counter() - t0 < timeout:
        tele, age = rx.sample()
        if tele is not None and age < config.TELE_FRESH_SEC:
            if tele.get("scene") in config.HORNET_SCENES and tele.get("boss_present"):
                print(f"  [開打] 遙測確認：scene={tele.get('scene')}, boss 已出現")
                return True
        else:
            b = float(cap.grab_obs().mean())
            if b > WHITE_THRESH:
                saw_white = True
            elif saw_white and b < WHITE_THRESH - 50:
                print("  [開打] 白畫面已結束（無遙測，用像素判斷）")
                return True
        time.sleep(0.08)
    print("  ⚠ 等待開打逾時。")
    return False


class EpisodeMonitor:
    """用遙測判斷一場戰鬥是否結束。回傳 'win' / 'lose' / 'left' / None。

    終止判斷（呼應之前討論：第一次 player_hp==0 鎖敗、boss 消失鎖勝，忽略重生殘影）：
      - lose : player_hp 變 0
      - win  : 曾看到 boss，且玩家還活著時 boss 死亡/消失
      - left : 已離開大黃蜂場景
    """
    def __init__(self):
        self.boss_seen = False

    def update(self, tele):
        if tele is None:
            return None
        scene = tele.get("scene")
        hp = tele.get("player_hp", -1)
        if tele.get("boss_present"):
            self.boss_seen = True
        # 先判敗（死亡瞬間 player_hp 乾淨歸 0，早於場景切換）
        if hp == 0:
            return "lose"
        if scene not in config.HORNET_SCENES:
            return "left" if self.boss_seen else None
        # 玩家還活著、boss 曾出現、現在 boss 沒了或血量見底 => 勝
        if self.boss_seen and (not tele.get("boss_present") or tele.get("boss_hp_raw", 1) <= 0):
            return "win"
        return None


def start_challenge(cap, act, rx):
    """完整開場序列。回傳是否成功進入戰鬥。"""
    print("自動開場中...")
    wait_for_hall(cap, rx)         # 先等死亡/勝利轉場(含白畫面)整個結束、回到大廳
    time.sleep(LYING_WAIT)         # 等地上躺臥期過、角色可操控
    wake_up(cap, act)              # 叫醒站好，再開菜單（避免把起身誤判成菜單）
    if not open_menu(cap, act):
        return False
    time.sleep(0.4)
    ensure_attuned(cap, act)       # 校正到調諧級（菜單預設未必停在它，且可能被誤觸移走）
    # 按 Z 確認難度；若沒開打就再按一次 Z（不重按 UP，避免動到難度）
    for z_try in range(2):
        print("  [確認難度] 按 Z 選調諧級" + ("（重試）" if z_try else ""))
        _tap(act, Action.JUMP.value)
        if wait_fight_start(cap, rx):
            time.sleep(0.3)   # 給一點緩衝讓畫面穩定
            return True
    print("  ⚠ 按 Z 後仍未開打。")
    return False


def _selftest():
    """離線驗證難度偵測：不用開遊戲，直接拿參考圖（與幾個合成情境）測 detect_difficulty。"""
    class _FakeCap:
        def __init__(self, bgr):
            self._rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        def grab_raw(self):
            return self._rgb.copy()

    ref = cv2.imread(DIFFICULTY_REF)
    assert ref is not None, f"讀不到 {DIFFICULTY_REF}"
    ref = cv2.resize(ref, (_REF_W, _REF_H))

    # 情境1：原圖（調諧級被選）-> 應為 attuned
    print("原圖（調諧級）:", detect_difficulty(_FakeCap(ref)))

    # 情境2：把調諧級的左右游標塗黑，模擬「沒有任何游標在頂列」-> 不應再判 attuned
    no_top = ref.copy()
    no_top[255:300, 600:650] = 0      # 左游標
    no_top[255:300, 930:1000] = 0     # 右游標
    print("頂列游標移除:", detect_difficulty(_FakeCap(no_top)))

    # 情境3：把游標從頂列搬到中列（模擬選到進升級）-> 應為 lower
    moved = no_top.copy()
    moved[255+52:300+52, 930:1000] = ref[255:300, 930:1000]
    print("游標移到中列:", detect_difficulty(_FakeCap(moved)))


if __name__ == "__main__":
    _selftest()
