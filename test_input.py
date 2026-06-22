"""階段 0 最關鍵的驗證：證明 Python 能讓 HK 角色動。

用法：
  1. 開遊戲，進到神居一個可以自由走動的地方（站在大黃蜂雕像前那塊空地就好）。
  2. 執行：  python test_input.py              # 預設後端（config.INPUT_BACKEND）
            python test_input.py --input keyboard
            python test_input.py --input gamepad   # 測虛擬手把
  3. 倒數 5 秒內：keyboard 後端要用滑鼠點一下遊戲視窗取得焦點；gamepad 後端不需焦點。
  4. 看角色會不會照腳本動：右走→左走→跳→攻擊→衝刺。

若角色完全沒反應，代表這個輸入方式對 HK 無效，要換方案（scancode 細節 / 虛擬手把），
這比後面任何事都重要，先卡死在這裡也沒關係。
gamepad 後端還需確認 HK 內手把綁定與 gamepad_setup.py 的對應一致。
"""
import argparse
import time

import config
from config import Action
from inputs import make_actuator


def hold(act, keys, seconds, label):
    print(f"  [{label}] 按住 {keys} {seconds}s")
    act.apply_keys(set(keys))
    time.sleep(seconds)
    act.apply_keys(set())
    time.sleep(0.3)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", choices=["keyboard", "gamepad"], default=config.INPUT_BACKEND)
    return ap.parse_args()


def main():
    args = parse_args()

    print(f"輸入後端: {args.input}")
    act = make_actuator(args.input)
    print("5 秒後開始" + ("..." if args.input == "gamepad"
                          else "，請點一下遊戲視窗讓它取得焦點..."))
    for i in range(5, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    print("開始輸入測試：")
    try:
        hold(act, [Action.LEFT.value], 0.8, "左")
        hold(act, [Action.RIGHT.value], 0.8, "右")
        hold(act, [Action.UP.value], 0.8, "上")
        hold(act, [Action.DOWN.value], 0.8, "下")
        # hold(act, [Action.LEFT.value], 0.8, "左走")
        # hold(act, [Action.RIGHT.value], 0.8, "右走")
        # hold(act, [Action.JUMP.value], 0.1, "短跳")
        # hold(act, [Action.JUMP.value], 0.5, "長跳(按久一點，應該跳更高)")
        # hold(act, [Action.ATTACK.value], 0.1, "攻擊")
        # hold(act, [Action.DASH.value], 0.1, "衝刺")
        # # 組合鍵：右走同時攻擊
        # hold(act, [Action.RIGHT.value, Action.ATTACK.value], 0.4, "右走+攻擊")
        # hold(act, [Action.DREAM_NAIL.value], 1.0, "夢之釘")
    finally:
        act.release_all()
    print("測試結束。角色有照上面動作動嗎？")


if __name__ == "__main__":
    main()
