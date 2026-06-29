"""遙測接收器：收 reward mod 透過 UDP 打來的 boss/玩家狀態。

設計要點（呼應之前的討論）：
- UDP、射後不理，最新封包覆蓋舊的；mod 高頻送沒關係。
- 背景執行緒只保留「最新一筆」，主迴圈每個 tick 取最新值，不會 backlog、不會 block。
- mod 還沒寫好前，可用本檔的 fake sender 灌假資料，先把整條 pipeline 跑通。
"""
import json
import socket
import sys
import threading
import time

from config import TELEMETRY_HOST, TELEMETRY_PORT


class TelemetryReceiver:
    def __init__(self, host=TELEMETRY_HOST, port=TELEMETRY_PORT):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.settimeout(0.2)
        self._latest = None       # 最新解析出的 dict
        self._latest_t = 0.0      # 收到的時間 (perf_counter)
        self._running = False
        self._thread = None
        # 每場錄製：收到的每筆遙測 + env 插的 marker，episode 末寫檔（覆蓋）供檢查。
        self._rec = None          # None=未錄製；list[(相對t, dict)]=錄製中
        self._rec_t0 = 0.0
        self._rec_lock = threading.Lock()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            try:
                data, _ = self._sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                d = json.loads(data.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue
            t = time.perf_counter()
            self._latest = d
            self._latest_t = t
            with self._rec_lock:
                if self._rec is not None:
                    self._rec.append((t - self._rec_t0, d))

    def sample(self):
        """回傳 (最新dict 或 None, 距今幾秒)。"""
        if self._latest is None:
            return None, float("inf")
        return self._latest, time.perf_counter() - self._latest_t

    def start_recording(self):
        """開始新一場錄製（清掉上一場），時間軸從現在歸零。"""
        with self._rec_lock:
            self._rec = []
            self._rec_t0 = time.perf_counter()

    def mark(self, label):
        """在錄製中插一條 marker（如 RESET/ARMED/EP_END），標出 env 端事件時點。"""
        with self._rec_lock:
            if self._rec is not None:
                self._rec.append((time.perf_counter() - self._rec_t0, {"_mark": label}))

    def dump_recording(self, path):
        """把本場錄製覆蓋寫到 path（每筆遙測一行 + marker 行）。"""
        with self._rec_lock:
            rec = list(self._rec) if self._rec is not None else []
        try:
            with open(path, "w", encoding="utf-8") as f:
                for t, d in rec:
                    if "_mark" in d:
                        f.write(f"t={t:7.3f}  ===== {d['_mark']} =====\n")
                    else:
                        f.write(f"t={t:7.3f}  {json.dumps(d, ensure_ascii=False)}\n")
        except OSError:
            pass

    def stop(self):
        self._running = False
        self._sock.close()


def _fake_sender():
    """測試用：模擬 mod，60Hz 送假資料。boss 血量慢慢掉。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    boss = 1.0
    player = 9
    print(f"fake sender -> {TELEMETRY_HOST}:{TELEMETRY_PORT}，Ctrl+C 結束")
    try:
        while True:
            boss = max(0.0, boss - 0.001)
            pkt = json.dumps({
                "player_hp": player,
                "boss_hp": round(boss, 3),
                "boss_dead": boss <= 0.0,
                "in_fight": True,
            }).encode("utf-8")
            sock.sendto(pkt, (TELEMETRY_HOST, TELEMETRY_PORT))
            time.sleep(1 / 60)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "send":
        _fake_sender()
    else:
        # 接收模式：印出收到的資料
        rx = TelemetryReceiver()
        rx.start()
        print(f"接收中 {TELEMETRY_HOST}:{TELEMETRY_PORT}（另開一個視窗跑 'python telemetry.py send'）")
        try:
            while True:
                data, age = rx.sample()
                print(f"latest={data}  age={age:.2f}s")
                time.sleep(0.5)
        except KeyboardInterrupt:
            rx.stop()
