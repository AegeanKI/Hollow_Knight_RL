using System;
using System.Collections.Generic;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Text;
using Modding;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace HKReward
{
    // Modding API 的進入點。只負責掛上一個每幀執行的 MonoBehaviour（TelemetryPump）。
    public class HKReward : Mod
    {
        public const string Host = "127.0.0.1";
        public const int Port = 48789;   // 對應 Python config.TELEMETRY_PORT（51789 在 Windows UDP 排除範圍會 bind 失敗）

        private GameObject _pumpGo;

        public HKReward() : base("HKReward") { }

        public override string GetVersion() => "0.1.0";

        public override void Initialize()
        {
            // 讓遊戲失焦時仍持續執行，配合虛擬手把即可在背景訓練
            Application.runInBackground = true;

            _pumpGo = new GameObject("HKRewardPump");
            UnityEngine.Object.DontDestroyOnLoad(_pumpGo);
            _pumpGo.AddComponent<TelemetryPump>().Init(this);
            Log($"HKReward initialized, UDP -> {Host}:{Port}");
        }
    }

    // 真正幹活的：以固定頻率讀遊戲狀態並用 UDP 送出。
    public class TelemetryPump : MonoBehaviour
    {
        private const float SendInterval = 1f / 30f;  // 30Hz；Python 端只取最新值，送快一點無妨

        private HKReward _mod;
        private UdpClient _udp;
        private IPEndPoint _ep;
        private float _accum;

        // 記錄每個 HealthManager 看過的最大 hp，用來把 boss 血量正規化成 0..1。
        // （有些 boss 的 hp 是進場後才由 FSM 設定，用 running-max 比較穩。）
        private readonly Dictionary<int, int> _maxHp = new Dictionary<int, int>();

        public void Init(HKReward mod)
        {
            _mod = mod;
            _udp = new UdpClient();
            _ep = new IPEndPoint(IPAddress.Parse(HKReward.Host), HKReward.Port);
        }

        private void Update()
        {
            _accum += Time.unscaledDeltaTime;
            if (_accum < SendInterval) return;
            _accum = 0f;
            try { SendTelemetry(); }
            catch { /* 送失敗（Python 沒開）就略過，別影響遊戲 */ }
        }

        private void SendTelemetry()
        {
            // ---- 玩家狀態 ----
            int playerHp = -1, playerMax = -1;
            int soul = -1, soulMax = -1, soulReserve = -1, soulReserveMax = -1;
            var pd = PlayerData.instance;
            if (pd != null)
            {
                playerHp = pd.health;
                playerMax = pd.maxHealth;
                soul = pd.MPCharge;            // 主魂槽 (0..maxMP)
                soulMax = pd.maxMP;
                soulReserve = pd.MPReserve;     // 儲備魂槽 (主槽滿後溢出到這)
                soulReserveMax = pd.MPReserveMax;
            }

            // ---- boss = 場上「看過的最大 hp 池」且還活著的 HealthManager ----
            bool bossPresent = false;
            float bossHpNorm = -1f;
            int bossRaw = -1, bossMax = -1;

            var hms = UnityEngine.Object.FindObjectsOfType<HealthManager>();
            int bestMax = 0;
            HealthManager boss = null;
            foreach (var hm in hms)
            {
                if (hm == null) continue;
                int id = hm.GetInstanceID();
                int cur = hm.hp;
                int seen = _maxHp.TryGetValue(id, out var m) ? m : 0;
                if (cur > seen) { seen = cur; _maxHp[id] = seen; }
                if (seen > bestMax) { bestMax = seen; boss = hm; }
            }
            if (boss != null && bestMax > 0)
            {
                bossPresent = true;
                bossRaw = Mathf.Max(0, boss.hp);
                bossMax = bestMax;
                bossHpNorm = (float)bossRaw / bestMax;
            }

            string scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;

            // ---- 手刻 JSON（避免額外相依，float 用 InvariantCulture）----
            var sb = new StringBuilder(256);
            sb.Append('{');
            sb.Append("\"player_hp\":").Append(playerHp).Append(',');
            sb.Append("\"player_max\":").Append(playerMax).Append(',');
            sb.Append("\"soul\":").Append(soul).Append(',');
            sb.Append("\"soul_max\":").Append(soulMax).Append(',');
            sb.Append("\"soul_reserve\":").Append(soulReserve).Append(',');
            sb.Append("\"soul_reserve_max\":").Append(soulReserveMax).Append(',');
            sb.Append("\"soul_total\":").Append(soul + soulReserve).Append(',');
            sb.Append("\"boss_present\":").Append(bossPresent ? "true" : "false").Append(',');
            sb.Append("\"boss_hp\":").Append(bossHpNorm.ToString("0.####", CultureInfo.InvariantCulture)).Append(',');
            sb.Append("\"boss_hp_raw\":").Append(bossRaw).Append(',');
            sb.Append("\"boss_max\":").Append(bossMax).Append(',');
            sb.Append("\"in_fight\":").Append((bossPresent && playerHp > 0) ? "true" : "false").Append(',');
            sb.Append("\"scene\":\"").Append(scene).Append('"');
            sb.Append('}');

            byte[] data = Encoding.UTF8.GetBytes(sb.ToString());
            _udp.Send(data, data.Length, _ep);
        }

        private void OnDestroy()
        {
            try { _udp?.Close(); } catch { }
        }
    }
}
