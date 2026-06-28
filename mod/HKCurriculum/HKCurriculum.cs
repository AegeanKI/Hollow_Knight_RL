using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Reflection;
using Modding;
using UnityEngine;

namespace HKCurriculum
{
    // 自適應難度 mod（A1 curriculum）：每場戰鬥依「滑動勝率」調整難度。
    // 做法 D：不動 boss 血量，而是「放大 agent 對 boss 的傷害」(×1/scale)，讓整場戰鬥
    // 等比壓縮（所有 phase 都在、各自縮短），boss 在打出 scale×滿血 的真實傷害時死。
    // 輸太多 -> 降 scale（傷害放更大、更好贏，最低 50%）讓 agent 收集到勝利訊號；
    // 贏太多 -> 升 scale（回到 ×1，最高 100%）。eval 固定不放大（Python 寫 eval 旗標檔）。
    //
    // 與 HKReward 分工：HKReward 只讀遙測；本 mod 只控難度。Python 用絕對 boss_hp_raw 判
    // 勝負/傷害——boss 真的扣了那麼多血，判定仍正確。
    public class HKCurriculum : Mod
    {
        private GameObject _go;
        private CurriculumPump _pump;

        public HKCurriculum() : base("HKCurriculum") { }

        public override string GetVersion() => "0.1.0";

        public void Info(string msg) => Log(msg);   // 讓 pump 也能寫 log

        public override void Initialize()
        {
            Application.runInBackground = true;     // 失焦仍跑，配合背景訓練
            _go = new GameObject("HKCurriculumPump");
            UnityEngine.Object.DontDestroyOnLoad(_go);
            _pump = _go.AddComponent<CurriculumPump>();
            _pump.Init(this);
            ModHooks.OnEnableEnemyHook += OnEnableEnemy;
            On.HealthManager.TakeDamage += OnTakeDamage;   // 做法D：放大 agent 對 boss 的傷害
            Log($"HKCurriculum initialized, scale={_pump.Scale:0.00}");
        }

        private bool OnEnableEnemy(GameObject enemy, bool isAlreadyDead)
        {
            _pump?.OnEnableEnemy(enemy, isAlreadyDead);
            return isAlreadyDead;                   // 原樣回傳，不改死活
        }

        // 攔截「對敵人造成傷害」：是 boss 就把傷害乘上 1/scale（所有攻擊類型都走這條漏斗）
        private void OnTakeDamage(On.HealthManager.orig_TakeDamage orig, HealthManager self, HitInstance hit)
        {
            if (_pump != null) hit = _pump.AmplifyIfBoss(self, hit);
            orig(self, hit);
        }
    }

    public class CurriculumPump : MonoBehaviour
    {
        // ---- 可調參數 ----
        private const float ScaleMin = 0.50f;    // 最低降到 50%
        private const float ScaleMax = 1.00f;    // 最高回到 100%
        // 步階 0.02（原 0.05）：縮小升難躍變，讓「在能力邊界橫跳」時每趟難度跳更小、
        // 對 policy 的單趟傷害更小（解棘輪式崩潰；2026-06-27）。代價=爬到滿難度更慢，
        // 但現階段瓶頸是穩定不是爬速；privileged critic 也讓小步不再造成 value shock。
        private const float ScaleStep = 0.02f;   // 每次調整步階
        // 滑動勝率視窗（場數）。**必須明顯大於學習尺度(eps/update=8)**，否則升難度後策略
        // 只跑 ~1 次 update 還沒適應就被判「太難」而反覆降回（thrashing）。30≈4 次 update，
        // 給策略時間在新難度學透才評估；還會 ping-pong 就再加大到 40-50。
        private const int Window = 30;
        // 升難門檻 0.70（原 0.60）：要求贏得更穩才升，避免一摸到能勝就升進撐不住的難度、
        // 反覆橫跳硬砸 policy（2026-06-27）。
        private const float RaiseAbove = 0.70f;  // 勝率 > 此 -> 升難度
        private const float LowerBelow = 0.30f;  // 勝率 < 此 -> 降難度
        private const int BossMinHp = 200;       // 視為 boss 的最低滿血（過濾雜魚）
        private const int WaitFrames = 5;        // 等 FSM 設好滿血再判定是不是 boss 的幀數（實測可調）
        private const float PollInterval = 1f / 15f;

        // 目標 boss 場景（對應 Python config.HORNET_SCENES）
        private static readonly HashSet<string> BossScenes =
            new HashSet<string> { "GG_Hornet_1", "GG_Hornet_2" };

        private HKCurriculum _mod;
        private float _accum;

        public float Scale { get; private set; } = ScaleMax;
        private readonly Queue<bool> _results = new Queue<bool>();   // true=win
        private readonly HashSet<int> _handled = new HashSet<int>(); // 本場已處理的 enemy 實例

        // 本場狀態
        private HealthManager _boss;
        private bool _fightActive;
        private bool _resolved;
        private bool _evalThisFight;
        private bool _dropThisFight;    // Python 偵測到畫面遮擋 -> 本場 win/lose 不計入勝率

        private string _stateFile;     // 持久化難度+勝率視窗（mod 目錄，停/續訓/重開不重置）
        private string _logFile;       // 每場輸贏的歷史紀錄（append；給事後分析）
        private string _scaleOutFile;  // 寫目前 scale 供 Python 讀來記 log（temp）
        private string _evalFlagFile;  // Python 寫此檔 -> 本場用 100% 且不計入（temp）
        private string _dropFlagFile;  // Python 寫此檔 -> 本場（遮擋）不計入勝率，但難度照舊（temp）

        // 完整路徑取現在場景名（避免被 HK 自己的同名 SceneManager 型別遮蔽）
        private static string ActiveScene() =>
            UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;

        public void Init(HKCurriculum mod)
        {
            _mod = mod;
            string dir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
            _stateFile = Path.Combine(dir, "curriculum_state.txt");
            _logFile = Path.Combine(dir, "curriculum_log.csv");
            _scaleOutFile = Path.Combine(Path.GetTempPath(), "hk_curriculum_scale.txt");
            _evalFlagFile = Path.Combine(Path.GetTempPath(), "hk_curriculum_eval.flag");
            _dropFlagFile = Path.Combine(Path.GetTempPath(), "hk_curriculum_drop.flag");
            LoadState();
            WriteScaleOut();
        }

        // boss 進場：抓到目標 boss 後記錄起來（供傷害放大與勝負偵測用），只處理一次
        public void OnEnableEnemy(GameObject enemy, bool isAlreadyDead)
        {
            if (isAlreadyDead || enemy == null) return;
            if (!BossScenes.Contains(ActiveScene())) return;
            var hm = enemy.GetComponent<HealthManager>();
            if (hm == null) return;
            int id = hm.GetInstanceID();
            if (!_handled.Add(id)) return;          // 此實例已處理過 -> 不重複
            StartCoroutine(SetupFightAfterInit(hm));
        }

        // 等 FSM 設好滿血後，確認是 boss 並記錄起來（不動血量；傷害放大在 TakeDamage hook 做）
        private IEnumerator SetupFightAfterInit(HealthManager hm)
        {
            for (int i = 0; i < WaitFrames; i++) yield return null;
            if (hm == null || hm.hp < BossMinHp) yield break;        // 不是 boss（雜魚），跳過

            _evalThisFight = File.Exists(_evalFlagFile);             // 本場 eval -> 不放大、不計入
            _dropThisFight = false;                                  // 新一場：清掉上一場的遮擋旗標
            _boss = hm;
            _fightActive = true;
            _resolved = false;
            float m = DamageMultiplier();
            _mod.Info($"[curriculum] fight start: bossHp={hm.hp} "
                      + (_evalThisFight ? "EVAL 100% dmg×1.00" : $"scale={Scale:0.00} dmg×{m:0.00}"));
        }

        // 是這場的 boss 就把傷害乘上 1/scale（eval 或 scale=1 不放大）。HK 傷害是整數 -> round。
        public HitInstance AmplifyIfBoss(HealthManager self, HitInstance hit)
        {
            if (self != null && self == _boss)
            {
                float m = DamageMultiplier();
                if (m > 1f) hit.DamageDealt = Mathf.RoundToInt(hit.DamageDealt * m);
            }
            return hit;
        }

        private float DamageMultiplier()
        {
            if (_evalThisFight || Scale >= ScaleMax || Scale <= 0f) return 1f;
            return 1f / Scale;   // scale=0.6 -> ×1.667（整場等比壓縮）
        }

        private void Update()
        {
            _accum += Time.unscaledDeltaTime;
            if (_accum < PollInterval) return;
            _accum = 0f;

            bool inBossScene = BossScenes.Contains(ActiveScene());
            if (!inBossScene)
            {
                // 離開戰鬥場景：重置本場狀態（沒結算的就當棄場、不計）
                _fightActive = false;
                _dropThisFight = false;
                _handled.Clear();
                return;
            }
            if (!_fightActive || _resolved) return;

            // 遮擋旗標：Python 偵測到畫面被遮擋後寫此檔，閂住 -> 本場（即將「等輸」的敗）不計入勝率。
            // 用閂的（看到一次就記住），Python 之後清檔的時機就不必精準。
            if (!_dropThisFight && File.Exists(_dropFlagFile)) _dropThisFight = true;

            var pd = PlayerData.instance;
            int playerHp = pd != null ? pd.health : -1;

            if (playerHp == 0) Resolve(false);                      // 敗（死亡瞬間 hp 乾淨歸 0，優先鎖）
            else if (_boss == null || _boss.hp <= 0) Resolve(true); // 勝（boss 死/消失，玩家還活）
        }

        private void Resolve(bool win)
        {
            _resolved = true;
            _fightActive = false;
            if (_dropThisFight)
            {
                _mod.Info($"[curriculum] dropped fight (win={win})：畫面遮擋，不計入自適應");
                return;
            }
            if (_evalThisFight)
            {
                _mod.Info($"[curriculum] eval fight done (win={win})，不計入自適應");
                return;
            }
            _results.Enqueue(win);
            while (_results.Count > Window) _results.Dequeue();
            AdjustScale();
            SaveState();
            WriteScaleOut();
            AppendLog(win);
            _mod.Info($"[curriculum] {(win ? "WIN" : "LOSE")} winrate={WinRate():0.00} scale->{Scale:0.00}");
        }

        private float WinRate()
        {
            if (_results.Count == 0) return 0f;
            int w = 0;
            foreach (var r in _results) if (r) w++;
            return (float)w / _results.Count;
        }

        // 自適應難度（滑動勝率視窗 + 死區遲滯）：
        //  - 視窗永遠只留最近 Window 場（每場滑動）；滿 Window 後「每打一場就重新評估一次」。
        //  - 勝率 > RaiseAbove 升、< LowerBelow 降（嚴格比較；剛好等於門檻算死區、不動）。
        //  - 只有「真的升/降」才清空視窗 = 冷卻：之後要再湊滿 Window 場才可能下一次調整。
        //  - 落在 [LowerBelow, RaiseAbove] 死區 -> 維持同一 scale、視窗繼續滑（可停留很多場，正常）。
        //  - 觸頂(=ScaleMax)/觸底(=ScaleMin)被 clamp 而 scale 沒實際改變時「不清空」：否則卡在
        //    地板又打不過會每滿 30 場就清一次、永遠重新累積，視窗無法跨 30 場累積進步。
        private void AdjustScale()
        {
            if (_results.Count < Window) return;     // 視窗未滿先不動（剛清空後要重新累積）
            float wr = WinRate();
            float before = Scale;
            if (wr > RaiseAbove) Scale = Mathf.Min(ScaleMax, Scale + ScaleStep);
            else if (wr < LowerBelow) Scale = Mathf.Max(ScaleMin, Scale - ScaleStep);
            else return;                             // 死區：不動、也不清空（繼續滑動評估）
            if (Scale != before) _results.Clear();   // 只有真的升/降才清空；觸頂/觸底沒變不清
        }

        // ---- 持久化 / 檔案交握 ----
        // 持久化格式（兩行）：第1行 scale；第2行 勝率視窗（逗號分隔的 1/0）
        private void LoadState()
        {
            try
            {
                if (!File.Exists(_stateFile)) return;
                var lines = File.ReadAllLines(_stateFile);
                if (lines.Length >= 1 &&
                    float.TryParse(lines[0].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out var v))
                    Scale = Mathf.Clamp(v, ScaleMin, ScaleMax);
                _results.Clear();
                if (lines.Length >= 2)
                    foreach (var tok in lines[1].Split(','))
                    {
                        if (tok == "1") _results.Enqueue(true);
                        else if (tok == "0") _results.Enqueue(false);
                    }
                while (_results.Count > Window) _results.Dequeue();
            }
            catch { }
        }

        private void SaveState()
        {
            try
            {
                var win = new System.Text.StringBuilder();
                bool first = true;
                foreach (var r in _results)
                {
                    if (!first) win.Append(',');
                    win.Append(r ? '1' : '0');
                    first = false;
                }
                File.WriteAllText(_stateFile,
                    Scale.ToString("0.####", CultureInfo.InvariantCulture) + "\n" + win);
            }
            catch { }
        }

        // 每場輸贏歷史（append-only），供事後分析；遊戲重開也不會清掉
        private void AppendLog(bool win)
        {
            try
            {
                File.AppendAllText(_logFile,
                    (win ? "W" : "L") + "," + Scale.ToString("0.####", CultureInfo.InvariantCulture) + "\n");
            }
            catch { }
        }

        private void WriteScaleOut()
        {
            try { File.WriteAllText(_scaleOutFile, Scale.ToString("0.####", CultureInfo.InvariantCulture)); }
            catch { }
        }
    }
}
