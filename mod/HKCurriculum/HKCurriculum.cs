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
        // ---- 可調參數（Init 時從 curriculum_config.txt 讀；不存在則寫預設）----
        // 預設值＝下面的初始化值；改參數編輯 config 檔即可、不必重 build。完整語意見舊註解/memory。
        private float ScaleMin = 0.50f;          // 最低降到 50%
        private float ScaleMax = 1.00f;          // 最高回到 100%
        private float ScaleStep = 0.02f;         // 每次調整步階（小步解棘輪崩潰）
        private int Window = 30;                  // 滑動勝率視窗（須 > eps/update=8，否則 thrashing）
        private float RaiseAbove = 0.70f;        // 勝率 > 此 -> 升難度
        private float LowerBelow = 0.30f;        // 勝率 < 此 -> 降難度
        private int BossMinHp = 200;             // 視為 boss 的最低滿血（過濾雜魚）
        private int WaitFrames = 5;              // 等 FSM 設好滿血再判定是不是 boss 的幀數
        private float PollInterval = 1f / 15f;

        // ---- 殘局模式 (finale practice) 參數（同檔讀）----
        private int FinaleEvery = 0;             // >0：確定性「每 N 場(非eval)第 N 場殘局」；0=停用。第一場永遠正常(wasReady)
        private int FinalePlayerMinMasks = 2;    // 殘局玩家起始血(整數面具)隨機範圍 [min,max]；釘真實死亡區
        private int FinalePlayerMaxMasks = 3;    // (1-2 面具)＝練 HUD-conditioned actor 真正會輸的低血收尾分支
        private float FinaleBossMinFrac = 0.20f; // 殘局 boss 起始血＝真實滿血百分比 [min,max]（與玩家面具解耦）。
        private float FinaleBossMaxFrac = 0.30f; // 低血 AND 低 boss＝殊死收尾；boss 夠低才贏得到。boss_frac 僅供 Python ARMED log、不進 reward
        private bool FinaleDebug = false;        // 診斷開關：寫 arena_probe/hud_probe/FSM log/[heropos] spam（換王重 probe 才開）
        private float FinaleOpeningDelay = 5.0f; // 殘局：等 boss 進戰鬥態(過開場)的 fallback 上限秒數
        private float FinaleSettleDelay = 0.3f;  // 殘局：設殘血/移位後再等這秒數讓物理/遙測/面具穩定才 arm
        private float FinaleArenaXMin = -999f;   // 殘局放置 x 範圍覆寫（> -900 才生效；否則用 CameraLockArea/fallback）
        private float FinaleArenaXMax = -999f;   // GG_Hornet_1 實測可走 [15.3,37.7]
        private readonly System.Random _rng = new System.Random();

        // 場地座標（第一場正常戰鬥 capture；殘局放位置用）。未就緒前一律強制正常場。
        private bool _arenaReady;
        private float _floorY;
        private float _arenaXMin, _arenaXMax;
        private bool _finaleThisFight;           // 本場是否殘局
        private int _fightCounter;               // 非 eval 戰鬥計數（finale_every 確定性週期用；重啟歸零）

        // 目標 boss 場景（對應 Python config.HORNET_SCENES）
        private static readonly HashSet<string> BossScenes =
            new HashSet<string> { "GG_Hornet_1", "GG_Hornet_2" };

        private HKCurriculum _mod;
        private float _accum;

        public float Scale { get; private set; } = 1.00f;   // 真值由 LoadConfig/LoadState 設（預設＝ScaleMax）
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
        private string _configFile;    // 可調參數（mod 目錄；不存在則寫預設，編輯後重啟生效）
        private string _finaleFile;    // 殘局握手：mod 寫 finale/armed/true_max 供 Python reset 反應（temp）
        private string _probeFile;     // 場地/開場 FSM 候選值 dump（mod 目錄；給人工挑正確欄位）
        private string _hudProbeFile;  // 血量 HUD FSM dump（mod 目錄；找面具重畫事件用）

        // 完整路徑取現在場景名（避免被 HK 自己的同名 SceneManager 型別遮蔽）
        private static string ActiveScene() =>
            UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;

        public void Init(HKCurriculum mod)
        {
            _mod = mod;
            string dir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
            _stateFile = Path.Combine(dir, "curriculum_state.txt");
            _logFile = Path.Combine(dir, "curriculum_log.csv");
            _configFile = Path.Combine(dir, "curriculum_config.txt");
            _probeFile = Path.Combine(dir, "arena_probe.txt");
            _hudProbeFile = Path.Combine(dir, "hud_probe.txt");
            _scaleOutFile = Path.Combine(Path.GetTempPath(), "hk_curriculum_scale.txt");
            _evalFlagFile = Path.Combine(Path.GetTempPath(), "hk_curriculum_eval.flag");
            _dropFlagFile = Path.Combine(Path.GetTempPath(), "hk_curriculum_drop.flag");
            _finaleFile = Path.Combine(Path.GetTempPath(), "hk_curriculum_finale.txt");
            LoadConfig();          // 在 LoadState 之前（ScaleMin/ScaleMax 給 clamp 用）
            LoadState();
            Scale = Mathf.Clamp(Scale, ScaleMin, ScaleMax);   // 無 state 檔時 Scale 仍是初值，夾進 config 範圍
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

        // 等 FSM 設好滿血後，確認是 boss 並記錄起來。正常場不動血量（傷害放大在 TakeDamage hook 做）；
        // 殘局場降當前血 + 隨機位置 + 等開場跑完才 arm（細節見專案 progress.md「殘局模式」）。
        private IEnumerator SetupFightAfterInit(HealthManager hm)
        {
            for (int i = 0; i < WaitFrames; i++) yield return null;
            if (hm == null || hm.hp < BossMinHp) yield break;        // 不是 boss（雜魚），跳過

            _evalThisFight = File.Exists(_evalFlagFile);             // 本場 eval -> 不放大、不計入
            _dropThisFight = false;                                  // 新一場：清掉上一場的遮擋旗標
            _resolved = false;
            int trueMax = hm.hp;                                     // 降血前的真實滿血（FSM 已設好）

            // 第一場：強制正常，capture 場地座標（+ debug 時 dump 候選值）。
            // 用 wasReady 快照：CaptureArena 會把 _arenaReady 設 true，但「本場」仍須算第一場（強制正常）。
            bool wasReady = _arenaReady;
            if (!_arenaReady) { CaptureArena(hm); if (FinaleDebug) DumpArenaProbe(hm); }

            // 場次計數（eval 不算進殘局週期）。第一場 counter=1（wasReady=false → 仍強制正常）。
            if (!_evalThisFight) _fightCounter++;
            // 殘局：eval 永不、第一場永不（wasReady）；finale_every>0 時「每 N 場的第 N 場」。
            _finaleThisFight = !_evalThisFight && wasReady
                               && FinaleEvery > 0 && (_fightCounter % FinaleEvery == 0);

            if (!_finaleThisFight)                                   // 正常場（含 eval）：與原行為完全一致
            {
                WriteFinaleFile(false, true, trueMax);              // armed=1：Python 不等
                _boss = hm;
                _fightActive = true;
                float mn = DamageMultiplier();
                _mod.Info($"[curriculum] fight start: bossHp={hm.hp} "
                          + (_evalThisFight ? "EVAL 100% dmg×1.00" : $"scale={Scale:0.00} dmg×{mn:0.00}"));
                yield break;
            }

            // ---- 殘局場 ----
            WriteFinaleFile(true, false, trueMax);                  // 先告知 Python：殘局、尚未 armed
            _finaleCount++;

            // 先降血 + 移位（boss 的落地/開場動畫在這之後才等）。
            // 殘血：玩家取整數面具 [min,max]；boss 用「獨立百分比」[BossMinFrac,BossMaxFrac]（各自抽、解耦）。
            // 設計＝把玩家 AND boss 都釘在「真實 EVAL 死亡區」(玩家 1-2 面具、boss 打掉 ~77% 後的殘血)＝練
            // HUD-conditioned actor 真正會輸的低血殊死收尾分支(舊 5-6 面具版練的是健康收殘血、HUD 情境不對、
            // 不轉移)。boss 夠低(0.20-0.30)才贏得到。boss_frac/player_frac 可重疊(不再保證 boss≤player)。
            // boss_frac 經 armed 握手帶給 Python，僅供 ARMED 遙測 log、不進 reward(×boss_frac 已還原)。
            int pmax = PlayerMaxHp();
            int playerMasks = Mathf.Clamp(_rng.Next(FinalePlayerMinMasks, FinalePlayerMaxMasks + 1), 1, pmax);
            float bossFrac = RandRange(FinaleBossMinFrac, FinaleBossMaxFrac);
            SetPlayerHp(playerMasks);
            hm.hp = Mathf.Max(1, Mathf.RoundToInt(trueMax * bossFrac));
            PlaceCombatants(hm);

            // 移位後等 boss 真正進戰鬥態（離開含 intro/GG Land/Flourish 的開場 state，那段不攻擊＝free-hit）。
            // 需先「看過」開場 state 才認它離開，避免誤判提早。偵測不到 / 開場是 input-gated（agent idle 時
            // boss 不推進）→ 退回 finale_opening_delay 上限照常 arm（post-arm FSM log 會顯示是哪種）。
            _lastFsmLog = -1f;
            bool sawOpening = false;
            for (float t = 0f; t < FinaleOpeningDelay; t += Time.unscaledDeltaTime)
            {
                if (hm == null) { WriteFinaleFile(false, true, -1); yield break; }   // boss 消失，棄→當正常
                if (FinaleDebug && _finaleCount <= FsmProbeFights) LogBossFsmState(hm, t);
                bool inOpening = BossInOpening(hm);
                if (inOpening) sawOpening = true;
                else if (sawOpening) break;                                          // 看過開場且已進戰鬥
                yield return null;
            }
            if (hm == null) { WriteFinaleFile(false, true, -1); yield break; }

            // settle：等角色落地、物理/遙測/面具穩定再 arm（PlaceCombatants 放 floorY+1 靠重力沉下）。
            for (float s = 0f; s < FinaleSettleDelay; s += Time.unscaledDeltaTime)
            {
                if (hm == null) { WriteFinaleFile(false, true, -1); yield break; }
                yield return null;
            }

            _boss = hm;
            _fightActive = true;
            float m = DamageMultiplier();
            _mod.Info($"[curriculum] FINALE start: bossHp={hm.hp}/{trueMax}({bossFrac:0.00}) "
                      + $"playerHp={playerMasks}/{pmax} scale={Scale:0.00} dmg×{m:0.00}");
            WriteFinaleFile(true, true, trueMax, bossFrac);       // armed：Python 可以開始（帶 boss_frac 供 ARMED log）
            // 【診斷】開始記錄 arm 後 boss FSM（Update 裡跑 FinaleFsmPolls 次）；marker 分隔開場 gate 段
            if (FinaleDebug && _finaleCount <= FsmProbeFights)
            {
                _finaleActivePolls = 0;
                _lastFsmLog = -1f;
                try { File.AppendAllText(_probeFile, "[fsm] === ARMED (agent 開始) ===\n"); } catch { }
            }
        }

        private float _lastFsmLog = -1f;
        private int _finaleCount;                 // 跑過幾場殘局（用來只在前幾場 dump FSM state）
        private const int FsmProbeFights = 5;     // 只在前 N 場殘局記 boss FSM state 到 probe
        private int _finaleActivePolls;           // 【診斷】殘局 arm 後已記錄幾次 FSM
        private const int FinaleFsmPolls = 60;    // 【診斷】arm 後記錄 boss FSM 的 poll 數（~4s@15Hz）
        private float _heroXMin = 9999f, _heroXMax = -9999f; // 自動學習的放置 x 範圍（onFloor 觀測擴張）
        private const float FloorYTol = 1.0f;     // 視為「在初始 floor 高度」的 y 容差（濾掉跳躍/別樓層）

        private float RandRange(float a, float b) =>
            (b <= a) ? a : a + (float)_rng.NextDouble() * (b - a);

        private int PlayerMaxHp()
        {
            var pd = PlayerData.instance;
            return pd != null ? pd.maxHealth : 9;
        }

        // 殘局：直接設玩家當前血為整數面具數（vision-only 遙測讀 pd.health 即準）
        private void SetPlayerHp(int masks)
        {
            var pd = PlayerData.instance;
            if (pd == null) return;
            pd.health = Mathf.Clamp(masks, 1, pd.maxHealth);
            if (FinaleDebug && !_hudProbed) { _hudProbed = true; DumpHudFsms(); }   // debug 才 dump
            RefreshHealthHud();   // 直接寫 pd.health 不會重畫面具 → 送 HERO DAMAGED 讓面具重算
        }

        private bool _hudProbed;

        // 直接寫 pd.health 不會讓血量面具重畫；送 "HERO DAMAGED" 給 11 個 health_display FSM，
        // 讓每個面具依 pd.health 重新判定 full/empty（與正常受傷的重畫路徑相同，但不真的扣血/無 i-frame）。
        private void RefreshHealthHud()
        {
            try
            {
                var t = FindTypeByName("PlayMakerFSM");
                var send = t?.GetMethod("SendEvent", new[] { typeof(string) });
                var nameProp = t?.GetProperty("FsmName");
                if (send == null || nameProp == null) return;
                foreach (var o in UnityEngine.Object.FindObjectsOfType(t))
                    if ((nameProp.GetValue(o, null) as string) == "health_display")
                        send.Invoke(o, new object[] { "HERO DAMAGED" });
            }
            catch { }
        }

        // 反射 dump HUD 上「血量」相關 PlayMakerFSM（owner/FsmName/事件名），找重畫面具的事件。
        private void DumpHudFsms()
        {
            try
            {
                var t = FindTypeByName("PlayMakerFSM");
                if (t == null) { File.WriteAllText(_hudProbeFile, "PlayMakerFSM type not found\n"); return; }
                var all = UnityEngine.Object.FindObjectsOfType(t);
                var sb = new System.Text.StringBuilder();
                sb.AppendLine("# HUD probe：找血量面具 HUD 的刷新事件（owner 或 FsmName 含 health 的 FSM）");
                foreach (var o in all)
                {
                    var comp = o as Component;
                    string go = comp != null ? comp.gameObject.name : "?";
                    string fsm = t.GetProperty("FsmName")?.GetValue(o, null) as string ?? "?";
                    if ((go + fsm).ToLower().IndexOf("health") < 0) continue;
                    sb.Append($"GO={go} FSM={fsm} events=[");
                    if (t.GetProperty("FsmEvents")?.GetValue(o, null) is System.Collections.IEnumerable evs)
                        foreach (var e in evs)
                            sb.Append((e.GetType().GetProperty("Name")?.GetValue(e, null) as string) + ",");
                    sb.AppendLine("]");
                }
                File.WriteAllText(_hudProbeFile, sb.ToString());
                _mod?.Info($"[curriculum] HUD probe dumped -> {_hudProbeFile}");
            }
            catch (Exception e) { _mod?.Info($"[curriculum] HUD probe failed: {e.Message}"); }
        }

        private static Type FindTypeByName(string name)
        {
            foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
            {
                Type[] types;
                try { types = asm.GetTypes(); }
                catch (ReflectionTypeLoadException ex) { types = ex.Types; }  // 取已載入的(含 null)，別整個 assembly 跳過（Assembly-CSharp 常丟這個）
                catch { continue; }
                foreach (var tp in types) if (tp != null && tp.Name == name) return tp;
            }
            return null;
        }

        // 殘局：把玩家與 boss 放到場地內隨機 x（保證間距）、地板稍上方，讓重力沉下去。
        private void PlaceCombatants(HealthManager hm)
        {
            var hero = HeroController.instance;
            if (hero == null || hm == null) return;
            float margin = 2f, minSep = 4f;
            float lo = _heroXMin + margin, hi = _heroXMax - margin;   // 用自動學習的 live 玩家可走範圍
            if (hi <= lo) { lo = _heroXMin; hi = _heroXMax; }
            float px = RandRange(lo, hi), bx = RandRange(lo, hi);
            for (int g = 0; Mathf.Abs(bx - px) < minSep && g < 10; g++) bx = RandRange(lo, hi);
            var hp = hero.transform.position;
            hero.transform.position = new Vector3(px, _floorY + 1f, hp.z);    // 玩家放地板上方一點靠重力沉
            var bp = hm.transform.position;
            hm.transform.position = new Vector3(bx, bp.y, bp.z);              // boss 只改 x、保留原高 y → 自然從空中落下
        }

        // 第一場 capture 場地座標：floor y 用開場玩家落地 y；x 範圍 v1 用玩家 x±8 後備
        // （真正的 arena x-bounds（CameraLockArea）待 probe dump 確認欄位後再換上）。
        private void CaptureArena(HealthManager hm)
        {
            var hero = HeroController.instance;
            float cx = 0f;
            if (hero != null) { _floorY = hero.transform.position.y; cx = hero.transform.position.x; }
            _arenaXMin = cx - 8f; _arenaXMax = cx + 8f;   // 初始 seed：玩家 x±8
            TryCameraLockBounds();                          // 有 CameraLockArea 就覆寫 seed
            if (FinaleArenaXMin > -900f && FinaleArenaXMax > FinaleArenaXMin)
            { _arenaXMin = FinaleArenaXMin; _arenaXMax = FinaleArenaXMax; }   // config 手動覆寫(預設不開)
            // 自動學習：放置範圍從 seed 起，之後每場用觀測到的玩家 x running min/max 持續擴張(見 Update)。
            _heroXMin = _arenaXMin; _heroXMax = _arenaXMax;
            _arenaReady = true;
            _mod?.Info($"[curriculum] arena seed=[{_arenaXMin:0.0},{_arenaXMax:0.0}] floorY={_floorY:0.0}（之後自動擴張）");
        }

        // 用場上最寬的 CameraLockArea 的 cameraXMin/cameraXMax 當 arena x 範圍（反射，不硬引用型別）。
        // 找不到型別/欄位 → 保留 x±8 後備並把真欄位名 log 出來供修正。
        private void TryCameraLockBounds()
        {
            try
            {
                var t = FindTypeByName("CameraLockArea");
                if (t == null) return;
                const BindingFlags BF = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance;
                var fMin = t.GetField("cameraXMin", BF);
                var fMax = t.GetField("cameraXMax", BF);
                if (fMin == null || fMax == null)
                {
                    _mod?.Info("[curriculum] CameraLockArea fields: "
                               + string.Join(",", Array.ConvertAll(t.GetFields(BF), f => f.Name)));
                    return;
                }
                var areas = UnityEngine.Object.FindObjectsOfType(t);
                float bestSpan = -1f, bMin = 0f, bMax = 0f;
                foreach (var o in areas)
                {
                    float xmin = Convert.ToSingle(fMin.GetValue(o));
                    float xmax = Convert.ToSingle(fMax.GetValue(o));
                    if (xmax - xmin > bestSpan) { bestSpan = xmax - xmin; bMin = xmin; bMax = xmax; }
                }
                _mod?.Info($"[curriculum] CameraLockArea: {areas.Length} areas, span={bestSpan:0.0}");
                if (bestSpan > 0f) { _arenaXMin = bMin; _arenaXMax = bMax; }
            }
            catch (Exception e) { _mod?.Info($"[curriculum] CameraLockArea read failed: {e.Message}"); }
        }

        // 第一場 dump 可確認的座標候選值到檔（CameraLockArea/GameManager x-bounds、FSM state 名待補反射）
        private void DumpArenaProbe(HealthManager hm)
        {
            if (File.Exists(_probeFile)) return;   // 只寫一次（已拿到場地資訊就不再覆寫）
            try
            {
                var sb = new System.Text.StringBuilder();
                sb.AppendLine("# arena probe (第一場 capture)；用來挑殘局放位置的正確場地欄位");
                sb.AppendLine($"scene = {ActiveScene()}");
                var hero = HeroController.instance;
                if (hero != null) sb.AppendLine($"hero.pos = {hero.transform.position}  (floorY={_floorY})");
                if (hm != null) sb.AppendLine($"boss.pos = {hm.transform.position}  hp={hm.hp}");
                var pd = PlayerData.instance;
                if (pd != null) sb.AppendLine($"player hp={pd.health}/{pd.maxHealth}");
                sb.AppendLine($"arena x range (v1 fallback) = [{_arenaXMin}, {_arenaXMax}]");
                sb.AppendLine("# TODO: 反射 dump CameraLockArea / GameManager.sceneWidth 拿真 x-bounds");
                File.WriteAllText(_probeFile, sb.ToString());
                _mod.Info($"[curriculum] arena probe dumped -> {_probeFile}");
            }
            catch (Exception e) { _mod.Info($"[curriculum] probe dump failed: {e.Message}"); }
        }

        // 殘局開場期間：反射記錄 boss 各 PlayMakerFSM 的 ActiveStateName（不引用 PlayMaker，純反射），
        // 約每 0.25s 一筆 append 到 probe；給日後把「固定延遲」升級成「偵測 FSM 離開開場 state」。
        // boss 是否還在開場 state（不攻擊）。實測 GG_Hornet 開場序列：
        //   GG Intro 1(站立) → GG Fall(落下) → GG Land(落地) → Flourish(花式) → Run/GDash/...(真攻擊)
        // 開場 state 都有 "gg " 前綴或是 "flourish"；戰鬥 state(Run/Jump/GDash/G Dash/Throw/Sphere/Evade)
        // 都沒有 "gg "（"gdash"/"g dash" 不含 "gg "）→ 用 "gg "||"flourish" 涵蓋完整開場、不誤中戰鬥。
        private bool BossInOpening(HealthManager hm)
        {
            if (hm == null) return false;
            try
            {
                var t = FindTypeByName("PlayMakerFSM");
                var stateProp = t?.GetProperty("ActiveStateName");
                if (stateProp == null) return false;
                foreach (var c in hm.GetComponents<MonoBehaviour>())
                {
                    if (c == null || c.GetType().Name != "PlayMakerFSM") continue;
                    var sn = (stateProp.GetValue(c, null) as string)?.ToLower();
                    if (sn == null) continue;
                    if (sn.Contains("gg ") || sn.Contains("flourish")) return true;
                }
            }
            catch { }
            return false;
        }

        private void LogBossFsmState(HealthManager hm, float t)
        {
            if (hm == null || (_lastFsmLog >= 0f && t - _lastFsmLog < 0.25f)) return;
            _lastFsmLog = t;
            try
            {
                var sb = new System.Text.StringBuilder($"[fsm t={t:0.0}] ");
                foreach (var c in hm.GetComponents<MonoBehaviour>())
                {
                    if (c == null) continue;
                    var ct = c.GetType();
                    if (ct.Name != "PlayMakerFSM") continue;
                    var fn = ct.GetProperty("FsmName")?.GetValue(c, null) as string;
                    var sn = ct.GetProperty("ActiveStateName")?.GetValue(c, null) as string;
                    sb.Append($"{fn}:{sn} ");
                }
                File.AppendAllText(_probeFile, sb.ToString() + "\n");
            }
            catch { }
        }

        private void WriteFinaleFile(bool finale, bool armed, int trueMax, float bossFrac = -1f)
        {
            try
            {
                File.WriteAllText(_finaleFile,
                    $"finale={(finale ? 1 : 0)}\narmed={(armed ? 1 : 0)}\ntrue_max={trueMax}\n"
                    + $"boss_frac={bossFrac.ToString(CultureInfo.InvariantCulture)}\n");
            }
            catch { }
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

            // 自動學習放置 x 範圍：只在「站在初始 floor 高度（onGround 且 y≈floorY）」時才用該 x 擴張。
            // 否則把跳躍中/不同樓層高度的 x 算進去，之後在那個 x 放 (x, floorY+1) 可能卡地形/懸空。
            var heroD = HeroController.instance;
            if (heroD != null)
            {
                var hp = heroD.transform.position;
                // 只在「站在初始 floor 高度」時用該 x 擴張放置範圍（濾掉跳躍/別樓層；見前述）。
                if (heroD.cState.onGround && Mathf.Abs(hp.y - _floorY) < FloorYTol)
                {
                    bool expanded = false;
                    if (hp.x < _heroXMin) { _heroXMin = hp.x; expanded = true; }
                    if (hp.x > _heroXMax) { _heroXMax = hp.x; expanded = true; }
                    if (expanded) _mod.Info($"[curriculum] 放置範圍擴張→[{_heroXMin:0.0},{_heroXMax:0.0}]");
                }
            }

            if (!_fightActive || _resolved) return;

            // 【診斷】殘局 arm 後記錄 boss FSM state（看 armed 後 boss 經過哪些 state、何時真正可攻擊）。
            if (FinaleDebug && _finaleThisFight && _finaleCount <= FsmProbeFights && _finaleActivePolls < FinaleFsmPolls)
            {
                LogBossFsmState(_boss, _finaleActivePolls * PollInterval);
                _finaleActivePolls++;
            }

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
            if (_finaleThisFight)
            {
                _mod.Info($"[curriculum] finale fight done (win={win})，不計入自適應");
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

        // ---- 可調參數 config（不存在則寫預設；編輯後重啟 HK 生效）----
        private void LoadConfig()
        {
            var kv = new Dictionary<string, string>();
            try
            {
                if (File.Exists(_configFile))
                    foreach (var line in File.ReadAllLines(_configFile))
                    {
                        var t = line.Trim();
                        if (t.Length == 0 || t.StartsWith("#")) continue;
                        int eq = t.IndexOf('=');
                        if (eq > 0) kv[t.Substring(0, eq).Trim()] = t.Substring(eq + 1).Trim();
                    }
            }
            catch { }
            ScaleMin = GetF(kv, "scale_min", ScaleMin);
            ScaleMax = GetF(kv, "scale_max", ScaleMax);
            ScaleStep = GetF(kv, "scale_step", ScaleStep);
            Window = GetI(kv, "window", Window);
            RaiseAbove = GetF(kv, "raise_above", RaiseAbove);
            LowerBelow = GetF(kv, "lower_below", LowerBelow);
            BossMinHp = GetI(kv, "boss_min_hp", BossMinHp);
            WaitFrames = GetI(kv, "wait_frames", WaitFrames);
            PollInterval = GetF(kv, "poll_interval", PollInterval);
            FinaleEvery = GetI(kv, "finale_every", FinaleEvery);
            FinalePlayerMinMasks = GetI(kv, "finale_player_min_masks", FinalePlayerMinMasks);
            FinalePlayerMaxMasks = GetI(kv, "finale_player_max_masks", FinalePlayerMaxMasks);
            FinaleBossMinFrac = GetF(kv, "finale_boss_min_frac", FinaleBossMinFrac);
            FinaleBossMaxFrac = GetF(kv, "finale_boss_max_frac", FinaleBossMaxFrac);
            FinaleOpeningDelay = GetF(kv, "finale_opening_delay", FinaleOpeningDelay);
            FinaleSettleDelay = GetF(kv, "finale_settle_delay", FinaleSettleDelay);
            FinaleArenaXMin = GetF(kv, "finale_arena_xmin", FinaleArenaXMin);
            FinaleArenaXMax = GetF(kv, "finale_arena_xmax", FinaleArenaXMax);
            FinaleDebug = GetI(kv, "finale_debug", 0) != 0;
            if (!File.Exists(_configFile)) WriteDefaultConfig();
            _mod?.Info($"[curriculum] config: scaleMin={ScaleMin:0.00} step={ScaleStep:0.00} "
                       + $"window={Window} raise={RaiseAbove:0.00} finaleEvery={FinaleEvery}");
        }

        private static float GetF(Dictionary<string, string> kv, string k, float dflt) =>
            kv.TryGetValue(k, out var v) &&
            float.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out var f) ? f : dflt;

        private static int GetI(Dictionary<string, string> kv, string k, int dflt) =>
            kv.TryGetValue(k, out var v) &&
            int.TryParse(v, NumberStyles.Integer, CultureInfo.InvariantCulture, out var i) ? i : dflt;

        private void WriteDefaultConfig()
        {
            try
            {
                string I(float f) => f.ToString(CultureInfo.InvariantCulture);
                var sb = new System.Text.StringBuilder();
                sb.AppendLine("# HKCurriculum 可調參數。編輯後重啟 HK 生效。註解須自成一行（# 開頭）；");
                sb.AppendLine("# 不可寫成 key=value # 註解，否則值會含註解字串而解析失敗、退回預設。");
                sb.AppendLine("# === 自適應難度（作法D：放大傷害壓縮整場）===");
                sb.AppendLine("# scale 下限/上限：難度比例範圍。0.50=傷害放大2倍最好贏；1.00=真實滿難度。");
                sb.AppendLine($"scale_min={I(ScaleMin)}");
                sb.AppendLine($"scale_max={I(ScaleMax)}");
                sb.AppendLine("# 每次升/降難的步階。小步(0.02)較穩、解棘輪崩潰；想爬快可 0.05（風險：橫跳硬砸 policy）。");
                sb.AppendLine($"scale_step={I(ScaleStep)}");
                sb.AppendLine("# 滑動勝率視窗(場)。須 > eps/update(8) 否則 thrashing。建議 30；還 ping-pong 就 40-50。");
                sb.AppendLine($"window={Window}");
                sb.AppendLine("# 勝率 > raise_above 升難 / < lower_below 降難（中間死區維持）。建議 0.70 / 0.30。");
                sb.AppendLine($"raise_above={I(RaiseAbove)}");
                sb.AppendLine($"lower_below={I(LowerBelow)}");
                sb.AppendLine("# 視為 boss 的最低滿血(過濾雜魚)；等 FSM 設好滿血的幀數。一般不用動。");
                sb.AppendLine($"boss_min_hp={BossMinHp}");
                sb.AppendLine($"wait_frames={WaitFrames}");
                sb.AppendLine("# 勝負輪詢間隔(秒)。1/15≈0.0667=15Hz，對齊控制頻率即可。");
                sb.AppendLine($"poll_interval={I(PollInterval)}");
                sb.AppendLine("# === 殘局模式 (finale practice)：殘血開局練收尾 ===");
                sb.AppendLine("# 注意：Python 端 config.FINALE_ENABLED 也要 True 才會啟用握手。");
                sb.AppendLine("# finale_every>0：確定性「每 N 場(非eval)的第 N 場殘局」(固定順序)。0=停用。");
                sb.AppendLine("# 例 5＝前4場正常、第5場殘局。第一場永遠正常(不受此值影響)。");
                sb.AppendLine($"finale_every={FinaleEvery}");
                sb.AppendLine("# 殘局玩家起始血＝整數面具隨機 [min,max]。建議 2-3（釘真實死亡區=1-2面具）。");
                sb.AppendLine("# 取整數面具(非先取百分比再 floor)避免玩家實際比例偏低。");
                sb.AppendLine($"finale_player_min_masks={FinalePlayerMinMasks}");
                sb.AppendLine($"finale_player_max_masks={FinalePlayerMaxMasks}");
                sb.AppendLine("# 殘局 boss 起始血＝真實滿血的百分比 [min,max]（與玩家面具各自抽、解耦）。建議 0.20-0.30：");
                sb.AppendLine("# 玩家 AND boss 都釘在「真實 EVAL 死亡區」(打掉~77%後死的低血殊死)＝練 HUD-conditioned actor");
                sb.AppendLine("# 真正會輸的低血收尾分支(舊 5-6 面具版練健康收殘血、HUD 情境不對不轉移)。boss 夠低才贏得到，");
                sb.AppendLine("# boss_frac/player_frac 可重疊。boss_frac 僅供 Python ARMED log、不進 reward。");
                sb.AppendLine($"finale_boss_min_frac={I(FinaleBossMinFrac)}");
                sb.AppendLine($"finale_boss_max_frac={I(FinaleBossMaxFrac)}");
                sb.AppendLine("# 開場等待上限(秒)：實際是等 boss FSM 離開開場 state(含 intro)才放行、提早結束；");
                sb.AppendLine("# 此值僅當 FSM 偵測不到時的 fallback 上限。建議 2.0。");
                sb.AppendLine("# ★必須 < Python config.FINALE_ARM_TIMEOUT(6.0)，否則 Python 等不到 armed 會當正常場。");
                sb.AppendLine($"finale_opening_delay={I(FinaleOpeningDelay)}");
                sb.AppendLine("# 設殘血/移位後再等幾秒讓物理(角色落地)/遙測/面具穩定才 arm。建議 0.3。");
                sb.AppendLine($"finale_settle_delay={I(FinaleSettleDelay)}");
                sb.AppendLine("# 殘局放置 x 範圍覆寫（> -900 才生效；否則用 CameraLockArea/玩家x±8 後備）。");
                sb.AppendLine("# GG_Hornet_1 實測可走 [15.3,37.7]（CameraLockArea 在此場給 span=0 不可用）。");
                sb.AppendLine($"finale_arena_xmin={I(FinaleArenaXMin)}");
                sb.AppendLine($"finale_arena_xmax={I(FinaleArenaXMax)}");
                sb.AppendLine("# 診斷開關：1=寫 arena_probe/hud_probe/FSM log/[heropos] spam（換王重 probe 才開）。預設 0。");
                sb.AppendLine($"finale_debug={(FinaleDebug ? 1 : 0)}");
                File.WriteAllText(_configFile, sb.ToString());
                _mod?.Info($"[curriculum] wrote default config -> {_configFile}");
            }
            catch { }
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
