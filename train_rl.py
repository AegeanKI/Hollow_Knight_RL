"""PPO 微調：從 BC 權重出發，在真實大黃蜂戰鬥中學習。

- 動作隨機取樣（探索），每 EPISODES_PER_UPDATE 場（預設 8）做一次 PPO 更新。
- 更新發生在 episode 之間（人在雕像大廳、不在戰鬥），不搶即時操作的 GPU。
- F10 = 安全停止（會存檔後結束）。不用 Esc，因為 HK 內看設定要按 Esc。
- F9 = 暫停/繼續（在 episode 之間生效，會放開所有輸入，方便檢查映射/難度/護符）。

穩定性與訊號品質：
- 自動開場失敗不會讓訓練崩潰：env.reset() 退避重試，連續失敗就跳過該場。
- 遙測健康度：每場統計掉包率，>50% 視為 reward 不可信 -> 整場丟棄不納入更新；
  本輪平均掉包 >20% 會印警告（去查 reward mod / UDP）。

Checkpoint（都在 checkpoints/，內含 模型+optimizer+更新次數+場數，接續無縫）：
  rl_latest.pt   每次更新滾動覆蓋（最新）
  rl_best.pt     **由決定性 eval 的平均傷害**創新高才覆蓋（--eval-every 0 時退回用訓練 avg_dmg）
  rl_uXXXX.pt    每 --snapshot-every 次更新存一個「不覆蓋」的編號快照（歷史回溯點）

指標：每次 update 寫一列到 logs/metrics.csv（train/eval 傷害、勝場、loss、entropy、kl、
掉包率），方便畫曲線/跨 run 比較；判斷是否進步看這條曲線，不看單點。

用法：
  python train_rl.py                       # 全新：從 bc.pt 初始化開始
  python train_rl.py --resume              # 接續 rl_latest.pt
  python train_rl.py --ckpt rl_best.pt     # 從最佳那個接續
  python train_rl.py --ckpt rl_u0020.pt    # 從第 20 次更新的快照接續
  python train_rl.py --ckpt D:/some/dir/xxx.pt # 也吃完整路徑（--ckpt 優先於 --resume）
  python train_rl.py --snapshot-every 5    # 每 5 次更新存一個編號快照（0=關閉）
  python train_rl.py --eval-every 5 --eval-episodes 5  # 評估頻率/場數（rl_best 由此選；0=關閉 eval）
"""
import argparse
import csv
import os
import time
from collections import namedtuple

import numpy as np
import torch

import config
import curriculum
from ac_model import ActorCritic
from controls import ControlKeys
from env import BossDamageTracker, HollowKnightEnv
from ppo import RolloutBuffer, RunningMeanStd, ppo_update

# 固定輸入尺寸 -> 讓 cudnn 選好演算法（也避開 CUDNN_STATUS_NOT_SUPPORTED 的 plan warning）
torch.backends.cudnn.benchmark = True

LATEST = os.path.join(config.CKPT_DIR, config.RL_LATEST_CKPT)
BEST = os.path.join(config.CKPT_DIR, config.RL_BEST_CKPT)
LOG_PATH = os.path.join("logs", "train_rl.log")
CSV_PATH = os.path.join("logs", "metrics.csv")   # C8：每次 update 一列，方便畫曲線/比較

# B4 遙測健康門檻
TELE_DROP_WARN = 0.20       # 掉包率超過 -> 警告（訊號開始不可信）
TELE_DROP_DISCARD = 0.50    # 掉包率超過 -> 整場丟棄不納入更新（reward 已不可信）

CSV_COLS = ["update", "ep", "train_avg_dmg", "train_wins", "train_eps_used",
            "pi_loss", "vf_loss", "entropy", "kl", "tele_drop_mean",
            "eval_avg_dmg", "eval_max_dmg", "eval_wins", "eval_eps", "scale"]


def _open_csv():
    """開 metrics.csv（append）；欄位變更時把舊檔改名成 .old 再開新檔。
    舊檔被佔用（Excel/另一程序開著）導致改名失敗時，改寫到帶序號的新檔，避免訓練起不來。"""
    path = CSV_PATH
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            header_match = f.readline().strip() == ",".join(CSV_COLS)
        if not header_match:                                 # 欄位改過
            try:
                os.replace(path, path + ".old")              # 備份舊檔
            except OSError:                                  # 舊檔被鎖，改寫不衝突的新檔
                i = 1
                while os.path.exists(f"{CSV_PATH}.{i}"):
                    i += 1
                path = f"{CSV_PATH}.{i}"
                print(f"⚠ {CSV_PATH} 被佔用（Excel/另一程序開著？），指標改寫到 {path}")
    is_new = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=CSV_COLS)
    if is_new:
        w.writeheader()
        f.flush()
    return f, w


def save_ckpt(path, ac, opt, update_i, ep_i, best_dmg, ret_rms=None):
    ck = {"model": ac.state_dict(), "opt": opt.state_dict(),
          "update_i": update_i, "ep_i": ep_i, "best_dmg": best_dmg}
    if ret_rms is not None:
        ck["ret_rms"] = ret_rms.state_dict()      # ① return 正規化狀態（接續才一致）
    torch.save(ck, path)


def eval_one_episode(env, ac, device, should_stop):
    """跑單場決定性（不取樣）戰鬥，回傳 (result, damage)。
    無法完成（reset 失敗/被中止、或 0 步）回傳 None。"""
    obs, _ = env.reset(should_stop=should_stop)
    if obs is None:                              # reset 失敗/被中止
        return None
    boss = BossDamageTracker()
    done, info = False, {}
    while not done and not should_stop():
        ot = torch.from_numpy(obs).to(device)
        a, _, _ = ac.act(ot, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        boss.update(info)
    if not info:                                 # 0 步（一進去就被 stop）
        return None
    return info["result"], boss.dmg


def run_eval(env, ac, device, n_eps, should_stop):
    """跑 n_eps 場決定性評估，回傳 (results, damages)。
    全程設 curriculum eval 旗標 -> boss 固定 100% 滿血、不計入自適應難度。"""
    curriculum.begin_eval()
    try:
        res, dmgs = [], []
        for _ in range(n_eps):
            if should_stop():
                break
            ep = eval_one_episode(env, ac, device, should_stop)
            if ep is None:                       # reset 失敗/被中止/0 步 -> 停止評估
                break
            result, dmg = ep
            res.append(result)
            dmgs.append(dmg)
        return res, dmgs
    finally:
        curriculum.end_eval()


# 一場訓練 episode 收集到的資料（trans=PPO transitions；其餘為統計用）
# scale=這場實際打的難度（reset 後讀，mod 開場設定後到本場結束才會變）；mod 沒載入則 None
EpisodeData = namedtuple("EpisodeData", "trans result dmg ep_r steps drop scale")


def collect_one_episode(env, ac, device, should_stop):
    """跑單場訓練戰鬥（動作取樣探索），收集 transitions 與統計。
    回傳 EpisodeData；無法完成（reset 失敗/被中止、或 0 步）回傳 None。
    收不收進 buffer（遙測健康度）由呼叫端決定。"""
    obs, _ = env.reset(should_stop=should_stop)
    if obs is None:                              # 自動開場失敗/被中止
        return None
    scale = curriculum.read_scale()              # 這場實際打的難度（開場已定、本場固定）
    boss = BossDamageTracker()
    done, ep_r, steps, info = False, 0.0, 0, {}
    trans = []
    while not done and not should_stop():
        ot = torch.from_numpy(obs).to(device)
        action, logp, value = ac.act(ot)
        obs2, r, term, trunc, info = env.step(action)
        done = term or trunc
        # 超時(truncation)非真終局：用最後狀態 value bootstrap，避免低估結尾
        boot = ac.value(torch.from_numpy(obs2).to(device)) if (trunc and not term) else 0.0
        trans.append((obs, action, logp, r, value, float(done), float(term), boot))
        obs = obs2; ep_r += r; steps += 1
        boss.update(info)
    if steps == 0:                               # 0 步（剛 reset 完就被 stop）
        return None
    return EpisodeData(trans=trans, result=info["result"], dmg=boss.dmg,
                       ep_r=ep_r, steps=steps, drop=info.get("tele_drop", 0.0), scale=scale)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true", help="接續 rl_latest.pt")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="從指定 checkpoint 接續（路徑或 checkpoints/ 下的檔名），優先於 --resume")
    ap.add_argument("--snapshot-every", type=int, default=10,
                    help="每幾次更新存一個不覆蓋的編號快照 rl_uXXXX.pt（0=關閉）")
    ap.add_argument("--updates", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--ent-coef", type=float, default=0.0005, help="熵獎勵係數（太大策略會變隨機）")
    ap.add_argument("--episodes-per-update", type=int, default=config.EPISODES_PER_UPDATE,
                    help="每次更新收集幾場（越大梯度越穩）")
    ap.add_argument("--eval-every", type=int, default=5,
                    help="每幾次更新跑一次決定性評估（0=關閉）。rl_best 由此評估的傷害選出。")
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--input", choices=["keyboard", "gamepad"], default=config.INPUT_BACKEND,
                    help="輸入後端：keyboard(需焦點) 或 gamepad(虛擬手把，背景可、解放鍵盤)")
    return ap.parse_args()


def main():
    args = parse_args()

    os.makedirs("logs", exist_ok=True)
    logf = open(LOG_PATH, "a", encoding="utf-8")

    def log(msg):
        print(msg)
        logf.write(msg + "\n")
        logf.flush()

    log(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} 開始/接續訓練 "
        f"(ent_coef={args.ent_coef}, eps/update={args.episodes_per_update}) =====")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ac = ActorCritic().to(device)
    opt = torch.optim.Adam(ac.parameters(), lr=args.lr)
    ret_rms = RunningMeanStd()                    # ① return 正規化的跑動尺度
    update_i, ep_i, best_dmg = 0, 0, -1.0

    # 決定要從哪載入：--ckpt 指定 > --resume(latest) > 從 BC 初始化
    resume_path = None
    if args.ckpt:
        resume_path = args.ckpt if os.path.exists(args.ckpt) \
            else os.path.join(config.CKPT_DIR, args.ckpt)
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"找不到 checkpoint: {args.ckpt}")
    elif args.resume and os.path.exists(LATEST):
        resume_path = LATEST

    if resume_path:
        ck = torch.load(resume_path, map_location=device)
        ac.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        update_i, ep_i, best_dmg = ck["update_i"], ck["ep_i"], ck["best_dmg"]
        if "ret_rms" in ck:                       # ① 接續時還原 return 尺度；舊檔沒有就從頭估
            ret_rms.load_state_dict(ck["ret_rms"])
        log(f"接續訓練 from {resume_path}：update={update_i} ep={ep_i} best_dmg={best_dmg:.0f}")
    else:
        bc = torch.load(os.path.join(config.CKPT_DIR, config.BC_CKPT), map_location=device)
        ac.init_from_bc(bc["model"])
        log(f"從 BC 初始化 (macroF1 {bc['macroF1']:.3f})")

    ctrl = ControlKeys().start()

    print(f"{config.START_COUNTDOWN_SEC} 秒後開始，請點一下遊戲視窗取得焦點...（F10 安全停止；F9 暫停/繼續）")
    for i in range(config.START_COUNTDOWN_SEC, 0, -1):
        print(f"  {i}..."); time.sleep(1)

    env = HollowKnightEnv(backend=args.input)
    log(f"輸入後端：{args.input}")
    csvf, csvw = _open_csv()
    try:
        while update_i < args.updates and not ctrl.stop:
            buf = RolloutBuffer()
            ep_summ = []
            drops = []
            scales = []
            for _ in range(args.episodes_per_update):
                if ctrl.stop:                        # 上一場跑完就收到停止 -> 乾淨退出
                    break
                # 暫停點（episode 之間）：放開所有輸入，等使用者檢查遊戲設定後再續
                ctrl.wait_while_paused(on_pause=env.act.release_all, log=log)
                if ctrl.stop:                        # 暫停等待中按了 F10 -> 別再開下一場
                    break
                ep = collect_one_episode(env, ac, device, ctrl.should_stop)
                if ep is None:                       # B3：自動開場失敗/被中止/0 步 -> 跳過本場，不崩
                    continue
                ep_i += 1
                drops.append(ep.drop)
                scales.append(ep.scale)
                # B4：遙測掉太兇 -> reward 訊號不可信，整場丟棄不納入更新（但仍記錄）
                healthy = ep.drop <= TELE_DROP_DISCARD
                if healthy:
                    for tr in ep.trans:
                        buf.add(*tr)
                flag = ("" if ep.drop <= TELE_DROP_WARN
                        else f"  ⚠遙測掉包{ep.drop:.0%}" + ("（已丟棄本場）" if not healthy else ""))
                ep_summ.append((ep.result, ep.dmg, ep.ep_r, ep.steps, healthy))
                # 括號內 = 實際打出的傷害 = dmg×scale（扣掉作法D的放大；跨 scale 可比）
                real_tag = f" ({ep.dmg * ep.scale:.0f})" if ep.scale is not None else ""
                scale_tag = f" scale={ep.scale:.2f}" if ep.scale is not None else ""
                log(f"  EP{ep_i}: {str(ep.result):>5} dmg={ep.dmg:4.0f}{real_tag} "
                    f"reward={ep.ep_r:6.2f} steps={ep.steps}{flag}{scale_tag}")

            if len(buf) == 0:                        # 本輪沒有任何可用資料（全失敗/全丟棄/被停）
                if ctrl.stop:
                    break
                log("  ⚠ 本輪沒有可用 episode（開場失敗或遙測全壞），略過此次更新。")
                continue
            st = ppo_update(ac, opt, buf, device, ret_rms=ret_rms, ent_coef=args.ent_coef)
            update_i += 1
            used = [e for e in ep_summ if e[4]]      # 真正納入更新的（遙測健康）場
            avg_dmg = float(np.mean([d for _, d, _, _, _ in used])) if used else 0.0
            wins = sum(1 for r, _, _, _, _ in used if r == "win")
            drop_mean = float(np.mean(drops)) if drops else 0.0
            # 括號內 = 實際傷害平均 = 逐場 dmg×scale 再平均（扣掉作法D放大；跨 scale 可比）
            real = [d * s for (_, d, _, _, h), s in zip(ep_summ, scales) if h and s is not None]
            avg_real = float(np.mean(real)) if real else None
            real_tag = f" ({avg_real:.0f})" if avg_real is not None else ""
            # 本輪各場實際難度（每場開場時讀）；可能在 update 中途被調過，故 log 顯示範圍
            sc = [s for s in scales if s is not None]
            scale = sc[-1] if sc else None       # CSV 記最後一場（最接近當下）；None=mod 沒載入
            scale_str = ("" if not sc else
                         (f"{min(sc):.2f}" if min(sc) == max(sc) else f"{min(sc):.2f}-{max(sc):.2f}"))
            log(f"[UPDATE {update_i}] avg_dmg={avg_dmg:.0f}{real_tag} wins={wins}/{len(used)} "
                f"pi={st['pi_loss']:.3f} vf={st['vf_loss']:.3f} ent={st['entropy']:.2f} "
                f"kl={st['kl']:.3f} retσ={st['ret_std']:.1f} tele_drop={drop_mean:.0%}"
                + (f" scale={scale_str}" if scale_str else ""))
            if drop_mean > TELE_DROP_WARN:
                log(f"  ⚠ 本輪平均遙測掉包 {drop_mean:.0%}，請檢查 reward mod / UDP 是否正常。")

            save_ckpt(LATEST, ac, opt, update_i, ep_i, best_dmg, ret_rms)
            if args.snapshot_every > 0 and update_i % args.snapshot_every == 0:
                snap = os.path.join(config.CKPT_DIR, f"rl_u{update_i:04d}.pt")
                save_ckpt(snap, ac, opt, update_i, ep_i, best_dmg, ret_rms)
                log(f"  保存快照 {snap}")

            # ---- B6：rl_best 由「決定性評估的傷害」選出，而非訓練取樣的 avg_dmg ----
            eval_dmg = eval_max = eval_wins = eval_n = None
            if args.eval_every > 0 and update_i % args.eval_every == 0 and not ctrl.stop:
                res, dmgs = run_eval(env, ac, device, args.eval_episodes,
                                     should_stop=ctrl.should_stop)
                if dmgs:
                    eval_dmg, eval_max = float(np.mean(dmgs)), float(np.max(dmgs))
                    eval_wins, eval_n = sum(1 for r in res if r == "win"), len(res)
                    log(f"  [EVAL] avg_dmg={eval_dmg:.0f} max={eval_max:.0f} "
                        f"wins={eval_wins}/{eval_n} results={res}")
                    if eval_dmg > best_dmg:
                        best_dmg = eval_dmg
                        save_ckpt(BEST, ac, opt, update_i, ep_i, best_dmg, ret_rms)
                        log(f"  ★ 新最佳(eval)平均傷害 {best_dmg:.0f}，存 rl_best.pt")
                        save_ckpt(LATEST, ac, opt, update_i, ep_i, best_dmg, ret_rms)  # 同步 latest 的 best 欄
            elif args.eval_every == 0:
                # 沒開 eval 的退路：退回用訓練 avg_dmg 維持 best（legacy 行為）
                if avg_dmg > best_dmg:
                    best_dmg = avg_dmg
                    save_ckpt(BEST, ac, opt, update_i, ep_i, best_dmg, ret_rms)
                    save_ckpt(LATEST, ac, opt, update_i, ep_i, best_dmg, ret_rms)
                    log(f"  ★ 新最佳(train)平均傷害 {best_dmg:.0f}，存 rl_best.pt")

            # ---- C8：每次 update 寫一列到 metrics.csv ----
            csvw.writerow({
                "update": update_i, "ep": ep_i,
                "train_avg_dmg": round(avg_dmg, 1), "train_wins": wins,
                "train_eps_used": len(used),
                "pi_loss": round(st["pi_loss"], 4), "vf_loss": round(st["vf_loss"], 4),
                "entropy": round(st["entropy"], 4), "kl": round(st["kl"], 4),
                "tele_drop_mean": round(drop_mean, 4),
                "eval_avg_dmg": "" if eval_dmg is None else round(eval_dmg, 1),
                "eval_max_dmg": "" if eval_max is None else round(eval_max, 1),
                "eval_wins": "" if eval_wins is None else eval_wins,
                "eval_eps": "" if eval_n is None else eval_n,
                "scale": "" if scale is None else round(scale, 4),
            })
            csvf.flush()
    finally:
        env.close()
        save_ckpt(LATEST, ac, opt, update_i, ep_i, best_dmg, ret_rms)
        log(f"已停止並存檔。update={update_i} ep={ep_i}")
        csvf.close()
        logf.close()


if __name__ == "__main__":
    main()
