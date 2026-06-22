"""離線驗證 RolloutBuffer.finish() 的 truncate / terminal / 切斷 三種收尾。
不依賴遊戲環境，純數值比對 GAE。"""
import numpy as np
from ppo import RolloutBuffer

GAMMA, LAM = 0.99, 0.95


def make_buf(term, trunc, boot_val, n=3, rew=1.0, val=5.0):
    """造一條 n 步、固定 rew/val 的 episode，最後一步帶 (term, trunc, boot)。"""
    buf = RolloutBuffer()
    for t in range(n):
        last = (t == n - 1)
        done = float((term or trunc) and last)
        tm = float(term and last)
        boot = boot_val if (last and trunc and not term) else 0.0
        buf.add(obs=np.zeros(1, np.float32), act=np.zeros(1, np.float32),
                logp=0.0, rew=rew, val=val, done=done, term=tm, boot=boot)
    return buf


def manual_gae(rew, val, next_val_last, term_last, trunc_last):
    """手算 GAE 當對照組。"""
    n = len(rew)
    adv = np.zeros(n); last = 0.0
    for t in reversed(range(n)):
        if t == n - 1:
            nonterminal = 0.0 if term_last else 1.0
            next_val = next_val_last
            done = 1.0 if (term_last or trunc_last) else 0.0
        else:
            nonterminal = 1.0; next_val = val[t + 1]; done = 0.0
        delta = rew[t] + GAMMA * next_val * nonterminal - val[t]
        last = delta + GAMMA * LAM * (1.0 - done) * last
        adv[t] = last
    return adv, adv + np.asarray(val)


def check(name, buf, next_val_last, term_last, trunc_last):
    adv, ret = buf.finish(GAMMA, LAM)
    rew = np.asarray(buf.rew); val = np.asarray(buf.val)
    eadv, eret = manual_gae(rew, val, next_val_last, term_last, trunc_last)
    ok = np.allclose(adv, eadv) and np.allclose(ret, eret)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"       adv={np.round(adv,4)}  ret={np.round(ret,4)}")
    assert ok, f"{name} mismatch:\n  got {adv}\n  exp {eadv}"
    return adv


VAL, BOOT = 5.0, 8.0

# 1) 真終局（死/勝）：未來價值砍 0
a_term = check("terminal (死/勝, 未來=0)",
               make_buf(term=True,  trunc=False, boot_val=0.0),
               next_val_last=0.0, term_last=True,  trunc_last=False)

# 2) 超時 truncate（被切斷）：用 V(s_next) bootstrap
a_trunc = check("truncate (超時, bootstrap V)",
                make_buf(term=False, trunc=True,  boot_val=BOOT),
                next_val_last=BOOT, term_last=False, trunc_last=True)

# 3) F10 中途切斷（buffer 末步且非邊界）：保守 next_val=0
a_cut = check("mid-cut (F10 切斷, 保守=0)",
              make_buf(term=False, trunc=False, boot_val=0.0),
              next_val_last=0.0, term_last=False, trunc_last=False)

print("\n--- 關鍵差異 (最後一步 advantage) ---")
print(f"terminal 末步 adv = {a_term[-1]:+.4f}   (delta = {1.0 + 0:.4f} - {VAL} = {1.0-VAL:+.4f})")
print(f"truncate 末步 adv = {a_trunc[-1]:+.4f}   (delta = {1.0} + 0.99*{BOOT} - {VAL} = {1.0+GAMMA*BOOT-VAL:+.4f})")
assert a_trunc[-1] > a_term[-1], "truncate 應因 bootstrap 而高於 terminal"
print(f"\n[OK] truncate 末步比 terminal 高 {a_trunc[-1]-a_term[-1]:+.4f}（bootstrap 生效，未把結尾未來低估成 0）")
print("\n全部通過。")
