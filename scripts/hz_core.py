#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hz_core —— 杭州麻将（白板财神 / 纯自摸）精确核心。

设计目标：
  1. 向听数对财神（白板万能）**精确**：不再用「库向听 − 财神数」的近似。
  2. 统一支持副露（吃/碰/杠折算面子）、七对（禁副露；四张=2 对）。
  3. 快（全手牌 LRU + 每花色分解记忆化），供搜索引擎每次决策成百上千次调用。

牌编码：索引 0..33
   0- 8  1w..9w（万）    9-17  1b..9b（筒）   18-26  1t..9t（条）
  27 东  28 南  29 西  30 北  31 中  32 发  33 白(财神)
手牌统一用长度 34 的计数数组表示。

向听数定义（含财神）：
  结构 = N 面子 + 1 将（N = 4 − 副露数）。对每种拆分统计 R = 能放进结构的真牌数
  （完整面子 3、搭子 2、将 2、剩余每个空槽用 1 张孤张打底），
  财神 g 张各可顶替任意 1 张缺牌，于是
      向听 = (3N + 1) − R_max − g        （13 张形：≥0；14 张形：−1 即已胡）
  七对：向听 = 6 − 对子数 − g（clamp 0；14 张形 −1 即胡）。
"""
from functools import lru_cache

# ---------------- 牌码 ----------------
SUITS = "wbt"
HONORS = ["东", "南", "西", "北", "中", "发", "白"]
GOD = 33
TILE_NAMES = [f"{r}{s}" for s in SUITS for r in range(1, 10)] + HONORS
TILE_INDEX = {n: i for i, n in enumerate(TILE_NAMES)}
N_KINDS = 34


def tile_to_idx(t):
    return TILE_INDEX[t]


def idx_to_tile(i):
    return TILE_NAMES[i]


def hand_to_counts(hand):
    c = [0] * 34
    for t in hand:
        c[TILE_INDEX[t]] += 1
    return c


def counts_to_hand(counts):
    out = []
    for i, n in enumerate(counts):
        out.extend([TILE_NAMES[i]] * n)
    return out


def is_suited(i):
    return i < 27


def suit_of(i):
    return i // 9 if i < 27 else 3


def rank_of(i):
    return i % 9 + 1 if i < 27 else 0


# ---------------- 单花色分解（记忆化） ----------------
# 返回 frozenset 的 (m, t, p)：m 完整面子数、t 搭子数（含对子当搭子）、p 是否用了将(0/1)

_suit_memo = {}
_honor_memo = {}


def _decomp(counts9, honor):
    key = tuple(counts9)
    memo = _honor_memo if honor else _suit_memo
    r = memo.get(key)
    if r is not None:
        return r
    c = list(counts9)
    n = len(c)
    out = set()
    sub = {}

    def rec(i, p):
        while i < n and c[i] == 0:
            i += 1
        if i == n:
            return {(0, 0, 0)}
        k = (tuple(c[i:]), i, p)
        got = sub.get(k)
        if got is not None:
            return got
        res = set()
        # 孤张
        c[i] -= 1
        res |= rec(i, p)
        c[i] += 1
        # 刻子
        if c[i] >= 3:
            c[i] -= 3
            res |= {(m + 1, t, q) for (m, t, q) in rec(i, p)}
            c[i] += 3
        # 对子：做将 / 做搭子
        if c[i] >= 2:
            c[i] -= 2
            if p == 0:
                res |= {(m, t, 1) for (m, t, q) in rec(i, 1)}
            res |= {(m, t + 1, q) for (m, t, q) in rec(i, p)}
            c[i] += 2
        if not honor:
            # 顺子
            if i <= n - 3 and c[i + 1] > 0 and c[i + 2] > 0:
                c[i] -= 1; c[i + 1] -= 1; c[i + 2] -= 1
                res |= {(m + 1, t, q) for (m, t, q) in rec(i, p)}
                c[i] += 1; c[i + 1] += 1; c[i + 2] += 1
            # 两面/边张搭子
            if i <= n - 2 and c[i + 1] > 0:
                c[i] -= 1; c[i + 1] -= 1
                res |= {(m, t + 1, q) for (m, t, q) in rec(i, p)}
                c[i] += 1; c[i + 1] += 1
            # 坎张搭子
            if i <= n - 3 and c[i + 2] > 0:
                c[i] -= 1; c[i + 2] -= 1
                res |= {(m, t + 1, q) for (m, t, q) in rec(i, p)}
                c[i] += 1; c[i + 2] += 1
        # Pareto 剪枝：去掉被支配的项
        res = _pareto(res)
        sub[k] = res
        return res

    out = rec(0, 0)
    memo[key] = out
    return out


def _pareto(s):
    if len(s) <= 1:
        return s
    items = list(s)
    keep = []
    for a in items:
        dominated = False
        for b in items:
            if b != a and b[0] >= a[0] and b[1] >= a[1] and b[2] >= a[2]:
                dominated = True
                break
        if not dominated:
            keep.append(a)
    return set(keep)


def _merge(A, B):
    out = set()
    for (m1, t1, p1) in A:
        for (m2, t2, p2) in B:
            p = p1 + p2
            t = t1 + t2
            if p > 1:
                t += p - 1
                p = 1
            out.add((m1 + m2, t, p))
    return _pareto(out)


# ---------------- 向听 / 胡牌 ----------------

@lru_cache(maxsize=1 << 18)
def _shanten_normal_key(key, n_melds):
    counts = key
    N = 4 - n_melds
    g = counts[33]
    real = sum(counts[:33])
    if N <= 0:
        # 4 副露：手里 1 张(13形) 或 2 张(14形)，只需将
        if real + g >= 2:
            # 14 形：两张成对或有财神 → 胡
            if real == 2:
                return -1 if any(counts[i] == 2 for i in range(33)) else 0
            return -1
        return 0
    sets = _decomp(counts[0:9], False)
    sets = _merge(sets, _decomp(counts[9:18], False))
    sets = _merge(sets, _decomp(counts[18:27], False))
    sets = _merge(sets, _decomp(counts[27:33], True))
    best = -1
    for (m, t, p) in sets:
        if m > N:
            m = N
        if t > N - m:
            t = N - m
        L = real - 3 * m - 2 * t - 2 * p
        seeds = (N - m - t) + (1 - p)
        if L < seeds:
            seeds = L
        R = 3 * m + 2 * t + 2 * p + seeds
        if R > best:
            best = R
    return 3 * N + 1 - best - g


def shanten_normal(counts, n_melds=0):
    """普通胡型向听（13 形 ≥0；14 形 −1 = 已胡）。"""
    return _shanten_normal_key(tuple(counts), n_melds)


def shanten_qidui(counts):
    """七对向听（无副露才有意义；14 形 −1 = 已胡）。"""
    g = counts[33]
    pairs = 0
    for i in range(33):
        pairs += counts[i] >> 1
    s = 6 - pairs - g
    total = sum(counts)
    if total <= 13:
        return s if s > 0 else 0
    return s if s > -1 else -1


def shanten(counts, n_melds=0):
    s = shanten_normal(counts, n_melds)
    if n_melds == 0:
        s7 = shanten_qidui(counts)
        if s7 < s:
            s = s7
    return s


def is_win(counts, n_melds=0):
    """14 形（3N+2 张）能否胡。"""
    if shanten_normal(counts, n_melds) <= -1:
        return True
    if n_melds == 0 and shanten_qidui(counts) <= -1:
        return True
    return False


def is_qidui_win(counts):
    return shanten_qidui(counts) <= -1


def win_tiles(counts13, n_melds=0):
    """13 形：摸哪些牌能胡（索引列表）。"""
    out = []
    c = list(counts13)
    for x in range(34):
        c[x] += 1
        if is_win(c, n_melds):
            out.append(x)
        c[x] -= 1
    return out


def is_baotou(counts13, n_melds=0):
    """爆头 = 13 形含财神且摸任意一张皆胡。"""
    if counts13[33] == 0:
        return False
    c = list(counts13)
    for x in range(34):
        c[x] += 1
        ok = is_win(c, n_melds)
        c[x] -= 1
        if not ok:
            return False
    return True


def useful_tiles(counts13, n_melds=0):
    """13 形：摸哪些牌能减少向听（索引列表）。"""
    base = shanten(counts13, n_melds)
    out = []
    c = list(counts13)
    for x in range(34):
        c[x] += 1
        if shanten(c, n_melds) < base:
            out.append(x)
        c[x] -= 1
    return out


# ---------------- 番数 ----------------

def qidui_quads(counts14):
    """七对中的四张组数。v21 白板口径修正：
    4 白板在「其余牌存在落单（白板作为百搭补配成对）」时不再计 1 组四张；
    仅当其余牌全自然对、白板两两自配时才计豪华组。"""
    g = counts14[33]
    q = sum(1 for i in range(33) if counts14[i] == 4)
    if g == 4:
        # 检查真牌是否有落单（count==1，需白板补配成对）
        has_single = any(counts14[i] == 1 for i in range(33))
        if not has_single:
            q += 1  # 无落单 → 白板两两自配，计豪华组
    return q


def calc_fan(counts14, n_melds=0, baotou=False, chain_count=0, piao=0):
    """番数。总番 = 分支 × 2^链 × (4白板×2) × (爆头×2)。
    baotou：摸牌前 13 形是否爆头态（服务端 god.baotou，或 is_baotou 计算）。
    返回 (fan, detail)。不能胡返回 (0, [])。"""
    g = counts14[33]
    qd = n_melds == 0 and is_qidui_win(counts14)
    normal = shanten_normal(counts14, n_melds) <= -1
    if not qd and not normal:
        return 0, []
    detail = []
    if qd:
        q = qidui_quads(counts14)
        base = 2 << q
        detail.append("七对" if q == 0 else f"豪华七对×{q}")
    else:
        base = 1
        detail.append("平胡")
    fan = base
    if chain_count > 0:
        fan <<= chain_count
        if piao == 0:
            detail.append("杠开" if chain_count == 1 else f"连杠×{chain_count}")
        elif piao == chain_count:
            detail.append("财飘" if chain_count == 1 else
                          f"{['', '', '双', '三', '四', '五', '六'][min(chain_count, 6)]}财飘")
        else:
            detail.append(f"杠飘链×{chain_count}")
    if g + piao == 4:
        fan <<= 1
        detail.append("4个白板")
    if baotou:
        fan <<= 1
        detail.append("爆头")
    return fan, detail


def score_table(fan, dealer_win, base=1):
    """返回 (赢家得分, 每个输家付分)。庄家胡：三闲各付 8×fan；闲家胡：庄付 8×fan，两闲各付 1×fan。"""
    if dealer_win:
        pay = 8 * fan * base
        return 3 * pay, {"nondealer": pay}
    return 10 * fan * base, {"dealer": 8 * fan * base, "nondealer": fan * base}


# ---------------- 自测 ----------------
if __name__ == "__main__":
    import time
    H = lambda s: hand_to_counts(s.split())
    tests = [
        ("1w 2w 3w 4w 5w 6w 7w 8w 9w 1b 2b 3b 1t 1t", 0, True),
        ("1w 2w 3w 4w 5w 6w 7w 8w 9w 1b 2b 3b 白 白", 0, True),
        ("1w 1w 2w 2w 3w 3w 4w 4w 5w 5w 6w 6w 7w 7w", 0, True),
        ("1w 白 3w 4w 5w 6w 7w 8w 9w 1b 2b 3b 1t 1t", 0, True),
        ("东 东 东 南 南 南 西 西 西 北 北 北 中 中", 0, True),
        ("1w 2w 3w 4w 5w 6w 7w 8w 9w 1b 2b 3b 1t 2t", 0, False),
        ("4w 5w 6w 7w 8w 9w 2b 3b 4b 5t 5t", 1, True),
        ("1w 1w 2w 2w 3w 3w 4w 4w 5w 5w 6w", 1, False),
        ("西 5b 白 2t 3t 4t 5t 西 7b 6b 3t", 1, False),
    ]
    for s, nm, exp in tests:
        r = is_win(H(s), nm)
        print(("OK " if r == exp else "BAD"), s, nm, r)
    # 向听
    for s, nm in [("1w 2w 3w 4w 5w 6w 7w 8w 9w 1b 2b 3b 1t", 0),
                  ("1w 1w 1w 2w 5w 白 白 白 1b 2b 3b 4b 5b", 0),
                  ("1w 4w 7w 1b 4b 7b 1t 4t 7t 东 南 西 北", 0),
                  ("1w 4w 7w 1b 4b 7b 1t 4t 7t 东 南 西 白", 0),
                  ("1w 1w 2w 2w 3w 3w 4w 4w 5w 5w 6w 6w 9t", 0),
                  ("1w 1w 2w 2w 3w 3w 4w 4w 5w 5w 6w 白 9t", 0),
                  ("1w 2w 3w 4w 5w 6w 7w 8w 9w 1b 2b 3b 白", 0)]:
        c = H(s)
        print(s, "| 向听", shanten(c, nm), "普通", shanten_normal(c, nm), "七对", shanten_qidui(c),
              "| 爆头", is_baotou(c, nm), "| 胡张", [idx_to_tile(x) for x in win_tiles(c, nm)][:8])
    # 番
    for s, bt in [("1w 2w 3w 4w 5w 6w 7w 8w 9w 1b 2b 3b 白 白", True),
                  ("1w 1w 1w 1w 2w 2w 3w 3w 4w 4w 5w 5w 6w 6w", False),
                  ("白 白 白 1w 2w 3w 4w 5w 6w 7w 8w 9w 1t 白", True)]:
        print(s, calc_fan(H(s), 0, bt))
    # 速度
    import random
    rnd = random.Random(1)
    pool = [i for i in range(34) for _ in range(4)]
    hands = []
    for _ in range(3000):
        rnd.shuffle(pool)
        c = [0] * 34
        for x in pool[:13]:
            c[x] += 1
        hands.append(c)
    t0 = time.time()
    for c in hands:
        shanten(c)
    dt = time.time() - t0
    print(f"3000 random hands shanten: {dt*1000:.0f} ms ({dt/3000*1e6:.0f} us/hand)")
    t0 = time.time()
    for c in hands[:300]:
        useful_tiles(c)
    dt = time.time() - t0
    print(f"300 useful_tiles: {dt*1000:.0f} ms ({dt/300*1e3:.1f} ms/hand)")
