#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hz_bot v2 —— 杭州麻将平台对接（hz_brain 决策引擎），事件驱动重构版。

用法：
  python hz_bot.py --token <令牌> [--name 名字] [--server https://10.240.169.190:18080]
  python hz_bot.py --token <全局令牌> --match [--loops N]      # 天梯自动匹配（/api/match）
  python hz_bot.py --token <令牌> --game <game_id>               # 直接打某一场
  可选：--sse（用 /notify SSE 触发拉取，默认长轮询游标）  --chi-early（出牌一到即抢发 chi）
        --rate 12（全局请求令牌桶 /s）  --dump DIR（每场原始收发落盘 jsonl，供验收审计）
  环境变量 HZ_TOKEN / HZ_SERVER 可替代参数。

相对 v1 的核心变化（N1 需求 a–g）：
  a. 不再「每次 seq=0 重拉 + 0.1s 忙轮询」：用游标长轮询 /state?seq=游标（服务端有事件即回，本身就是推送）；
     --sse 时改为收到 /notify 帧后再拉增量。
  b. 本地状态机（GameState）由 seq=0 快照初始化，之后只吃增量事件；仅在 gap / 事件不连续 / 409 时 seq=0 重建。
  c. 他家出牌事件一到，本地立刻算碰/杠/吃并 POST（不等快照）。碰窗口不需要 pass（窗口固定走满，pass 无收益，省请求）。
     吃窗口在碰窗口关闭（约 +1s）后才开放：先算好吃的决定，在「timeout/peng 事件到达」「+1.1s 定时器」「+1.6s 兜底」
     三个时机重试提交（--chi-early 再加「出牌一到就试一次」），成功即止。
  d. 全局令牌桶（默认 12/s，POST 优先：GET 需余量 ≥3，POST ≥1），429 只做 0.15s 级短退避，不再 sleep 1.5s。
  e. 每个响应窗口只提交一次动作（键 = 局号 + 河牌总数 + 出的牌）；每个出牌决策点只提交一次。
  f. 本方吃/碰后 drawn_tile 置空（不能胡，只能打牌），直到下次本方 tile_drawn。
  g. 抓打圈：本人受限 ⇔ catch_play && god_discarder_seat != seat；圈内他家出的牌不开窗口。
"""
import json, ssl, sys, os, time, argparse, threading, socket, collections, random, urllib.request, urllib.error
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hz_brain import Brain, best_discard_value
from hz_core import (GOD, TILE_NAMES, TILE_INDEX, hand_to_counts, counts_to_hand, is_baotou,
                     is_suited, rank_of)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SERVER = os.environ.get("HZ_SERVER", "https://10.240.169.190:18080")
TOKEN = os.environ.get("HZ_TOKEN", "")
NAME = "hz"
KNOWN_GUIDE_VERSION = 34  # 2026-09-14 官网最新（v34 今日榜加 last 键：纯加法，bot 不消费榜单，零改动；
                          # v33 杠后补牌不再自动结算：补牌点 phase=draw 与普通摸牌同构，act_draw 主动判胡/续杠/财飘——
                          # 与 hz_sim 既有建模一致、零改动；v32 杠爆判定修正：服务端结算口径，bot 消费 fan/scores 零改动；v31 局间固定 5s 停顿：settled 态轮询已按「再等一拍」处理，见 run()）
_SSL = ssl._create_unverified_context()
GOD_T = TILE_NAMES[GOD]

OPTS = {
    "sse": False,          # 用 /notify SSE 触发增量拉取
    "chi_early": False,    # 出牌事件一到就抢发一次 chi（服务端若在碰阶段拒绝则 409，随后按正常时机重试）
    "rate": 14,            # 全局限速：滑动 1s 窗口最多请求数（服务端上限 16/s/用户）
    "burst": 14,
    "chi_timers": (1.1, 1.6),  # 出牌事件到达后多少秒尝试提交 chi（碰窗口 1s 走满后才开吃窗口）
    "dump": None,          # 原始收发落盘目录
    "poll_timeout": 40,
    "coalesce": 0.12,      # 收到事件批后再拉下一批前的合并等待（秒），0 关闭
}
STATS = collections.Counter()
_STATS_LOCK = threading.Lock()
_LOG_LOCK = threading.Lock()
# 锦标赛可变规则（指南 §1.6：以 GET /api/tournaments/{id} 的 config 为准，不可写死）。
# 由 apply_config 填充；GameState 初始化时合并进 st.rules → 传给 brain（YouCaiBiKao 影响估值与可胡判定）。
RULES = {}


def apply_config(cfg, src=""):
    """把服务端 config 映射为 bot 规则/时序。返回是否有非默认项。"""
    if not isinstance(cfg, dict) or not cfg:
        return False
    changed = {}
    ycbk = bool(cfg.get("YouCaiBiKao"))
    if RULES.get("YouCaiBiKao") != ycbk:
        RULES["YouCaiBiKao"] = ycbk
    if ycbk:
        changed["YouCaiBiKao"] = True
    peng = cfg.get("PengTimeoutSec")
    if isinstance(peng, (int, float)) and peng > 0 and abs(peng - 1.0) > 1e-6:
        # 吃窗口在碰窗口走满后才开：定时器随碰窗口长度平移
        OPTS["chi_timers"] = (round(peng + 0.1, 2), round(peng + 0.6, 2))
        changed["PengTimeoutSec"] = peng
    disc = cfg.get("DiscardTimeoutSec")
    if isinstance(disc, (int, float)) and disc > 0 and disc != 3:
        changed["DiscardTimeoutSec"] = disc
        if disc < 1.5:
            log("警告：出牌时限", disc, "s 很短，建议 --params meld_lookahead=0")
    base = cfg.get("BaseScore")
    if isinstance(base, (int, float)) and base != 1:
        changed["BaseScore"] = base   # 估值按比例缩放，不影响决策
    if changed:
        log("规则覆盖", src, changed, "chi_timers=", OPTS["chi_timers"])
    else:
        log("规则默认", src, {k: cfg.get(k) for k in ("M", "Rounds", "BaseScore", "YouCaiBiKao", "DiscardTimeoutSec", "PengTimeoutSec", "ChiTimeoutSec")})
    return bool(changed)


def log(*a):
    with _LOG_LOCK:
        print(time.strftime("%H:%M:%S"), f"[{NAME}]", *a, flush=True)


def stat(k, n=1):
    with _STATS_LOCK:
        STATS[k] += n


# ---------------- 全局令牌桶 ----------------
class TokenBucket:
    """全局限速器：滑动 1s 窗口内最多 rate 次请求（与服务端 16/s 的判定方式一致，令牌桶的突发会被判 429）。
    GET 按 FIFO 公平排队（否则 10 场并发时会有场次饿死几秒）；POST 插队，且 GET 给 POST 预留 reserve 个名额。"""

    def __init__(self, rate, burst=None):
        self.rate = int(rate)
        self.reserve = 2
        self.hist = collections.deque()      # 最近 1s 内的发出时刻
        self.cv = threading.Condition()
        self.queue = collections.deque()

    @property
    def burst(self):
        return self.rate

    @burst.setter
    def burst(self, v):
        pass

    @property
    def tokens(self):
        return self.rate - len(self.hist)

    @tokens.setter
    def tokens(self, v):
        pass

    def acquire(self, priority=False):
        limit = self.rate if priority else self.rate - self.reserve
        ticket = None
        with self.cv:
            if not priority:
                ticket = object()
                self.queue.append(ticket)
            waited = False
            while True:
                now = time.monotonic()
                # 窗口取 1.1s：服务端窗口边界含义未知（可能闭区间），留 0.1s 余量避免边界处两批叠加
                while self.hist and self.hist[0] <= now - 1.1:
                    self.hist.popleft()
                head = priority or self.queue[0] is ticket
                if head and len(self.hist) < limit:
                    self.hist.append(now)
                    if ticket is not None:
                        self.queue.popleft()
                    self.cv.notify_all()
                    return
                if not waited:
                    stat("bucket_wait_post" if priority else "bucket_wait_get")
                    waited = True
                wait = (self.hist[0] + 1.1 - now) if (head and self.hist) else 0.05
                self.cv.wait(timeout=max(0.002, min(wait, 0.25)))


BUCKET = TokenBucket(OPTS["rate"], OPTS["burst"])


class ApiError(Exception):
    def __init__(self, status, body):
        super().__init__(f"{status}: {body[:200]}")
        self.status, self.body = status, body


def _dump(gid, kind, obj):
    d = OPTS.get("dump")
    if not d:
        return
    try:
        os.makedirs(d, exist_ok=True)
        with _LOG_LOCK:
            with open(os.path.join(d, f"{gid}.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": round(time.time(), 3), "k": kind, "v": obj}, ensure_ascii=False) + "\n")
    except Exception:
        pass


_API_TIMING = {}   # thread id → {"wait": 令牌桶等待秒, "http": 网络往返秒, "attempts": 次数}（供 post() 落盘慢请求分解）


def api(method, path, body=None, timeout=None, priority=None):
    """带令牌桶的 HTTP 调用。429 短退避重试；5xx/网络错误有限重试。"""
    if priority is None:
        priority = method == "POST"
    if timeout is None:
        timeout = OPTS["poll_timeout"] if method == "GET" else 10
    data = json.dumps(body).encode() if body is not None else None
    last = None
    tm = {"wait": 0.0, "http": 0.0, "attempts": 0}
    _API_TIMING[threading.get_ident()] = tm
    for attempt in range(6):
        tm["attempts"] = attempt + 1
        t_a = time.monotonic()
        BUCKET.acquire(priority)
        t_b = time.monotonic()
        tm["wait"] += t_b - t_a
        req = urllib.request.Request(SERVER.rstrip("/") + path, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", "Bearer " + TOKEN)
        stat("req_" + method)
        try:
            try:
                with urllib.request.urlopen(req, timeout=timeout, context=_SSL) as r:
                    raw = r.read()
            finally:
                tm["http"] += time.monotonic() - t_b
            try:
                return json.loads(raw.decode("utf-8", "replace"))
            except ValueError:
                # 200 但正文不是 JSON（网关错误页等）：按瞬态网络错误重试，绝不能让对局线程死掉
                stat("bad_json")
                last = ApiError(0, f"bad json: {raw[:80]!r}")
                if attempt < 5:
                    time.sleep(0.3 * (attempt + 1))
                    continue
                raise last
        except urllib.error.HTTPError as e:
            if e.code == 429:
                stat("http_429")
                _dump("_ratelimit", "429", {"path": path, "method": method, "attempt": attempt})
                # 测试赛 21 次 429 全部是网络停摆（3.5s 无任何收发）恢复后 10 场线程的 GET 同时落到服务端；
                # 退避加随机抖动，避免各线程再次同拍重试。POST（动作）退避短、GET 长。
                back = (0.12 if priority else 0.25) * (attempt + 1) + random.uniform(0.0, 0.2)
                time.sleep(back)
                last = ApiError(429, "RATE_LIMITED")
                continue
            if e.code in (502, 503, 504) and attempt < 5:
                stat("http_5xx")
                time.sleep(0.5)
                last = ApiError(e.code, "")
                continue
            raise ApiError(e.code, e.read().decode(errors="replace"))
        except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as e:
            stat("net_err")
            last = ApiError(0, f"network error: {e!r}")
            if attempt < 5:
                time.sleep(0.3 * (attempt + 1))
                continue
    raise last or ApiError(0, "unknown")


# ---------------- 本地状态机 ----------------
class GameState:
    """由 seq=0 快照初始化，之后按事件增量维护；能随时生成 hz_brain 所需快照。"""

    def __init__(self):
        self.me = -1
        self.dealer = 0
        self.round_no = 0
        self.scores = [0, 0, 0, 0]
        self.hand = [0] * 34
        self.melds = [[], [], [], []]
        self.discards = [[], [], [], []]
        self.hand_counts = [13] * 4
        self.wall = 83
        self.last_discard = None
        self.last_discard_seat = None
        self.drawn = None
        self.chain = 0
        self.piao = 0                # 链内已飘出的白板数（4 白板 = 手留 + 链内飘出）
        self.restricted = False      # 本人处于抓打圈受限（只能打刚摸的、不可吃碰杠）
        self.god_discarder = -1
        self.circle_left = 0
        self.phase = "idle"
        self.turn = -1
        self.responding = []
        self.rules = {"wall_reserve": 20, "gang_min_wall": 21, "max_chi": 2}
        self.rules.update(RULES)     # 锦标赛 config（YouCaiBiKao 等）
        self.draw_id = 0             # 每次进入「本人出牌决策点」+1
        self.last_event_ts = 0
        self.my_peng_closed = False  # 当前响应窗口：我的碰超时事件已出现（碰已不可能）
        self.last_discard_ts = 0.0   # 最近出牌事件的服务端 ts（秒）

    # ---- 快照 ----
    def load_snapshot(self, s):
        self.me = s.get("seat", -1)
        self.dealer = s.get("dealer", 0)
        self.round_no = s.get("round_no", 0)
        self.scores = list(s.get("scores") or [0, 0, 0, 0])
        self.hand = hand_to_counts(s.get("my_hand") or [])
        ms = s.get("melds") or [[], [], [], []]
        self.melds = []
        for seat_melds in ms:
            out = []
            for m in seat_melds or []:
                kind = m.get("kind", "peng")
                tiles = list(m.get("tiles") or [])
                if kind == "peng" and len(tiles) == 4:
                    kind = "gang_bu"
                out.append({"kind": kind, "tiles": tiles})
            self.melds.append(out)
        while len(self.melds) < 4:
            self.melds.append([])
        self.discards = [list(d or []) for d in (s.get("discards") or [[], [], [], []])]
        while len(self.discards) < 4:
            self.discards.append([])
        self.hand_counts = list(s.get("hand_counts") or [13] * 4)
        self.wall = s.get("wall_remaining", 0)
        self.last_discard = s.get("last_discard") or None
        self.phase = s.get("phase", "idle")
        self.turn = s.get("turn", -1)
        self.responding = list(s.get("responding_seats") or [])
        self.drawn = s.get("drawn_tile") or None
        if s.get("rules"):
            self.rules.update(s["rules"])
        god = s.get("god") or {}
        self.chain = int(god.get("chain_count") or 0)
        pc = god.get("piao_count")
        self.piao = int(pc) if pc is not None else min(self.chain, self.piao)
        cp = bool(god.get("catch_play"))
        gds = god.get("god_discarder_seat")
        if gds is None:
            gds = -1
        self.god_discarder = gds if cp else -1
        self.restricted = cp and gds != self.me
        self.circle_left = 3 if cp else 0   # 近似（圈内剩余受限出牌数未知，按事件递减到 0 即清）
        # last_discard_seat 推断：响应阶段 turn 通常为出牌者；否则找河牌末尾匹配
        src = s.get("last_discard_seat")
        if src is None and self.last_discard:
            if self.phase in ("response_peng", "response_chi") and 0 <= self.turn < 4 \
                    and self.discards[self.turn] and self.discards[self.turn][-1] == self.last_discard:
                src = self.turn
            else:
                for seat, d in enumerate(self.discards):
                    if d and d[-1] == self.last_discard:
                        src = seat
        self.last_discard_seat = src
        self.last_discard_ts = 0.0
        self.my_peng_closed = (self.phase == "response_chi")
        if self.phase == "draw" and self.turn == self.me:
            self.draw_id += 1

    # ---- 事件 ----
    def apply(self, e):
        """返回触发器：my_draw / my_turn / window / peng_closed / round_ended / game_ended / None"""
        t = e.get("type")
        s = e.get("seat")
        tile = e.get("tile") or None
        d = e.get("data") or {}
        self.last_event_ts = e.get("ts", 0)
        me = self.me
        if t == "tile_drawn":
            self.wall -= 1
            if 0 <= s < 4:
                self.hand_counts[s] += 1
            self.phase, self.turn, self.responding = "draw", s, []
            if s == me and tile:
                self.hand[TILE_INDEX[tile]] += 1
                self.drawn = tile
                self.draw_id += 1
                return "my_draw"
            return None
        if t == "tile_discarded":
            if tile is None:
                return None
            ti = TILE_INDEX[tile]
            if 0 <= s < 4:
                self.hand_counts[s] -= 1
                self.discards[s].append(tile)
            if s == me:
                if self.hand[ti] > 0:
                    self.hand[ti] -= 1
                self.drawn = None
                self.restricted = False
                if ti == GOD and is_baotou(self.hand, len(self.melds[me])):
                    self.chain += 1; self.piao += 1
                else:
                    self.chain = 0; self.piao = 0
            self.last_discard, self.last_discard_seat = tile, s
            self.last_discard_ts = float(e.get("ts") or 0)
            self.phase, self.turn = "response", s
            self.my_peng_closed = False
            if ti == GOD:
                # 新抓打圈：其余三家受限
                self.god_discarder = s
                self.circle_left = 3
                self.restricted = (s != me)
                return None
            if d.get("catch_play"):
                self.circle_left -= 1
                if self.circle_left <= 0:
                    self.circle_left = 0
                    self.god_discarder = -1
                    self.restricted = False   # 圈已走满：即便本人的受限出牌事件丢失/被快照吞掉也不再残留受限态
                return None
            if s != me and not self.restricted:
                return "window"
            return None
        if t == "timeout":
            if d.get("kind") == "response" and d.get("window") == "peng":
                if s == me:
                    self.my_peng_closed = True   # 本窗口的碰机会已过（事件批里已包含我的碰超时）
                return "peng_closed"
            return None
        if t == "pass":
            return None
        if t == "peng":
            if tile and 0 <= s < 4:
                self.hand_counts[s] -= 2
                self.melds[s].append({"kind": "peng", "tiles": [tile] * 3})
                if s == me:
                    self.hand[TILE_INDEX[tile]] = max(0, self.hand[TILE_INDEX[tile]] - 2)
                    self.chain = 0; self.piao = 0
            self.phase, self.turn, self.drawn = "draw", s, None
            if s == me:
                self.draw_id += 1
                return "my_turn"
            return None
        if t == "chi":
            tiles = list(d.get("tiles") or [])
            if tile and tiles and 0 <= s < 4:
                others = list(tiles)
                if tile in others:
                    others.remove(tile)
                self.hand_counts[s] -= 2
                self.melds[s].append({"kind": "chi", "tiles": tiles})
                if s == me:
                    for x in others:
                        self.hand[TILE_INDEX[x]] = max(0, self.hand[TILE_INDEX[x]] - 1)
                    self.chain = 0; self.piao = 0
            self.phase, self.turn, self.drawn = "draw", s, None
            if s == me:
                self.draw_id += 1
                return "my_turn"
            return None
        if t == "gang":
            k = d.get("kind")
            if tile and 0 <= s < 4:
                ti = TILE_INDEX[tile]
                if k == "an":
                    self.hand_counts[s] -= 4
                    self.melds[s].append({"kind": "gang_an", "tiles": [tile] * 4})
                    if s == me:
                        self.hand[ti] = max(0, self.hand[ti] - 4)
                elif k == "bu":
                    self.hand_counts[s] -= 1
                    for m in self.melds[s]:
                        if m["kind"] == "peng" and m["tiles"] and m["tiles"][0] == tile:
                            m["kind"] = "gang_bu"; m["tiles"] = [tile] * 4
                            break
                    if s == me:
                        self.hand[ti] = max(0, self.hand[ti] - 1)
                else:
                    self.hand_counts[s] -= 3
                    self.melds[s].append({"kind": "gang_ming", "tiles": [tile] * 4})
                    if s == me:
                        self.hand[ti] = max(0, self.hand[ti] - 3)
                if s == me:
                    self.chain += 1
            self.phase, self.turn, self.drawn = "draw", s, None   # 随后有补牌 tile_drawn
            return None
        if t == "round_ended":
            self.phase = "settled"
            return "round_ended"
        if t == "game_ended":
            self.phase = "finished"
            return "game_ended"
        return None

    # ---- 供 brain 的快照 ----
    def snap(self, phase, turn, responding):
        me = self.me
        h = self.hand
        nm = len(self.melds[me])
        if phase == "draw" and self.drawn:
            h13 = list(h); h13[TILE_INDEX[self.drawn]] -= 1
            bt = is_baotou(h13, nm) if sum(h13) % 3 == 1 else False
        else:
            bt = is_baotou(h, nm) if sum(h) % 3 == 1 else False
        return {
            "seat": me, "dealer": self.dealer, "turn": turn, "phase": phase,
            "my_hand": counts_to_hand(h),
            "drawn_tile": self.drawn if phase == "draw" else None,
            "melds": [[dict(m) for m in ms] for ms in self.melds],
            "discards": [list(d) for d in self.discards],
            "hand_counts": list(self.hand_counts),
            "wall_remaining": self.wall,
            "last_discard": self.last_discard, "last_discard_seat": self.last_discard_seat,
            "responding_seats": list(responding),
            "god": {"baotou": bt, "chain_count": self.chain, "piao_count": self.piao, "catch_play": self.restricted,
                    "god_discarder_seat": self.god_discarder},
            "scores": list(self.scores), "round_no": self.round_no,
            "rules": dict(self.rules),
        }

    def window_key(self):
        return (self.round_no, sum(len(d) for d in self.discards), self.last_discard)

    def can_peng(self, tile):
        return tile != GOD_T and self.hand[TILE_INDEX[tile]] >= 2 and not self.restricted

    def chi_count(self):
        return sum(1 for m in self.melds[self.me] if m["kind"] == "chi")

    def can_chi(self, tile):
        if tile == GOD_T or self.restricted:
            return False
        ti = TILE_INDEX[tile]
        if not is_suited(ti) or self.chi_count() >= self.rules.get("max_chi", 2):
            return False
        h = self.hand; r = rank_of(ti)
        for a, b in ((-2, -1), (-1, 1), (1, 2)):
            if 1 <= r + a <= 9 and 1 <= r + b <= 9 and h[ti + a] > 0 and h[ti + b] > 0:
                return True
        return False


# ---------------- 单场对局 ----------------
class GamePlayer:
    def __init__(self, gid, brain):
        self.gid = gid
        self.brain = brain
        self.st = GameState()
        self.lock = threading.RLock()
        self.cursor = 0
        self.need_full = True
        self.acted_draw_id = -1
        self.acted_window = None
        self.chi_plan = None       # {"key", "act", "attempts", "done", "t0"}
        self.lat_resp = []         # 他家出牌事件收到 → 本方动作提交完成 的耗时
        self.lat_draw = []
        self.last_discard_rx = 0.0 # 最近一次 tile_discarded 事件的本地接收时刻
        self.last_rx = 0.0         # 最近一次 /state 返回（事件批或快照）的本地接收时刻（慢提交分解基准）
        self.think_t = None        # 最近一次 brain 决策耗时（秒）
        self.notify_ev = threading.Event()
        self.sse_alive = False
        self.finished = False
        self.gap_streak = 0        # 连续收到 seq 不前进的 gap 快照次数（新局发牌后等庄家首打；局间停顿服务端会挂起、不产生 gap）
        self.tag = gid[-14:] if len(gid) > 14 else gid

    # ---- 提交 ----
    def _timing(self, t0):
        """本次 POST 的耗时分解：since_rx = 收到触发事件批 → 现在；wait = 令牌桶排队；http = 网络往返；think = 决策。"""
        now = time.monotonic()
        tm = _API_TIMING.get(threading.get_ident()) or {}
        since_rx = now - self.last_rx if self.last_rx else 0.0
        d = {"since_rx": round(since_rx, 3), "post": round(now - t0, 3), "wait": round(tm.get("wait", 0.0), 3),
             "http": round(tm.get("http", 0.0), 3), "attempts": tm.get("attempts", 1)}
        if self.think_t is not None:
            d["think"] = round(self.think_t, 3)
        return d

    def post(self, act, why=""):
        a = dict(act)
        a.setdefault("tile", "")
        if a["action"] == "chi" and "tiles" in a:
            a["tiles"] = list(a["tiles"])
        t0 = time.monotonic()
        try:
            r = api("POST", f"/api/games/{self.gid}/action", a)
            tm = self._timing(t0)
            _dump(self.gid, "post_ok", {"act": a, "why": why, "resp": r, "ms": round(tm["post"] * 1000), "tm": tm})
            stat("post_ok")
            if tm["since_rx"] > 1.5:
                # 收到事件 → 提交完成 超过 1.5s：距 3s 出牌窗口已很近，单独告警并分解（谁慢：决策/排队/网络）
                stat("post_slow")
                log(self.tag, "⚠ 慢提交", a["action"], a.get("tile"), why, tm)
            if a["action"] in ("hu", "gang", "peng", "chi"):
                log(self.tag, "→", a["action"], a.get("tile"), a.get("tiles", ""), why, self.brain.last_info or "")
            elif self.brain.last_info.get("piao"):
                log(self.tag, "→ 财飘", self.brain.last_info)
            return True
        except ApiError as e:
            tm = self._timing(t0)
            _dump(self.gid, "post_err", {"act": a, "why": why, "status": e.status, "body": e.body[:200], "tm": tm})
            if e.status == 409:
                stat("http_409")
                stat("http_409_" + a["action"])
                if tm["since_rx"] > 1.5:
                    stat("post_slow")
                    log(self.tag, "⚠ 慢提交→409", a["action"], a.get("tile"), why, tm, e.body[:60])
                return False
            if e.status in (401, 403, 404):
                log(self.tag, "action 致命错误", e.status, e.body[:100])
                raise
            log(self.tag, "action 错误", e.status, e.body[:100])
            return False

    # ---- 出牌决策点 ----
    def do_draw(self, why, allow_rebuild=True):
        st = self.st
        if self.acted_draw_id == st.draw_id:
            return
        self.acted_draw_id = st.draw_id
        t0 = time.monotonic()
        snap = st.snap("draw", st.me, [])
        act = self.brain.act_draw(snap)
        self.think_t = time.monotonic() - t0
        ok = self.post(act, why)
        self.lat_draw.append(time.monotonic() - t0)
        if ok:
            return
        # 409：本地状态可能漂移 → seq=0 重建后按快照重决策一次；再不行则兜底出牌
        log(self.tag, "409 on", act, "hand=", snap["my_hand"], "drawn=", snap["drawn_tile"], "why=", why)
        if allow_rebuild:
            self.rebuild("409_draw", allow_rebuild=False)
            return
        if act.get("action") != "discard":
            self.fallback_discard(snap)

    def fallback_discard(self, snap):
        hand = snap.get("my_hand") or []
        if not hand:
            return
        tile = None
        if snap["god"].get("catch_play") and snap.get("drawn_tile"):
            tile = snap["drawn_tile"]
        else:
            try:
                c = hand_to_counts(hand)
                ctx = self.brain.make_ctx(snap)
                d, _ = best_discard_value(c, len(snap["melds"][snap["seat"]]), ctx)
                if d is not None:
                    tile = TILE_NAMES[d]
            except Exception:
                pass
            if tile is None:
                cands = [t for t in hand if t != GOD_T] or hand
                tile = cands[-1]
        self.post({"action": "discard", "tile": tile}, "fallback")

    # ---- 响应窗口 ----
    def on_window(self, src="event"):
        st = self.st
        key = st.window_key()
        if self.acted_window == key:
            return
        self.acted_window = key
        self.chi_plan = None
        tile = st.last_discard
        if not tile or st.last_discard_seat is None or st.last_discard_seat == st.me:
            return
        t0 = time.monotonic()
        peng_open = not st.my_peng_closed
        if st.can_peng(tile) and peng_open:
            act = self.brain.act_response(st.snap("response_peng", st.last_discard_seat, [st.me]))
            self.think_t = time.monotonic() - t0
            if act.get("action") in ("peng", "gang"):
                ok = self.post(act, f"win/{src}")
                if not ok and act["action"] == "gang":
                    ok = self.post({"action": "peng", "tile": tile}, "gang→peng")
                self.lat_resp.append(time.monotonic() - t0)
                if ok:
                    return
        if (st.last_discard_seat + 1) % 4 == st.me and st.can_chi(tile):
            t1 = time.monotonic()
            act = self.brain.act_response(st.snap("response_chi", st.last_discard_seat, [st.me]))
            self.think_t = time.monotonic() - t1
            if act.get("action") == "chi":
                self.chi_plan = {"key": key, "act": act, "attempts": 0, "done": False, "t0": t0}
                if not peng_open:
                    # 事件批里已带我的碰超时：吃窗口此刻已开，立即提交
                    self.try_chi("pengclosed")
                elif OPTS["chi_early"] or src == "snap_chi":
                    self.try_chi("early" if src != "snap_chi" else "snap")
                if self.chi_plan and not self.chi_plan["done"]:
                    # 定时器以本地收到事件为基准；若事件明显晚到（服务端 ts 只有秒级精度，真实时刻 ∈ [ts, ts+1)，
                    # 故只用保守下界 age_lo = now-ts-1），则相应提前，保证不会早于真实出牌后 dt 秒（早试会 409 白费额度）。
                    now = time.time()
                    age_lo = max(0.0, now - st.last_discard_ts - 1.0) if st.last_discard_ts else 0.0
                    for dt in OPTS["chi_timers"]:
                        d = round(max(0.05, dt - age_lo), 2)
                        th = threading.Timer(d, self._chi_timer, args=(key, d))
                        th.daemon = True
                        th.start()

    def _chi_timer(self, key, dt):
        try:
            with self.lock:
                self.try_chi(f"timer{dt}", key)
        except Exception as e:
            log(self.tag, "chi timer 异常", repr(e))

    def try_chi(self, why, key=None):
        p = self.chi_plan
        if not p or p["done"] or p["attempts"] >= 4:
            return
        if key is not None and p["key"] != key:
            return
        if self.st.window_key() != p["key"] or self.st.phase != "response":
            self.chi_plan = None
            return
        p["attempts"] += 1
        stat("chi_try_" + why.rstrip("0123456789."))
        if self.post(p["act"], f"chi/{why}#{p['attempts']}"):
            p["done"] = True
            self.lat_resp.append(time.monotonic() - p["t0"])

    # ---- 快照 / 重建 ----
    def rebuild(self, why, allow_rebuild=True):
        res = api("GET", f"/api/games/{self.gid}/state?seq=0", priority=True)
        _dump(self.gid, "full", res)
        stat("rebuild_" + why.split("_")[0])
        return self.handle_response(res, allow_rebuild=allow_rebuild)

    def handle_snapshot(self, snap, allow_rebuild=True):
        st = self.st
        st.load_snapshot(snap)
        self.chi_plan = None
        me = st.me
        if me is None or me < 0:
            return
        phase = st.phase
        if phase == "draw" and st.turn == me:
            self.do_draw("snap", allow_rebuild=allow_rebuild)
        elif phase in ("response_peng", "response_chi") and me in st.responding:
            key = st.window_key()
            if phase == "response_peng":
                self.on_window("snap")
            else:
                # 已在吃阶段：直接（重）算吃的决定并提交
                self.acted_window = None
                self.on_window("snap_chi")

    def handle_response(self, res, allow_rebuild=True):
        """处理一次 /state 返回。返回 False 表示对局结束。"""
        self.last_rx = time.monotonic()
        if res.get("finished"):
            snap = res.get("snapshot") or {}
            log(self.tag, "本场结束 scores=", snap.get("scores"))
            self.finished = True
            return False
        if res.get("pending"):
            return True
        snap = res.get("snapshot")
        if snap is not None:
            new_seq = res.get("seq", self.cursor) or 0
            if res.get("gap"):
                stat("gap_snapshot")
                # 新局发牌后、首个动作前，seq 不前进而每次轮询都立即回 gap 快照（测试赛：一局边界最多 7 次/3s）。
                # 记录连续重复次数，run() 据此在下次拉取前小睡，省下无意义请求；有事件后 seq 前进即归零。
                self.gap_streak = self.gap_streak + 1 if new_seq == self.cursor else 0
            else:
                self.gap_streak = 0
            self.cursor = new_seq
            self.need_full = False
            self.handle_snapshot(snap, allow_rebuild=allow_rebuild)
            return True
        if res.get("gap"):
            self.need_full = True
            return True
        evs = res.get("events") or []
        triggers = []
        for e in evs:
            sq = e.get("seq")
            if sq is None or sq != self.cursor + 1:
                stat("seq_discontinuity")
                self.need_full = True
                break
            self.cursor = sq
            tr = self.st.apply(e)
            if e.get("type") == "tile_discarded":
                self.last_discard_rx = time.monotonic()
            if tr:
                triggers.append(tr)
        if self.need_full:
            return True
        self.act_on(triggers)
        return True

    def act_on(self, triggers):
        st = self.st
        for tr in triggers:
            if tr == "round_ended":
                self.chi_plan = None
                self.acted_window = None
                self.need_full = True     # 新局发牌无事件：下次用 seq=0 重建（服务端也可能返回 gap 快照）
        # 只按批次末尾的状态决策
        if "game_ended" in triggers:
            return
        if "round_ended" in triggers:
            return
        if st.phase == "draw" and st.turn == st.me and self.acted_draw_id != st.draw_id:
            why = "draw" if "my_draw" in triggers else ("meld" if "my_turn" in triggers else "state")
            self.do_draw(why)
            return
        if "window" in triggers and st.phase == "response":
            self.on_window("event")
        elif "peng_closed" in triggers and self.chi_plan and not self.chi_plan["done"]:
            self.try_chi("pengclosed")

    # ---- SSE ----
    def sse_thread(self):
        req = urllib.request.Request(SERVER.rstrip("/") + f"/api/games/{self.gid}/notify")
        req.add_header("Authorization", "Bearer " + TOKEN)
        req.add_header("Accept", "text/event-stream")
        try:
            with urllib.request.urlopen(req, timeout=75, context=_SSL) as r:
                self.sse_alive = True
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    try:
                        fr = json.loads(line)
                    except Exception:
                        continue
                    _dump(self.gid, "sse", fr)
                    sq = fr.get("seq")
                    if isinstance(sq, int) and sq > self.cursor:
                        self.notify_ev.set()
                    if fr.get("closed"):
                        self.notify_ev.set()
                        break
                    if self.finished:
                        break
        except Exception as e:
            log(self.tag, "SSE 断开，回退长轮询", repr(e)[:120])
        finally:
            self.sse_alive = False
            self.notify_ev.set()

    # ---- 主循环 ----
    def run(self):
        st = self.st
        notfound = 0
        if OPTS["sse"]:
            th = threading.Thread(target=self.sse_thread, daemon=True)
            th.start()
            time.sleep(0.2)
        while not self.finished:
            try:
                if self.need_full:
                    res = api("GET", f"/api/games/{self.gid}/state?seq=0", priority=True)
                    _dump(self.gid, "full", res)
                else:
                    if OPTS["sse"] and self.sse_alive:
                        # 等通知；超时兜底也拉一次（服务端会 pending 30s 后返回）。
                        # 开局/新局发牌无事件（seq 不变 → 无 SSE 帧），庄家 3s 内就要出牌：发牌态只等 0.5s。
                        w_to = 0.5 if st.phase in ("deal", "idle", "settled") else 35
                        self.notify_ev.wait(timeout=w_to)
                        time.sleep(0.05)   # 合并紧邻帧
                        self.notify_ev.clear()
                        if self.finished:
                            break
                    res = api("GET", f"/api/games/{self.gid}/state?seq={self.cursor}")
                    if OPTS.get("dump"):
                        # 落盘本次 GET 的令牌桶等待/往返（秒）：用于把「事件送达时延」拆成客户端桶等待 vs 服务端挂起
                        tmg = _API_TIMING.get(threading.get_ident()) or {}
                        res["_tm"] = {"wait": round(tmg.get("wait", 0.0), 3), "http": round(tmg.get("http", 0.0), 3)}
                    _dump(self.gid, "inc", res)
                    if res.get("gap") and res.get("snapshot") is not None and OPTS["coalesce"] > 0:
                        # 局间/发牌后 seq 不前进：每次轮询都立即回同一份 gap 快照（测试赛一局边界最多 7 次/3s，
                        # 10 场并发时白占 GET 配额）。处理后若仍不是我出牌，小睡再拉；有事件后 seq 前进即恢复。
                        with self.lock:
                            try:
                                if not self.handle_response(res):
                                    break
                            except ApiError as e:
                                if e.status in (401, 403, 404):
                                    raise
                                log(self.tag, "处理异常", e.status, e.body[:80])
                                self.need_full = True
                            # 第 2 次起同一份 gap 快照才小睡（首份之后立即再拉：庄家秒打时不耽误碰窗口）；
                            # 0.25s + 往返 ≈ 3/s，与 10 场并发下令牌桶本就限制的节奏相当，空闲期则省一半请求
                            w = 0.25 if (self.gap_streak >= 1 and not (st.phase == "draw" and st.turn == st.me)) else 0.0
                        if w > 0:
                            stat("gap_wait")
                            time.sleep(w)
                        continue
                    if res.get("events") and OPTS["coalesce"] > 0:
                        # 服务端常在同一瞬间连发多条（pass×3/timeout/tile_drawn）：先处理这批，再按局面等一小会儿再拉，
                        # 让下一批合并、省请求配额（10 场并发时配额是瓶颈）。等待长短由 idle_wait 按「下一个与我有关的
                        # 事件最早何时到」决定，不影响碰/吃/出牌时限。
                        with self.lock:
                            try:
                                if not self.handle_response(res):
                                    break
                            except ApiError as e:
                                if e.status in (401, 403, 404):
                                    raise
                                log(self.tag, "处理异常", e.status, e.body[:80])
                                self.need_full = True
                            # SSE 模式下有帧才拉，不需要主动等待合并（等了反而拖慢摸牌后出牌）
                            w = 0.0 if (OPTS["sse"] and self.sse_alive) else self.idle_wait()
                        if w > 0:
                            time.sleep(w)
                        continue
            except ApiError as e:
                if e.status in (403, 404):
                    notfound += 1
                    if notfound > 10:
                        log(self.tag, "不可访问，结束")
                        return
                    time.sleep(1.0)
                    continue
                log(self.tag, "state 异常", e.status, e.body[:80], "重试")
                time.sleep(0.5)
                continue
            notfound = 0
            with self.lock:
                try:
                    if not self.handle_response(res):
                        break
                except ApiError as e:
                    if e.status in (401, 403, 404):
                        raise
                    log(self.tag, "处理异常", e.status, e.body[:80])
                    self.need_full = True
        self.finished = True
        self.notify_ev.set()
        self.report()

    def idle_wait(self):
        """事件批处理完后、再次拉取前的合并等待。原则：不能耽误碰(1s)/吃/出牌(3s)时限。"""
        st = self.st
        base = OPTS["coalesce"]
        if st.phase == "draw":
            return 0.05 if st.turn == st.me else base     # 等他家出牌：随时可能到
        if st.phase == "response":
            if self.chi_plan and not self.chi_plan["done"]:
                return min(base, 0.2)
            elapsed = time.monotonic() - self.last_discard_rx
            if st.last_discard_ts:
                # 事件批晚到（网络停摆 1.5～3.6s 后一次性送达，测试赛 6 次）：按服务端 ts 的保守下界修正，
                # 否则会把陈旧的出牌当成刚发生而多等 0.5～1.2s；下一批（可能含我的摸牌）应立即去拉
                elapsed = max(elapsed, time.time() - st.last_discard_ts - 1.0)
            nxt = (st.last_discard_seat + 1) % 4 if st.last_discard_seat is not None else -1
            if nxt == st.me:
                # 我的摸牌在出牌后约 +2s（若无人碰）；提前一点拉
                return max(base, min(1.2, 1.8 - elapsed))
            # 下一张出牌最早在 +2s（下家摸牌后）；有人碰则更早，取折中上限 0.5s
            return max(base, min(0.5, 1.6 - elapsed))
        return base

    def report(self):
        def pct(xs, q):
            if not xs:
                return 0.0
            ys = sorted(xs); return ys[min(len(ys) - 1, int(q * len(ys)))] * 1000
        log(self.tag, f"延迟 resp n={len(self.lat_resp)} p50={pct(self.lat_resp, .5):.0f}ms max={pct(self.lat_resp, 1):.0f}ms "
                      f"draw n={len(self.lat_draw)} p50={pct(self.lat_draw, .5):.0f}ms max={pct(self.lat_draw, 1):.0f}ms "
                      f"（慢提交 >1.5s 见 ⚠ 行 / stats post_slow）")


def play_game(gid, brain):
    GamePlayer(gid, brain).run()


# ---------------- 赛事生命周期 ----------------

def check_guide_version():
    try:
        req = urllib.request.Request(SERVER.rstrip("/") + "/portal/api/guide/version")
        with urllib.request.urlopen(req, timeout=10, context=_SSL) as r:
            meta = json.load(r)
        if meta.get("version", 0) > KNOWN_GUIDE_VERSION:
            br = [c["summary"] for c in meta.get("changes", []) if c.get("type") == "breaking" and c.get("version", 0) > KNOWN_GUIDE_VERSION]
            log(f"服务器指南 v{meta['version']} > 已知 v{KNOWN_GUIDE_VERSION}；BREAKING：", br)
    except Exception as e:
        log("指南版本检查失败（忽略）", repr(e))


def run_games(game_ids, make_brain):
    threads = []
    for gid in game_ids:
        th = threading.Thread(target=_safe_play, args=(gid, make_brain()), daemon=True)
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    log("批次结束 stats=", dict(STATS))


_DONE_GAMES = set()   # 正常打完（或确认不可访问）的场次：赛事主循环不再接管，避免服务端 active_games 短暂滞后引起空接管


def _safe_play(gid, brain):
    try:
        play_game(gid, brain)
        _DONE_GAMES.add(gid)
    except Exception as e:
        stat("game_thread_crash")
        log(gid[-14:], "对局线程异常", repr(e))


def tournament_loop(make_brain):
    me = None
    for k in range(10 ** 6):   # 开赛前服务端可能正在 502：/api/me 拿不到就一直重试，绝不退出
        try:
            me = api("GET", "/api/me")
            break
        except ApiError as e:
            if e.status == 401:
                raise
            if k in (0, 5, 20) or k % 100 == 0:
                log("/api/me 异常", e.status, e.body[:80], f"第 {k + 1} 次，继续重试")
            time.sleep(min(10.0, 1.0 + 0.5 * k))
    tid = me.get("tournament_id") or ""
    scoped = bool(tid)
    if not tid:
        log("令牌未绑定锦标赛（全局令牌请用 --match）")
        return
    log("参赛", tid)
    cfg_seen = False
    err_streak = 0
    said = set()   # 只打一次的日志键
    threads = {}   # game_id → 对局线程（监督式接管，见 running 分支）
    while True:
        # 赛事主循环绝不能因服务端 5xx 风暴 / 网络抖动退出（内网台账：排位期 502 风暴 20% 房位，持续数分钟）：
        # 除 401（令牌失效）外，一律记日志、退避后继续轮询；对局线程内部已各自容错。
        try:
            t = api("GET", f"/api/tournaments/{tid}")
        except ApiError as e:
            if e.status == 401:
                raise
            err_streak += 1
            stat("tour_poll_err")
            if err_streak in (1, 5, 20) or err_streak % 100 == 0:
                log("赛事轮询异常", e.status, e.body[:80], f"连续 {err_streak} 次，继续等")
            time.sleep(min(10.0, 1.0 + 0.5 * err_streak))
            continue
        if err_streak:
            log("赛事轮询恢复（此前连续异常", err_streak, "次）")
            err_streak = 0
        status = t.get("status")
        if not cfg_seen and t.get("config"):
            apply_config(t["config"], "tournament")
            cfg_seen = True
        stage = t.get("stage") or {}
        stname = stage.get("name", "")
        skey = (status, stage.get("no"))
        if skey not in said:
            said.add(skey)
            log("赛事状态", status, stname or "", "stage_status=", t.get("stage_status"), "crashed=", t.get("stage_crashed"),
                "qualified=", t.get("qualified"), t.get("qualify_role") or "", "my_games=", len(t.get("my_games") or []))
        if status in ("finished", "closed", "void"):
            log("终态:", status, "voided=", t.get("voided"), "ranking=", (t.get("ranking") or [])[:8])
            return
        if status == "registering":
            # 幂等：报名 + 到位；被拒（已报名/已到位/TOURNAMENT_STARTED）都不致命，下一轮按最新 status 走
            for ep in ("register", "ready"):
                try:
                    api("POST", f"/api/tournaments/{tid}/{ep}")
                    if ("reg", ep) not in said:
                        said.add(("reg", ep)); log("报名期", ep, "OK")
                except ApiError as e:
                    if e.status == 401:
                        raise
                    if ("regfail", ep, e.status) not in said:
                        said.add(("regfail", ep, e.status)); log("报名期", ep, "被拒", e.status, e.body[:80])
            time.sleep(2)
        elif status == "stage_open":
            # 阶段确认不跨阶段继承：每轮 stage_open 都要 ready；幂等，每 3s 重发一次直到 status 变化，
            # 服务端按「已确认且开赛时刻 90s 内在线」取人，所以确认期内持续轮询本身就是必要的。
            # qualified=False 也**不退出**：指南「阶段 1 崩溃重赛的确认期例外」等边角情况，多发一次 ready 无害（409 NOT_QUALIFIED 而已）。
            try:
                api("POST", f"/api/tournaments/{tid}/ready")
                if ("confirm", stage.get("no")) not in said:
                    said.add(("confirm", stage.get("no"))); log("确认出席:", stname, "qualified=", t.get("qualified"))
            except ApiError as e:
                if e.status == 401:
                    raise
                if ("confirmfail", stage.get("no"), e.status) not in said:
                    said.add(("confirmfail", stage.get("no"), e.status)); log("确认被拒:", e.status, e.body[:100])
            time.sleep(3)
        elif status == "stage_done":
            time.sleep(3)
        elif status == "running":
            try:
                me2 = api("GET", "/api/me")
            except ApiError as e:
                if e.status == 401:
                    raise
                stat("tour_poll_err")
                time.sleep(1.0)
                continue
            act = [g["game_id"] for g in (me2.get("active_games") or [])]
            if not scoped:
                mine = set(t.get("my_games") or [])
                act = [g for g in act if g in mine]
            # 监督式而非阻塞式：每场一个线程，主循环每 2s 对照 active_games——
            #  * 新出现的 game_id（开赛 / 决赛加赛）立即开线程；
            #  * 线程已死但服务端仍列为进行中（对局线程异常退出）→ 重新接管，而不是等同批其它 9 场打完；
            #  * 服务端不再列出的场 = 已结束，线程自然退出。
            for gid in act:
                if gid in _DONE_GAMES:
                    continue
                th = threads.get(gid)
                if th is None or not th.is_alive():
                    if th is not None:
                        stat("game_retake"); log(gid[-14:], "线程已退出但服务端仍进行中 → 重新接管")
                    else:
                        log("新场次", gid[-14:], f"（当前 {sum(1 for x in threads.values() if x.is_alive()) + 1} 场）")
                    th = threading.Thread(target=_safe_play, args=(gid, make_brain()), daemon=True)
                    th.start()
                    threads[gid] = th
            time.sleep(2 if act else 1)
        else:
            time.sleep(1)


def match_loop(make_brain, loops=1):
    for i in range(loops):
        try:
            r = api("POST", "/api/match", {})
        except ApiError as e:
            log("match 失败", e.status, e.body[:120])
            if e.status == 403 and "FEATURE_DISABLED" in e.body:
                log("自由匹配已被管理员关闭（v29 FEATURE_DISABLED，永久条件）——停止匹配")
                return
            if e.status == 404 and "NO_ROOM_AVAILABLE" in e.body:
                log("NO_ROOM_AVAILABLE（永久条件）——停止匹配")
                return
            time.sleep(10)
            continue
        rid = r.get("room_id")
        log(f"[{i+1}/{loops}] 房间 {rid} config={r.get('config')}")
        apply_config(r.get("config"), "match")
        played = set()
        idle = 0
        while True:
            try:
                t = api("GET", f"/api/tournaments/{rid}")
            except ApiError as e:
                if e.status == 404:
                    log("房间已关闭")
                    break
                time.sleep(2)
                continue
            status = t.get("status")
            if status in ("finished", "closed", "void"):
                log("房间终态", status, "ranking=", (t.get("ranking") or [])[:4])
                break
            me2 = api("GET", "/api/me")
            act = [g["game_id"] for g in (me2.get("active_games") or []) if g["game_id"] in set(t.get("my_games") or [])]
            act = [g for g in act if g not in played]
            if act:
                played.update(act)
                log("开打", len(act), "场")
                run_games(act, make_brain)
                idle = 0
            else:
                idle += 1
                time.sleep(2)
                if idle > 900:
                    log("等待过久，放弃本房")
                    break


def main():
    global SERVER, TOKEN, NAME, BUCKET
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=TOKEN)
    ap.add_argument("--server", default=SERVER)
    ap.add_argument("--name", default="hz")
    ap.add_argument("--game", default=None, help="直接打某场；可逗号分隔多场")
    ap.add_argument("--match", action="store_true")
    ap.add_argument("--loops", type=int, default=1)
    ap.add_argument("--params", default="", help="覆盖引擎参数，如 hazard_base=0.08,lookahead=1")
    ap.add_argument("--sse", action="store_true", help="用 /notify SSE 触发增量拉取（默认长轮询游标）")
    ap.add_argument("--chi-early", action="store_true", help="他家出牌一到即抢发 chi（可能 409 后再按时机重试）")
    ap.add_argument("--rate", type=float, default=14, help="全局限速：滑动 1s 窗口最多请求数（服务端 16/s）")
    ap.add_argument("--coalesce", type=float, default=0.12, help="事件批之间的合并等待秒数（省配额；0 关闭）")
    ap.add_argument("--dump", default=None, help="原始收发落盘目录（每场一个 jsonl）")
    ap.add_argument("--no-guide-check", action="store_true")
    ap.add_argument("--rules", default="", help="--game 直打时手动指定规则，如 YouCaiBiKao=true（锦标赛/匹配模式自动读 config）")
    args = ap.parse_args()
    SERVER, TOKEN, NAME = args.server, args.token, args.name
    if not TOKEN:
        print("缺少令牌：--token 或 HZ_TOKEN")
        raise SystemExit(1)
    OPTS["sse"], OPTS["chi_early"], OPTS["rate"], OPTS["dump"] = args.sse, args.chi_early, args.rate, args.dump
    OPTS["coalesce"] = max(0.0, args.coalesce)
    BUCKET.rate = max(3, int(args.rate))
    from hz_brain import parse_params
    params = parse_params(args.params)      # 支持 @预设 与 k=[列表]（逗号在方括号内不切分）
    log("引擎参数覆盖:", params or "(默认)")
    make_brain = lambda: Brain(params)
    if args.rules:
        cfg = {}
        for item in args.rules.split(","):
            k, v = item.split("=", 1)
            cfg[k.strip()] = json.loads(v.lower()) if v.lower() in ("true", "false") else (float(v) if v.replace(".", "", 1).isdigit() else v)
        apply_config(cfg, "cli")
    if not args.no_guide_check:
        check_guide_version()
    try:
        if args.game:
            run_games([g for g in args.game.split(",") if g], make_brain)
        elif args.match:
            match_loop(make_brain, args.loops)
        else:
            tournament_loop(make_brain)
    except ApiError as e:
        log("API 错误", e.status, e.body[:200])
    except KeyboardInterrupt:
        log("手动停止")
    log("stats=", dict(STATS))


if __name__ == "__main__":
    main()
