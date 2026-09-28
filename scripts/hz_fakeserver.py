#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hz_fakeserver —— 本地假平台服务器，用于 hz_bot 的协议/时序联调（不联网也能测）。

尽量按 guide v27 + 真实事件流（data_ranking 400 局）观察到的行为复刻：
  * GET /api/games/{id}/state?seq=N  ：seq=0 全量快照 {snapshot,seq}；seq=N → {events:[...]}（N 之后）；
    无新事件挂起 ≤30s → {"pending":true}；跨局（新局尚无事件）或落后 >256 → {snapshot,seq,gap:true}；
    结束 → {"finished":true,"snapshot":...}
  * GET /api/games/{id}/notify        ：SSE，初始帧 {"seq":N}，每次状态变化 {"seq":N}，30s keepalive，结束 {"seq":N,"closed":true}
  * POST /api/games/{id}/action       ：纯验证，非法/重复 → 409 INVALID_ACTION
  * 每用户 16 req/s 限流 → 429 RATE_LIMITED
  * 时序：出牌思考 3s（超时：能胡自动胡，否则打最右一张）；碰窗口 1s 固定走满（三家）→ 吃窗口 1s（下家）→ 下家摸牌；
    碰/杠/吃提交即刻生效；碰窗口内提交 chi → 409；抓打圈内出牌不开窗口（仅豁免方）；打出财神不开窗口；
    round_ended 后约 1s 新局发牌（无事件）；8 局后 game_ended。
  * 其它三家为内置 agent（GreedyAgent/Brain），带随机反应延迟（可模拟部分超时）。
  * 落盘：每场 <out>/<gid>.json，格式同 data_ranking（blocks/start_hands/events），可直接喂 hz_audit/hz_replay。

用法：
  python hz_fakeserver.py --port 18099 --games 10 --rounds 8 --out data_fake/run1 [--brain-others]
  然后另开终端：python hz_bot.py --server http://127.0.0.1:18099 --token T --game fake_g0,fake_g1,... --no-guide-check
  （服务器启动时会打印完整的 --game 参数）
"""
import json, sys, os, io, time, random, threading, argparse, collections
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hz_core import (GOD, TILE_NAMES, TILE_INDEX, hand_to_counts, counts_to_hand, is_win, is_baotou,
                     calc_fan, is_suited, rank_of)
from hz_sim import GreedyAgent
from hz_brain import Brain

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

GOD_T = TILE_NAMES[GOD]
TIMING = {"discard": 3.0, "peng": 1.0, "chi": 1.0, "round_gap": 5.0, "pending": 30.0, "gap_lag": 256}   # round_gap=5：指南 v31 局间固定 5s
LOG_LOCK = threading.Lock()


def log(*a):
    with LOG_LOCK:
        print(time.strftime("%H:%M:%S"), "[fake]", *a, flush=True)


class Engine:
    def __init__(self, gid, ext_seat, agents, seed=0, rounds=8, latency=(0.1, 0.9), p_lag=0.05, out=None):
        self.gid, self.ext_seat, self.agents = gid, ext_seat, agents
        self.rng = random.Random(seed)
        self.rounds_total = rounds
        self.latency, self.p_lag = latency, p_lag
        self.out = out
        self.cv = threading.Condition()
        self.events = []
        self.seq = 0
        self.round_start_seq = 1
        self.round_no = 0
        self.dealer = 0
        self.scores = [0, 0, 0, 0]
        self.finished = False
        self.blocks = []
        self.round_results = []
        self.internal = []   # (due_time, seat, act, phase_id)
        self.phase_id = 0
        self.stats = collections.Counter()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    # ---- 事件 ----
    def emit(self, typ, seat, tile="", data=None):
        self.seq += 1
        e = {"seq": self.seq, "type": typ, "seat": seat, "tile": tile or "", "data": data, "ts": int(time.time())}
        self.events.append(e)
        self.cur_block["events"].append(e)
        self.cv.notify_all()
        return e

    def _bump(self):
        self.phase_id += 1
        self.internal = [x for x in self.internal if x[3] >= self.phase_id]

    # ---- 局 ----
    def new_round(self):
        self.round_no += 1
        tiles = [i for i in range(34) for _ in range(4)]
        self.rng.shuffle(tiles)
        self.hands = [[0] * 34 for _ in range(4)]
        for s in range(4):
            for t in tiles[s * 13:(s + 1) * 13]:
                self.hands[s][t] += 1
        self.wall = tiles[52:]
        self.melds = [[] for _ in range(4)]
        self.discards = [[] for _ in range(4)]
        self.chi_count = [0] * 4
        self.chain = [0] * 4
        self.piao = [0] * 4
        self.restricted = [False] * 4
        self.god_discarder = -1
        self.circle_left = 0
        self.last_discard = None
        self.last_discard_seat = None
        self.drawn = None
        self.baotou13 = [False] * 4
        self.responded = set()
        self.responding = []
        self.round_start_seq = self.seq + 1
        # 庄家第 14 张直抽（无事件）
        t = self.wall.pop(0)
        self.hands[self.dealer][t] += 1
        self.drawn = TILE_NAMES[t]
        self.cur_block = {"round_no": self.round_no, "dealer": self.dealer, "seq_start": self.round_start_seq,
                          "seq_end": None, "truncated": False,
                          "start_hands": [counts_to_hand(h) for h in self.hands], "events": []}
        self.blocks.append(self.cur_block)
        self.phase, self.turn = "draw", self.dealer
        self.deadline = time.time() + TIMING["discard"]
        self._bump()
        self.plan_internal_draw()
        self.cv.notify_all()

    # ---- 快照 ----
    def snapshot(self, seat):
        if self.round_no == 0:
            return {"seat": seat, "phase": "deal", "round_no": 0}
        h = self.hands[seat] if 0 <= seat < 4 else [0] * 34
        nm = len(self.melds[seat]) if 0 <= seat < 4 else 0
        drawn = self.drawn if (self.phase == "draw" and self.turn == seat) else None
        if drawn:
            h13 = list(h); h13[TILE_INDEX[drawn]] -= 1
            bt = is_baotou(h13, nm)
        else:
            bt = is_baotou(h, nm) if sum(h) % 3 == 1 else False
        cp = self.circle_left > 0
        return {
            "seat": seat, "phase": self.phase, "turn": self.turn,
            "responding_seats": list(self.responding) if self.phase.startswith("response") else [],
            "drawn_tile": drawn, "my_hand": counts_to_hand(h),
            "discards": [list(d) for d in self.discards],
            "melds": [[{"kind": m["kind"], "tiles": list(m["tiles"])} for m in ms] for ms in self.melds],
            "hand_counts": [sum(x) for x in self.hands], "wall_remaining": len(self.wall),
            "last_discard": self.last_discard, "dealer": self.dealer, "round_no": self.round_no,
            "scores": list(self.scores),
            "god": {"baotou": bt, "chain_count": self.chain[seat] if 0 <= seat < 4 else 0,
                    "catch_play": cp, "god_discarder_seat": self.god_discarder if cp else -1},
        }

    # ---- 内置 agent 计划 ----
    def _lat(self):
        if self.rng.random() < self.p_lag:
            return 5.0   # 模拟卡顿：超时
        return self.rng.uniform(*self.latency)

    def plan_internal_draw(self):
        s = self.turn
        if s == self.ext_seat:
            return
        snap = self.snapshot(s)
        snap["last_discard_seat"] = self.last_discard_seat
        snap["rules"] = {"wall_reserve": 20, "gang_min_wall": 21, "max_chi": 2}
        try:
            act = self.agents[s].act_draw(snap) or {"action": "discard", "tile": snap["my_hand"][-1]}
        except Exception:
            act = {"action": "discard", "tile": snap["my_hand"][-1]}
        self.internal.append((time.time() + min(self._lat(), 2.5) * 0.6, s, act, self.phase_id))

    def plan_internal_response(self):
        for s in self.responding:
            if s == self.ext_seat:
                continue
            snap = self.snapshot(s)
            snap["last_discard_seat"] = self.last_discard_seat
            snap["rules"] = {"wall_reserve": 20, "gang_min_wall": 21, "max_chi": 2}
            try:
                act = self.agents[s].act_response(snap) or {"action": "pass"}
            except Exception:
                act = {"action": "pass"}
            if self.phase == "response_peng" and act.get("action") == "chi":
                act = {"action": "pass"}
            self.internal.append((time.time() + self._lat() * 0.9, s, act, self.phase_id))

    # ---- 规则 ----
    def can_hu(self, s):
        return self.drawn is not None and self.turn == s and self.phase == "draw" and is_win(self.hands[s], len(self.melds[s]))

    def rightmost(self, s):
        for i in range(33, -1, -1):
            if self.hands[s][i] > 0:
                return TILE_NAMES[i]

    def finish_hu(self, s):
        fan, detail = calc_fan(self.hands[s], len(self.melds[s]), self.baotou13[s], self.chain[s], self.piao[s])
        delta = [0, 0, 0, 0]
        for k in range(4):
            if k != s:
                delta[k] = -(8 if (s == self.dealer or k == self.dealer) else 1) * fan
        delta[s] = -sum(delta)
        for k in range(4):
            self.scores[k] += delta[k]
        self.round_results.append({"round_no": self.round_no, "winner": s, "fan": fan, "detail": detail, "scores": delta, "dealer": self.dealer})
        self.emit("round_ended", s, "", {"dealer": self.dealer, "detail": detail, "draw": False, "fan": fan,
                                         "round_no": self.round_no, "scores": delta})
        self.dealer = s
        self.end_round()

    def finish_draw_round(self):
        self.round_results.append({"round_no": self.round_no, "winner": -1, "fan": 0, "detail": [], "scores": [0] * 4, "dealer": self.dealer})
        self.emit("round_ended", -1, "", {"dealer": self.dealer, "detail": [], "draw": True, "fan": 0,
                                          "round_no": self.round_no, "scores": [0, 0, 0, 0]})
        self.end_round()

    def end_round(self):
        self.cur_block["seq_end"] = self.seq
        self.phase, self.turn, self.responding = "settled", -1, []
        self._bump()
        if self.round_no >= self.rounds_total:
            self.emit("game_ended", -1, "", {"scores": list(self.scores)})
            self.finished = True
            self.cv.notify_all()
            self.save()
            log(self.gid, "结束 scores=", self.scores, "stats=", dict(self.stats))
        else:
            self.deadline = time.time() + TIMING["round_gap"]

    def do_discard(self, s, tile, timeout=False):
        ti = TILE_INDEX[tile]
        self.hands[s][ti] -= 1
        self.discards[s].append(tile)
        self.last_discard, self.last_discard_seat = tile, s
        was_restricted = self.restricted[s]
        self.restricted[s] = False
        if ti == GOD and is_baotou(self.hands[s], len(self.melds[s])):
            self.chain[s] += 1; self.piao[s] += 1
        else:
            self.chain[s] = 0; self.piao[s] = 0
        self.drawn = None
        self.emit("tile_discarded", s, tile, {"catch_play": bool(was_restricted)})
        if timeout:
            self.emit("timeout", s, "", {"kind": "discard"})
        self.stats["discard"] += 1
        self._bump()
        if ti == GOD:
            self.god_discarder = s
            self.circle_left = 3
            for k in range(4):
                self.restricted[k] = (k != s)
            self.phase, self.responding = "response_peng", []
            self.deadline = time.time() + TIMING["peng"] + TIMING["chi"]
            return
        if was_restricted:
            self.circle_left -= 1
            if self.circle_left <= 0:
                self.circle_left = 0; self.god_discarder = -1
            # 圈内出牌：窗口仅对豁免方开放
            self.responding = [k for k in range(4) if k != s and not self.restricted[k]]
        else:
            self.responding = [(s + k) % 4 for k in (1, 2, 3)]
        self.responded = set()
        self.phase = "response_peng"
        self.deadline = time.time() + TIMING["peng"]
        self.plan_internal_response()

    def next_draw(self, s):
        if len(self.wall) <= 20:
            self.finish_draw_round()
            return
        t = self.wall.pop(0)
        self.baotou13[s] = is_baotou(self.hands[s], len(self.melds[s]))
        self.hands[s][t] += 1
        self.drawn = TILE_NAMES[t]
        self.phase, self.turn, self.responding = "draw", s, []
        self.emit("tile_drawn", s, self.drawn, None)
        self.deadline = time.time() + TIMING["discard"]
        self._bump()
        self.plan_internal_draw()

    def replenish(self, s):
        t = self.wall.pop()
        self.baotou13[s] = is_baotou(self.hands[s], len(self.melds[s]))
        self.hands[s][t] += 1
        self.drawn = TILE_NAMES[t]
        self.phase, self.turn, self.responding = "draw", s, []
        self.emit("tile_drawn", s, self.drawn, {"gang_replenish": True})
        self.deadline = time.time() + TIMING["discard"]
        self._bump()
        self.plan_internal_draw()

    # ---- 动作提交（HTTP 线程/内置 agent 均走这里，须持锁）----
    def submit(self, s, act):
        a = act.get("action")
        tile = act.get("tile") or None
        if self.finished or self.phase in ("settled", "deal"):
            return 409, "INVALID_ACTION: not in play"
        h = self.hands[s]
        nm = len(self.melds[s])
        if self.phase == "draw":
            if self.turn != s:
                return 409, "INVALID_ACTION: not your turn"
            if a == "hu":
                if not self.can_hu(s):
                    self.emit("timeout", s, "", {"kind": "hu_failed"})
                    return 409, "INVALID_ACTION: cannot hu"
                self.stats["hu"] += 1
                self.finish_hu(s)
                return 200, {"ok": True}
            if a == "gang":
                if not tile or tile not in TILE_INDEX or tile == GOD_T or len(self.wall) < 21:
                    return 409, "INVALID_ACTION: gang not allowed"
                ti = TILE_INDEX[tile]
                if h[ti] == 4:
                    h[ti] -= 4; self.melds[s].append({"kind": "gang_an", "tiles": [tile] * 4}); kind = "an"
                elif h[ti] >= 1 and any(m["kind"] == "peng" and m["tiles"][0] == tile for m in self.melds[s]):
                    if self.restricted[s]:
                        return 409, "INVALID_ACTION: restricted"
                    h[ti] -= 1
                    for m in self.melds[s]:
                        if m["kind"] == "peng" and m["tiles"][0] == tile:
                            m["kind"] = "gang_bu"; m["tiles"] = [tile] * 4; break
                    kind = "bu"
                else:
                    return 409, "INVALID_ACTION: no gang"
                self.chain[s] += 1
                self.restricted[s] = False
                self.emit("gang", s, tile, {"kind": kind})
                self.stats["gang"] += 1
                self.replenish(s)
                return 200, {"ok": True}
            if a == "discard":
                if not tile or tile not in TILE_INDEX or h[TILE_INDEX[tile]] <= 0:
                    return 409, "INVALID_ACTION: tile not in hand"
                if self.restricted[s] and self.drawn and tile != self.drawn:
                    return 409, "INVALID_ACTION: catch_play restricted"
                if sum(h) % 3 != 2:
                    return 409, "INVALID_ACTION: hand size"
                self.do_discard(s, tile)
                return 200, {"ok": True}
            return 409, "INVALID_ACTION: bad action in draw"
        # 响应窗口
        if s not in self.responding:
            return 409, "INVALID_ACTION: not responding"
        if s in self.responded:
            return 409, "INVALID_ACTION: already responded"
        last = self.last_discard
        ti = TILE_INDEX[last]
        if a == "pass":
            self.responded.add(s)
            self.emit("pass", s, "", None)
            return 200, {"ok": True}
        if self.phase == "response_peng":
            if a == "gang":
                if h[ti] >= 3 and len(self.wall) >= 21:
                    h[ti] -= 3
                    self.melds[s].append({"kind": "gang_ming", "tiles": [last] * 4})
                    self.chain[s] += 1
                    self.emit("gang", s, last, {"kind": "ming"})
                    self.stats["gang"] += 1
                    self.replenish(s)
                    return 200, {"ok": True}
                return 409, "INVALID_ACTION: cannot gang"
            if a == "peng":
                if h[ti] >= 2 and tile in (None, last):
                    h[ti] -= 2
                    self.melds[s].append({"kind": "peng", "tiles": [last] * 3})
                    self.chain[s] = 0; self.piao[s] = 0
                    self.emit("peng", s, last, None)
                    self.stats["peng"] += 1
                    self.phase, self.turn, self.responding = "draw", s, []
                    self.drawn = None
                    self.deadline = time.time() + TIMING["discard"]
                    self._bump()
                    self.plan_internal_draw()
                    return 200, {"ok": True}
                return 409, "INVALID_ACTION: cannot peng"
            if a == "chi":
                self.stats["chi_in_peng_window"] += 1
                return 409, "INVALID_ACTION: chi not open yet"
            return 409, "INVALID_ACTION"
        # response_chi
        if a == "chi":
            if not is_suited(ti) or self.chi_count[s] >= 2 or (self.last_discard_seat + 1) % 4 != s:
                return 409, "INVALID_ACTION: cannot chi"
            r = rank_of(ti); base = ti - (r - 1)
            combos = []
            for lo in (r - 2, r - 1, r):
                if lo < 1 or lo + 2 > 9:
                    continue
                need = [base + lo - 1 + k for k in range(3) if base + lo - 1 + k != ti]
                if all(h[x] >= 1 for x in need):
                    combos.append(need)
            want = act.get("tiles")
            chosen = None
            for c in combos:
                if not want or sorted(want) == sorted(TILE_NAMES[x] for x in c):
                    chosen = c; break
            if chosen is None:
                return 409, "INVALID_ACTION: chi combo"
            for x in chosen:
                h[x] -= 1
            meld = [TILE_NAMES[x] for x in chosen] + [last]
            self.melds[s].append({"kind": "chi", "tiles": meld})
            self.chi_count[s] += 1
            self.chain[s] = 0; self.piao[s] = 0
            self.emit("chi", s, last, {"tiles": meld})
            self.stats["chi"] += 1
            self.phase, self.turn, self.responding = "draw", s, []
            self.drawn = None
            self.deadline = time.time() + TIMING["discard"]
            self._bump()
            self.plan_internal_draw()
            return 200, {"ok": True}
        return 409, "INVALID_ACTION: bad action in chi window"

    # ---- 主循环 ----
    def loop(self):
        with self.cv:
            self.new_round()
            while not self.finished:
                now = time.time()
                # 内置 agent 到期动作
                due = [x for x in self.internal if x[0] <= now and x[3] == self.phase_id]
                for x in due:
                    if x in self.internal:      # submit 可能已 _bump 换掉队列
                        self.internal.remove(x)
                    if x[3] == self.phase_id:
                        self.submit(x[1], x[2])
                if self.finished:
                    break
                now = time.time()
                if now >= self.deadline:
                    self.on_deadline()
                    continue
                waits = [self.deadline - now] + [x[0] - now for x in self.internal if x[3] == self.phase_id]
                self.cv.wait(timeout=max(0.005, min(waits)))

    def on_deadline(self):
        ph = self.phase
        if ph == "settled":
            self.new_round()
        elif ph == "draw":
            s = self.turn
            if self.can_hu(s):
                self.stats["auto_hu"] += 1
                self.finish_hu(s)
            else:
                self.stats["discard_timeout"] += 1
                if s == self.ext_seat:
                    self.stats["ext_discard_timeout"] += 1
                    log(self.gid, f"外部 bot 出牌超时：局{self.round_no} 第{len(self.discards[s]) + 1}张 "
                                  f"{'首打(庄)' if not self.discards[s] and s == self.dealer else ''} wall={len(self.wall)}")
                tile = self.drawn if self.restricted[s] and self.drawn else self.rightmost(s)
                self.do_discard(s, tile, timeout=True)
        elif ph == "response_peng":
            if self.last_discard == GOD_T:
                # 财神出牌：无窗口事件，直接下家摸牌
                self.next_draw((self.last_discard_seat + 1) % 4)
                return
            for k in self.responding:
                if k not in self.responded:
                    self.emit("timeout", k, "", {"kind": "response", "window": "peng"})
            nxt = (self.last_discard_seat + 1) % 4
            self.responding = [nxt] if nxt in self.responding or not self.restricted[nxt] else []
            if self.restricted[nxt]:
                self.responding = []
            self.responded = set()
            self.phase = "response_chi"
            self.deadline = time.time() + TIMING["chi"]
            self._bump()
            self.plan_internal_response()
        elif ph == "response_chi":
            for k in self.responding:
                if k not in self.responded:
                    self.emit("timeout", k, "", {"kind": "response", "window": "chi"})
            self.next_draw((self.last_discard_seat + 1) % 4)

    def save(self):
        if not self.out:
            return
        os.makedirs(self.out, exist_ok=True)
        d = {"game_id": self.gid, "seats": [{"user_id": f"u_seat{s}" if s != self.ext_seat else "u_ext", "name": f"seat{s}"} for s in range(4)],
             "rounds": self.round_results, "status": "finished", "blocks": self.blocks}
        with open(os.path.join(self.out, self.gid + ".json"), "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)

    # ---- /state ----
    def state(self, seat, n):
        with self.cv:
            deadline = time.time() + TIMING["pending"]
            while True:
                if self.finished:
                    return {"finished": True, "snapshot": self.snapshot(seat), "seq": self.seq}
                if n == 0:
                    return {"snapshot": self.snapshot(seat), "seq": self.seq}
                # 游标来自上一局（新局发牌无事件）或落后过多 → 全量快照 + gap
                if n < self.round_start_seq or (self.seq - n) > TIMING["gap_lag"]:
                    return {"snapshot": self.snapshot(seat), "seq": self.seq, "gap": True}
                evs = [e for e in self.events if e["seq"] > n]
                if evs:
                    return {"events": evs}
                rem = deadline - time.time()
                if rem <= 0:
                    return {"pending": True}
                self.cv.wait(timeout=rem)


# ---------------- HTTP ----------------
class RateLimiter:
    def __init__(self, per_sec=16):
        self.per = per_sec
        self.hist = collections.defaultdict(collections.deque)
        self.lock = threading.Lock()

    def allow(self, tok):
        now = time.time()
        with self.lock:
            q = self.hist[tok]
            while q and q[0] < now - 1.0:
                q.popleft()
            if len(q) >= self.per:
                return False
            q.append(now)
            return True


GAMES = {}
RL = RateLimiter(16)
COUNTS = collections.Counter()
EXT_TOKEN = "T"
TOUR = None      # --tournament 模式：多阶段锦标赛生命周期模拟（指南 §2.6）


class Tournament:
    """复刻指南 §2.6 多阶段锦标赛协议：registering →(报名+ready)→ running → stage_done → stage_open(须重新 ready)
    → running → … → finished。每阶段新建 games_per_stage 场（game_id 含 _s{k}），/api/me 只列未结束场。
    用于验证 hz_bot.tournament_loop：报名/到位、阶段确认、空转不退出、5xx 风暴不退出、加赛新 id 自动接入。"""

    def __init__(self, args):
        self.tid = "t_fake"
        self.status = "registering"
        self.stage_no = 0
        self.stages = args.stages
        self.games_per_stage = args.games
        self.args = args
        self.registered = False
        self.ready = False            # 当前阶段是否已 ready（确认不跨阶段继承）
        self.my_games = []
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def info(self):
        with self.lock:
            d = {"id": self.tid, "status": self.status, "kind": "",
                 "config": {"M": self.games_per_stage, "Rounds": self.args.rounds, "BaseScore": 1, "Kind": "",
                            "YouCaiBiKao": False, "DiscardTimeoutSec": TIMING["discard"],
                            "PengTimeoutSec": TIMING["peng"], "ChiTimeoutSec": TIMING["chi"]},
                 "my_games": list(self.my_games), "ranking": [],
                 "registered_users": int(self.registered), "ready_users": int(self.ready) if self.stage_no == 0 else 0}
            if self.stage_no:
                d.update({"stage": {"no": self.stage_no, "role": "final" if self.stage_no == self.stages else "qualify",
                                    "total": self.stages, "name": f"fake-第{self.stage_no}轮"},
                          "stage_status": {"running": "running", "stage_done": "done", "stage_open": "open"}.get(self.status, "running"),
                          "stage_crashed": False, "qualified": True, "qualify_role": "finalist"})
            return d

    def register(self):
        with self.lock:
            if self.status != "registering":
                return 409, {"error": "TOURNAMENT_STARTED"}
            self.registered = True
            return 200, {"ok": True}

    def do_ready(self):
        with self.lock:
            if self.status == "registering":
                if not self.registered:
                    return 409, {"error": "NOT_REGISTERED"}
                self.ready = True
                return 200, {"ok": True}
            if self.status == "stage_open":
                self.ready = True
                return 200, {"ok": True}
            return 409, {"error": "TOURNAMENT_STARTED"}

    def active_games(self):
        with self.lock:
            return [g for g in self.my_games if not GAMES[g].finished]

    def _start_stage(self):
        self.stage_no += 1
        gids = []
        for i in range(self.games_per_stage):
            gid = f"fake_s{self.stage_no}_g{i}"
            agents = [Brain() if self.args.brain_others else GreedyAgent() for _ in range(4)]
            GAMES[gid] = Engine(gid, i % 4, agents, seed=self.args.seed * 1000 + self.stage_no * 100 + i,
                                rounds=self.args.rounds, p_lag=self.args.p_lag, out=self.args.out)
            gids.append(gid)
        with self.lock:
            self.my_games += gids
            self.status = "running"
            self.ready = False
        for g in gids:
            GAMES[g].thread.start()
        log(f"锦标赛 阶段{self.stage_no}/{self.stages} running：新场 {gids}")
        return gids

    def _wait(self, sec):
        t_end = time.time() + sec
        while time.time() < t_end:
            time.sleep(0.2)

    def loop(self):
        log(f"锦标赛 {self.tid} registering（{self.args.reg_sec}s）")
        self._wait(self.args.reg_sec)
        COUNTS["reg_ok" if (self.registered and self.ready) else "reg_missed"] += 1
        log("报名期结束 registered=", self.registered, "ready=", self.ready)
        for k in range(self.stages):
            if k > 0:
                with self.lock:
                    self.status = "stage_done"
                log(f"锦标赛 stage_done（等推进 {self.args.gap_sec}s）")
                self._wait(self.args.gap_sec)
                with self.lock:
                    self.status = "stage_open"
                log(f"锦标赛 stage_open（确认期 {self.args.gap_sec}s，须重新 ready）")
                self._wait(self.args.gap_sec)
                COUNTS["stage_confirm_ok" if self.ready else "stage_confirm_missed"] += 1
                log("确认期结束 ready=", self.ready)
            gids = self._start_stage()
            while any(not GAMES[g].finished for g in gids):
                time.sleep(0.5)
        with self.lock:
            self.status = "finished"
        log("锦标赛 finished")
# 故障注入（仅对外部 bot 生效）：
#   p_5xx    每个请求随机返回 502/503 的概率
#   p_drop   每个请求随机「半路断开」（不发响应直接关 socket）的概率
#   p_sse    SSE 连接每 10s 随机被掐断的概率
#   storm    (start_s, dur_s)：启动后第 start_s 秒开始，连续 dur_s 秒所有请求都 502（复刻 R1 首房的 502 风暴）
#   p_garbage 每个 state 响应随机返回非 JSON 正文的概率
FAULT = {"p_5xx": 0.0, "p_drop": 0.0, "p_sse": 0.0, "storm": None, "p_garbage": 0.0, "game_storm": None, "t0": time.time()}
FAULT_RNG = random.Random(7)


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def _fault(self, kind):
        """返回 True 表示本请求已被故障注入吞掉（响应已发或连接已断）。"""
        if self._auth() != EXT_TOKEN:
            return False
        st = FAULT["storm"]
        if st:
            el = time.time() - FAULT["t0"]
            if st[0] <= el < st[0] + st[1]:
                COUNTS["fault_storm"] += 1
                self._json(502, {"error": "BAD_GATEWAY"}); return True
        gs = FAULT["game_storm"]
        if gs and kind in ("state", "action") and gs[0] in self.path:
            # 只对某一场的 /state /action 持续 502（其它端点正常）→ 逼对局线程内部重试耗尽后退出，验证主循环重新接管
            el = time.time() - FAULT["t0"]
            if gs[1] <= el < gs[1] + gs[2]:
                COUNTS["fault_game_storm"] += 1
                self._json(502, {"error": "BAD_GATEWAY"}); return True
        r = FAULT_RNG.random()
        if r < FAULT["p_5xx"]:
            COUNTS["fault_5xx"] += 1
            self._json(FAULT_RNG.choice((502, 503)), {"error": "UPSTREAM"}); return True
        if r < FAULT["p_5xx"] + FAULT["p_drop"]:
            COUNTS["fault_drop"] += 1
            try:
                self.connection.shutdown(1)
            except OSError:
                pass
            self.close_connection = True
            return True
        if kind == "state" and r < FAULT["p_5xx"] + FAULT["p_drop"] + FAULT["p_garbage"]:
            COUNTS["fault_garbage"] += 1
            body = b"<html>502 Bad Gateway</html>"
            self.send_response(200); self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            return True
        return False

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _auth(self):
        a = self.headers.get("Authorization", "")
        return a.split(" ", 1)[1] if " " in a else ""

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        tok = self._auth()
        parts = u.path.strip("/").split("/")
        if u.path == "/api/me":
            if TOUR is not None:
                if self._fault("me"):
                    return
                return self._json(200, {"user_id": "u_ext", "active_games": [{"game_id": g} for g in TOUR.active_games()],
                                        "tournament_id": TOUR.tid})
            return self._json(200, {"user_id": "u_ext", "active_games": [{"game_id": g} for g in GAMES], "tournament_id": ""})
        if u.path == "/portal/api/guide/version":
            return self._json(200, {"version": 27, "changes": []})
        if TOUR is not None and parts[:2] == ["api", "tournaments"] and len(parts) == 3:
            if parts[2] not in (TOUR.tid, "me"):
                return self._json(404, {"error": "TOURNAMENT_NOT_FOUND"})
            COUNTS["tour_get"] += 1
            if self._fault("tour"):
                return
            return self._json(200, TOUR.info())
        if TOUR is not None and u.path == "/api/tournaments/me/rules":
            return self._json(200, {"tournament_id": TOUR.tid, "status": TOUR.status, "config": TOUR.info()["config"]})
        if len(parts) >= 4 and parts[1] == "games":
            gid = parts[2]
            g = GAMES.get(gid)
            if not g:
                return self._json(404, {"error": "GAME_NOT_FOUND"})
            seat = g.ext_seat if tok == EXT_TOKEN else -1
            if parts[3] == "state":
                COUNTS["state"] += 1
                if not RL.allow(tok):
                    COUNTS["429"] += 1
                    return self._json(429, {"error": "RATE_LIMITED"})
                if self._fault("state"):
                    return
                n = int((q.get("seq") or ["0"])[0])
                COUNTS["state_seq0" if n == 0 else "state_inc"] += 1
                return self._json(200, g.state(seat, n))
            if parts[3] == "notify":
                COUNTS["notify"] += 1
                if self._fault("notify"):
                    return
                return self.sse(g)
        self._json(404, {"error": "NOT_FOUND"})

    def sse(self, g):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            with g.cv:
                last = g.seq
                self.wfile.write(f'data: {json.dumps({"seq": last})}\n\n'.encode()); self.wfile.flush()
                while True:
                    if g.finished:
                        self.wfile.write(f'data: {json.dumps({"seq": g.seq, "closed": True})}\n\n'.encode()); self.wfile.flush()
                        return
                    if g.seq != last:
                        last = g.seq
                        self.wfile.write(f'data: {json.dumps({"seq": last})}\n\n'.encode()); self.wfile.flush()
                        continue
                    if not g.cv.wait(timeout=10 if FAULT["p_sse"] > 0 else 30):
                        if FAULT["p_sse"] > 0 and FAULT_RNG.random() < FAULT["p_sse"]:
                            COUNTS["fault_sse_kill"] += 1
                            return          # 直接断开 SSE，不发 closed
                        self.wfile.write(b": keepalive\n\n"); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def do_POST(self):
        u = urlparse(self.path)
        tok = self._auth()
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            body = json.loads(raw.decode() or "{}")
        except Exception:
            body = {}
        parts = u.path.strip("/").split("/")
        if TOUR is not None and parts[:2] == ["api", "tournaments"] and len(parts) == 4 and parts[3] in ("register", "ready"):
            if parts[2] not in (TOUR.tid, "me"):
                return self._json(404, {"error": "TOURNAMENT_NOT_FOUND"})
            if tok != EXT_TOKEN:
                return self._json(403, {"error": "FORBIDDEN"})
            if self._fault("tour"):
                return
            code, r = TOUR.register() if parts[3] == "register" else TOUR.do_ready()
            COUNTS[f"tour_{parts[3]}_{code}"] += 1
            return self._json(code, r)
        if len(parts) >= 4 and parts[1] == "games" and parts[3] == "action":
            g = GAMES.get(parts[2])
            if not g:
                return self._json(404, {"error": "GAME_NOT_FOUND"})
            COUNTS["action"] += 1
            if not RL.allow(tok):
                COUNTS["429"] += 1
                return self._json(429, {"error": "RATE_LIMITED"})
            if tok != EXT_TOKEN:
                return self._json(403, {"error": "FORBIDDEN"})
            if self._fault("action"):
                return
            with g.cv:
                code, r = g.submit(g.ext_seat, body)
                if code == 200:
                    COUNTS["action_ok"] += 1
                    COUNTS["ok_" + body.get("action", "?")] += 1
                    return self._json(200, r)
                COUNTS["409"] += 1
                COUNTS["409_" + body.get("action", "?")] += 1
                return self._json(409, {"error": "INVALID_ACTION", "message": r})
        self._json(404, {"error": "NOT_FOUND"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18099)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="data_fake/run")
    ap.add_argument("--brain-others", action="store_true", help="其它三家用 Brain（默认 GreedyAgent）")
    ap.add_argument("--p-lag", type=float, default=0.05)
    ap.add_argument("--discard-sec", type=float, default=3.0)
    ap.add_argument("--pending-sec", type=float, default=30.0)
    ap.add_argument("--start-delay", type=float, default=3.0, help="启动后延迟几秒开局（给 bot 连接时间）")
    ap.add_argument("--fault-5xx", type=float, default=0.0, help="每请求随机 502/503 概率")
    ap.add_argument("--fault-drop", type=float, default=0.0, help="每请求随机断连概率")
    ap.add_argument("--fault-sse", type=float, default=0.0, help="SSE 每 10s 被掐断概率")
    ap.add_argument("--fault-garbage", type=float, default=0.0, help="state 响应返回非 JSON 概率")
    ap.add_argument("--fault-storm", default="", help="start,dur：第 start 秒起连续 dur 秒全部 502")
    ap.add_argument("--fault-game-storm", default="",
                    help="gid_substr,start,dur：仅该场的 /state /action 在第 start 秒起连续 dur 秒 502（验证对局线程死后主循环重新接管）")
    ap.add_argument("--tournament", action="store_true",
                    help="多阶段锦标赛生命周期模式：bot 不带 --game 直接以锦标赛模式连接（令牌 T 绑定 t_fake）")
    ap.add_argument("--stages", type=int, default=2, help="--tournament：阶段数（每阶段 --games 场）")
    ap.add_argument("--reg-sec", type=float, default=6.0, help="--tournament：报名期秒数")
    ap.add_argument("--gap-sec", type=float, default=4.0, help="--tournament：stage_done / stage_open 各持续秒数")
    args = ap.parse_args()
    TIMING["discard"] = args.discard_sec
    TIMING["pending"] = args.pending_sec
    FAULT.update(p_5xx=args.fault_5xx, p_drop=args.fault_drop, p_sse=args.fault_sse, p_garbage=args.fault_garbage)
    if args.fault_storm:
        a, b = args.fault_storm.split(","); FAULT["storm"] = (float(a), float(b))
    if args.fault_game_storm:
        g, a, b = args.fault_game_storm.split(","); FAULT["game_storm"] = (g, float(a), float(b))
    if any((args.fault_5xx, args.fault_drop, args.fault_sse, args.fault_garbage, args.fault_storm, args.fault_game_storm)):
        log("故障注入:", {k: v for k, v in FAULT.items() if k != "t0"})
    global TOUR
    if args.tournament:
        TOUR = Tournament(args)
    else:
        for i in range(args.games):
            gid = f"fake_g{i}"
            agents = [Brain() if args.brain_others else GreedyAgent() for _ in range(4)]
            GAMES[gid] = Engine(gid, i % 4, agents, seed=args.seed * 1000 + i, rounds=args.rounds, p_lag=args.p_lag, out=args.out)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), H)
    srv.daemon_threads = True
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    if TOUR is not None:
        log(f"listening http://127.0.0.1:{args.port}  锦标赛模式 stages={args.stages} games/stage={args.games}  "
            f"bot 用：--server http://127.0.0.1:{args.port} --token T --no-guide-check（不带 --game）")
    else:
        log(f"listening http://127.0.0.1:{args.port}  games={args.games}  --game {','.join(GAMES)}")
    time.sleep(args.start_delay)
    FAULT["t0"] = time.time()
    t0 = time.time()
    try:
        if TOUR is not None:
            TOUR.thread.start()
            TOUR.thread.join()
            time.sleep(3)   # 让 bot 读到 finished
        else:
            for g in GAMES.values():
                g.thread.start()
            while any(not g.finished for g in GAMES.values()):
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    log(f"全部结束 用时 {time.time() - t0:.0f}s  http={dict(COUNTS)}")
    tot = collections.Counter()
    for g in GAMES.values():
        tot.update(g.stats)
    log("engine stats=", dict(tot))
    ext = [g.scores[g.ext_seat] for g in GAMES.values()]
    log("外部 bot 各场得分=", ext, "合计", sum(ext))
    srv.shutdown()


if __name__ == "__main__":
    main()
