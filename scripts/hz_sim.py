#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hz_sim —— 杭州麻将（平台规则）本地 4 人对局模拟器。

规则（对齐门户接入指南 §1）：
  * 136 张，白板 = 财神（万能）；财神不能被吃/碰/杠，可打出。
  * 打出财神 → 抓打圈：其余三家该圈只能打刚摸的牌（可自摸胡 / 暗杠），且不能吃碰明杠。
  * 只能自摸；吃 ≤ 2 摊；碰/杠不限；碰（含明杠）窗口先于吃窗口。
  * 牌墙最后 20 张保留不摸，剩 ≤20 张时禁杠；摸完流局，流局连庄，胡牌者坐庄。
  * 番：平胡1 / 七对2 / 豪华七对 4·8·16；杠开、财飘每个动作 ×2（链断清零）；4 白板 ×2；爆头 ×2。
  * 计分：庄胡三闲各付 8×fan；闲胡庄付 8×fan、两闲各付 1×fan。

Agent 接口（与平台 /action 的 JSON 完全一致，便于直接对接）：
  act_draw(snap)     -> {"action":"hu"} | {"action":"discard","tile":t} | {"action":"gang","tile":t}
  act_response(snap) -> {"action":"peng"|"gang"|"chi","tile":t,"tiles":[a,b]} | {"action":"pass"}
snap 字段与服务端快照一致：seat/dealer/turn/phase/my_hand/drawn_tile/melds/discards/
  hand_counts/wall_remaining/last_discard/responding_seats/god{baotou,chain_count,catch_play}/scores/round_no
  额外：rules（YouCaiBiKao 等）、last_discard_seat。
"""
import random
from hz_core import (GOD, TILE_NAMES, TILE_INDEX, hand_to_counts, counts_to_hand,
                     is_win, is_baotou, calc_fan, is_suited, rank_of, suit_of)

DEFAULT_RULES = {
    "YouCaiBiKao": False,
    "wall_reserve": 20,     # 最后 20 张不摸
    "gang_min_wall": 21,    # 剩余 ≥21 才能杠
    "max_chi": 2,
    "rounds": 8,
    "base": 1,
}


class Player:
    __slots__ = ("hand", "melds", "discards", "chi_count", "chain", "piao", "catch")

    def __init__(self):
        self.hand = [0] * 34
        self.melds = []        # {"kind": chi|peng|gang_ming|gang_an|gang_bu, "tiles": [...]}
        self.discards = []
        self.chi_count = 0
        self.chain = 0         # 动作链（杠/飘）
        self.piao = 0          # 链内飘出的白板数
        self.catch = False     # 处于抓打圈（本回合只能打刚摸的）

    def n_melds(self):
        return len(self.melds)

    def hand_size(self):
        return sum(self.hand)


class RoundResult:
    __slots__ = ("winner", "fan", "detail", "scores", "dealer", "turns", "wall_left", "log")

    def __init__(self, winner, fan, detail, scores, dealer, turns, wall_left, log):
        self.winner, self.fan, self.detail, self.scores = winner, fan, detail, scores
        self.dealer, self.turns, self.wall_left, self.log = dealer, turns, wall_left, log


class Game:
    def __init__(self, agents, rules=None, seed=None, verbose=False):
        assert len(agents) == 4
        self.agents = agents
        self.rules = dict(DEFAULT_RULES)
        if rules:
            self.rules.update(rules)
        self.rng = random.Random(seed)
        self.verbose = verbose
        self.scores = [0, 0, 0, 0]
        self.round_no = 0
        self.dealer = 0
        self.results = []
        self.stats = {"hu": [0] * 4, "chi": [0] * 4, "peng": [0] * 4, "gang": [0] * 4,
                      "piao": [0] * 4, "draw_rounds": 0, "invalid": [0] * 4}

    # ---------- 工具 ----------
    def _snapshot(self, seat, phase, turn, responding=None):
        p = self.players[seat]
        n_melds = p.n_melds()
        # 爆头态：13 形手牌（去掉刚摸的）是否听任意牌
        if self.drawn_tile is not None and turn == seat and phase == "draw":
            h13 = list(p.hand)
            h13[TILE_INDEX[self.drawn_tile]] -= 1
            baotou = is_baotou(h13, n_melds)
        else:
            baotou = is_baotou(p.hand, n_melds) if p.hand_size() % 3 == 1 else False
        return {
            "seat": seat,
            "dealer": self.dealer,
            "turn": turn,
            "phase": phase,
            "my_hand": counts_to_hand(p.hand),
            "drawn_tile": self.drawn_tile if (turn == seat and phase == "draw") else None,
            "melds": [[dict(m) for m in q.melds] for q in self.players],
            "discards": [list(q.discards) for q in self.players],
            "hand_counts": [q.hand_size() for q in self.players],
            "wall_remaining": len(self.wall),
            "last_discard": self.last_discard,
            "last_discard_seat": self.last_discard_seat,
            "responding_seats": responding or [],
            "god": {"baotou": baotou, "chain_count": p.chain, "piao_count": p.piao, "catch_play": p.catch},
            "scores": list(self.scores),
            "round_no": self.round_no,
            "rules": dict(self.rules),
        }

    def _draw_allowed(self):
        return len(self.wall) > self.rules["wall_reserve"]

    def _gang_allowed(self):
        return len(self.wall) >= self.rules["gang_min_wall"]

    def _draw(self, seat):
        t = self.wall.pop(0)
        self.players[seat].hand[t] += 1
        self.drawn_tile = TILE_NAMES[t]
        return t

    def _draw_back(self, seat):
        t = self.wall.pop()
        self.players[seat].hand[t] += 1
        self.drawn_tile = TILE_NAMES[t]
        return t

    def _can_hu(self, seat):
        p = self.players[seat]
        # 只能自摸：必须是刚摸牌（含杠后补牌）；吃碰之后不能直接胡
        if self.drawn_tile is None:
            return False
        if not is_win(p.hand, p.n_melds()):
            return False
        if self.rules["YouCaiBiKao"] and p.hand[GOD] > 0 and p.chain == 0:
            return self._baotou_state[seat]
        return True

    def _log(self, *a):
        if self.verbose:
            print(*a)

    # ---------- 一局 ----------
    def play_round(self):
        self.round_no += 1
        R = self.rules
        self.players = [Player() for _ in range(4)]
        tiles = [i for i in range(34) for _ in range(4)]
        self.rng.shuffle(tiles)
        for s in range(4):
            for t in tiles[s * 13:(s + 1) * 13]:
                self.players[s].hand[t] += 1
        self.wall = tiles[52:]
        self.drawn_tile = None
        self.last_discard = None
        self.last_discard_seat = None
        self._baotou_state = [False] * 4
        log = []
        turns = 0
        for ag in self.agents:
            if hasattr(ag, "new_round"):
                ag.new_round()

        cur = self.dealer
        need_draw = True
        while True:
            p = self.players[cur]
            if need_draw:
                if not self._draw_allowed():
                    return self._finish_draw(turns, log)
                self._baotou_state[cur] = is_baotou(p.hand, p.n_melds())
                self._draw(cur)
                turns += 1
            # ---- 出牌阶段（可能多次：杠后补牌继续）----
            while True:
                snap = self._snapshot(cur, "draw", cur)
                act = self._safe(cur, "act_draw", snap) or {}
                a = act.get("action")
                if a == "hu":
                    if self._can_hu(cur):
                        return self._finish_hu(cur, turns, log)
                    self.stats["invalid"][cur] += 1
                    a = None
                if a == "gang":
                    tile = act.get("tile")
                    kind = self._gang_kind(cur, tile)
                    if kind and self._gang_allowed():
                        self._do_gang(cur, tile, kind)
                        log.append((cur, kind, tile))
                        self.stats["gang"][cur] += 1
                        p.catch = False  # 暗杠不受抓打圈限制；补牌后正常出
                        # 杠后补牌
                        self._baotou_state[cur] = is_baotou(p.hand, p.n_melds())
                        self._draw_back(cur)
                        continue
                    self.stats["invalid"][cur] += 1
                    a = None
                # 出牌
                tile = act.get("tile") if a == "discard" else None
                if tile is None or tile not in TILE_INDEX or p.hand[TILE_INDEX[tile]] <= 0 \
                        or (p.catch and self.drawn_tile is not None and tile != self.drawn_tile):
                    if a is not None:
                        self.stats["invalid"][cur] += 1
                    # 兜底：能胡自动胡，否则打刚摸的 / 最右一张
                    if self._can_hu(cur):
                        return self._finish_hu(cur, turns, log)
                    tile = self.drawn_tile if self.drawn_tile else self._rightmost(p)
                break
            # ---- 执行出牌 ----
            self._discard(cur, tile)
            log.append((cur, "discard", tile))
            was_catch = p.catch
            p.catch = False
            self.drawn_tile = None
            if tile == TILE_NAMES[GOD]:
                # 抓打圈：其余三家
                for s in range(4):
                    if s != cur:
                        self.players[s].catch = True
                cur = (cur + 1) % 4
                need_draw = True
                continue
            # ---- 响应窗口（抓打圈内的出牌不可被吃碰杠）----
            taker = None
            if not was_catch:
                taker = self._response_window(cur, tile, log)
            if taker is None:
                cur = (cur + 1) % 4
                need_draw = True
            else:
                seat, kind = taker
                cur = seat
                if kind == "gang_ming":
                    q = self.players[seat]
                    self._baotou_state[seat] = is_baotou(q.hand, q.n_melds())
                    self._draw_back(seat)
                    turns += 1
                    need_draw = False
                else:
                    self.drawn_tile = None
                    need_draw = False

    def _rightmost(self, p):
        for i in range(33, -1, -1):
            if p.hand[i] > 0:
                return TILE_NAMES[i]
        return None

    def _discard(self, seat, tile):
        p = self.players[seat]
        ti = TILE_INDEX[tile]
        p.hand[ti] -= 1
        p.discards.append(tile)
        self.last_discard = tile
        self.last_discard_seat = seat
        # 动作链：爆头态打白板 = 飘（链续）；否则链断
        if ti == GOD and is_baotou(p.hand, p.n_melds()):
            p.chain += 1
            p.piao += 1
            self.stats["piao"][seat] += 1
        else:
            p.chain = 0
            p.piao = 0

    def _gang_kind(self, seat, tile):
        if tile not in TILE_INDEX or tile == TILE_NAMES[GOD]:
            return None
        p = self.players[seat]
        ti = TILE_INDEX[tile]
        if p.hand[ti] == 4:
            return "gang_an"
        if p.hand[ti] >= 1:
            for m in p.melds:
                if m["kind"] == "peng" and m["tiles"][0] == tile:
                    return "gang_bu"
        return None

    def _do_gang(self, seat, tile, kind):
        p = self.players[seat]
        ti = TILE_INDEX[tile]
        if kind == "gang_an":
            p.hand[ti] -= 4
            p.melds.append({"kind": "gang_an", "tiles": [tile] * 4})
        elif kind == "gang_bu":
            p.hand[ti] -= 1
            for m in p.melds:
                if m["kind"] == "peng" and m["tiles"][0] == tile:
                    m["kind"] = "gang_bu"
                    m["tiles"] = [tile] * 4
                    break
        else:  # gang_ming
            p.hand[ti] -= 3
            p.melds.append({"kind": "gang_ming", "tiles": [tile] * 4})
        p.chain += 1

    def _response_window(self, discarder, tile, log):
        ti = TILE_INDEX[tile]
        # 碰/明杠窗口（三家）
        others = [(discarder + k) % 4 for k in (1, 2, 3)]
        for s in others:
            q = self.players[s]
            if q.catch or q.hand[ti] < 2:
                continue
            snap = self._snapshot(s, "response_peng", discarder, responding=[s])
            act = self._safe(s, "act_response", snap) or {}
            a = act.get("action")
            if a == "gang" and q.hand[ti] >= 3 and self._gang_allowed():
                self._do_gang(s, tile, "gang_ming")
                self.stats["gang"][s] += 1
                log.append((s, "gang_ming", tile))
                return (s, "gang_ming")
            if a in ("peng", "gang") and q.hand[ti] >= 2:
                q.hand[ti] -= 2
                q.melds.append({"kind": "peng", "tiles": [tile] * 3})
                q.chain = 0; q.piao = 0
                self.stats["peng"][s] += 1
                log.append((s, "peng", tile))
                return (s, "peng")
        # 吃窗口（下家）
        s = (discarder + 1) % 4
        q = self.players[s]
        if q.catch or not is_suited(ti) or q.chi_count >= self.rules["max_chi"]:
            return None
        combos = self._chi_combos(q, ti)
        if not combos:
            return None
        snap = self._snapshot(s, "response_chi", discarder, responding=[s])
        act = self._safe(s, "act_response", snap) or {}
        if act.get("action") == "chi":
            want = act.get("tiles")
            chosen = None
            for c in combos:
                if want and sorted(want) == sorted(c):
                    chosen = c
                    break
            if chosen is None:
                chosen = combos[0] if not want else None
            if chosen:
                for t in chosen:
                    q.hand[TILE_INDEX[t]] -= 1
                # 与服务端一致：被吃的那张放最后（服务端 chi 事件/快照 5202/5202 均为 tiles[-1]；brain 据此补回河里的重复计数）
                q.melds.append({"kind": "chi", "tiles": list(chosen) + [tile]})
                q.chi_count += 1
                q.chain = 0; q.piao = 0
                self.stats["chi"][s] += 1
                log.append((s, "chi", tile, chosen))
                return (s, "chi")
            self.stats["invalid"][s] += 1
        return None

    @staticmethod
    def _chi_combos(q, ti):
        r = rank_of(ti)
        base = ti - (r - 1)
        out = []
        for lo in (r - 2, r - 1, r):
            if lo < 1 or lo + 2 > 9:
                continue
            need = [base + lo - 1 + k for k in range(3) if base + lo - 1 + k != ti]
            if all(q.hand[x] >= 1 for x in need):
                out.append([TILE_NAMES[x] for x in need])
        return out

    def _safe(self, seat, method, snap):
        try:
            return getattr(self.agents[seat], method)(snap)
        except Exception as e:  # agent 崩溃视为超时兜底
            self.stats["invalid"][seat] += 1
            if self.verbose:
                import traceback; traceback.print_exc()
            return None

    # ---------- 结算 ----------
    def _finish_hu(self, seat, turns, log):
        p = self.players[seat]
        fan, detail = calc_fan(p.hand, p.n_melds(), self._baotou_state[seat], p.chain, p.piao)
        base = self.rules["base"]
        delta = [0, 0, 0, 0]
        if seat == self.dealer:
            for s in range(4):
                if s != seat:
                    delta[s] = -8 * fan * base
        else:
            for s in range(4):
                if s == seat:
                    continue
                delta[s] = -(8 if s == self.dealer else 1) * fan * base
        delta[seat] = -sum(delta)
        for s in range(4):
            self.scores[s] += delta[s]
        self.stats["hu"][seat] += 1
        res = RoundResult(seat, fan, detail, delta, self.dealer, turns, len(self.wall), log)
        self.results.append(res)
        self._log(f"R{self.round_no} 座{seat}{'(庄)' if seat == self.dealer else ''} 胡 fan={fan} {detail} 手={counts_to_hand(p.hand)} 副露={p.melds}")
        self.dealer = seat
        return res

    def _finish_draw(self, turns, log):
        self.stats["draw_rounds"] += 1
        res = RoundResult(-1, 0, [], [0, 0, 0, 0], self.dealer, turns, len(self.wall), log)
        self.results.append(res)
        self._log(f"R{self.round_no} 流局")
        return res

    def play(self):
        for _ in range(self.rules["rounds"]):
            self.play_round()
        return self.scores


# ---------------- 简单基线 agent ----------------
class GreedyAgent:
    """基线：能胡就胡；出牌选「打后向听最小、进张（未见张数）最多」；碰/吃只在向听降低时做；不杠。"""
    name = "greedy"

    def __init__(self):
        pass

    def act_draw(self, snap):
        from hz_core import shanten, useful_tiles
        hand = snap["my_hand"]
        c = hand_to_counts(hand)
        nm = len(snap["melds"][snap["seat"]])
        if snap.get("drawn_tile") and is_win(c, nm):
            return {"action": "hu"}
        if snap["god"]["catch_play"] and snap.get("drawn_tile"):
            return {"action": "discard", "tile": snap["drawn_tile"]}
        unseen = _unseen(snap)
        best, best_key = None, None
        for t in set(hand):
            ti = TILE_INDEX[t]
            if ti == GOD:
                continue
            c[ti] -= 1
            s = shanten(c, nm)
            u = sum(unseen[x] for x in useful_tiles(c, nm))
            c[ti] += 1
            key = (-s, u)
            if best_key is None or key > best_key:
                best_key, best = key, t
        return {"action": "discard", "tile": best or hand[-1]}

    def act_response(self, snap):
        from hz_core import shanten
        hand = snap["my_hand"]
        c = hand_to_counts(hand)
        nm = len(snap["melds"][snap["seat"]])
        last = snap["last_discard"]
        ti = TILE_INDEX[last]
        cur = shanten(c, nm)
        if snap["phase"] == "response_peng" and c[ti] >= 2:
            c[ti] -= 2
            s = shanten(c, nm + 1)
            c[ti] += 2
            if s < cur:
                return {"action": "peng", "tile": last}
            return {"action": "pass"}
        if snap["phase"] == "response_chi":
            r = rank_of(ti); base = ti - (r - 1)
            for lo in (r - 2, r - 1, r):
                if lo < 1 or lo + 2 > 9:
                    continue
                need = [base + lo - 1 + k for k in range(3) if base + lo - 1 + k != ti]
                if all(c[x] >= 1 for x in need):
                    for x in need: c[x] -= 1
                    s = shanten(c, nm + 1)
                    for x in need: c[x] += 1
                    if s < cur:
                        return {"action": "chi", "tile": last, "tiles": [TILE_NAMES[x] for x in need]}
        return {"action": "pass"}


def _unseen(snap):
    """未见牌张数（4 − 自己手牌 − 所有河牌 − 所有副露）。"""
    u = [4] * 34
    for t in snap["my_hand"]:
        u[TILE_INDEX[t]] -= 1
    for d in snap["discards"]:
        for t in d:
            u[TILE_INDEX[t]] -= 1
    for ms in snap["melds"]:
        for m in ms:
            for t in m["tiles"]:
                u[TILE_INDEX[t]] -= 1
    # 自己副露里的牌已算在 melds 里；自己的手牌单独减了
    return [max(0, x) for x in u]


if __name__ == "__main__":
    import time, sys, io
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    t0 = time.time()
    g = Game([GreedyAgent() for _ in range(4)], seed=3, verbose=True)
    g.play()
    print("scores", g.scores, "stats", g.stats, f"{time.time()-t0:.1f}s")
