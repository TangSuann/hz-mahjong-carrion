#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hz_brain —— 杭州麻将期望值搜索决策引擎（纯自摸规则 → 只比「速度 × 番 × 庄闲倍率」）。

估值模型（单位：期望得分）
  把「自己每次摸牌」当作一步，向听 s 的手牌 A 张进张（未见张数）/ U 张未见牌：
     每步进一向听概率 p = A/U；两步之间他家先胡的存活率 γ（随他家副露数、局面进程调整）；
     剩余可摸次数 n 截断。
     V_s = γ·(Σ_x u_x·V_{s-1}(x 后最佳打法))/U · (1-((1-p)γ)^n) / (1-(1-p)γ)
  听牌 V_0：进张 = 胡张，V_{-1}(x) = fan(h+x) × 庄闲倍率（爆头/七对/豪华/4白板/链全部精确计番）。
  向听 ≤ lookahead 的层精确展开（摸牌 → 最佳打牌），更深层用「实际进张数 + 典型进张常数」的解析近似。

决策：
  出牌：对每个候选（非财神）比较 V(13 张)。
  吃/碰/明杠：V(副露后最佳 13 张, n_melds+1) + 节奏加成 vs V(不动)。
  暗杠/补杠：补牌一次的期望（含杠开 ×2）vs 直接出牌。
  胡/财飘：爆头态摸白板时比较 立刻胡 vs 飘（fan×2、γ、剩余牌）。
  抓打圈：只能打刚摸的牌（能胡则胡）。
"""
import time
from functools import lru_cache
from hz_core import (GOD, TILE_NAMES, TILE_INDEX, hand_to_counts, shanten, shanten_normal,
                     shanten_qidui, is_win, is_baotou, calc_fan, is_suited, rank_of)

DEFAULT_PARAMS = {
    "lookahead": 1,        # 精确展开到的向听层（1：一向听精确；2：二向听也精确，慢 ~10 倍）
                           # 修复版 2000 场：lookahead=2 → −0.12±1.24（噪声内、慢 3 倍），不值得。
    "hazard_base": 0.035,  # 每次自己摸牌之间被他家胡掉的基础概率
    "hazard_meld": 0.012,  # 他家每个副露增加的风险
    "hazard_turn": 0.0015, # 每过一巡增加的风险（他家手牌逐渐成型）
    # 经验风险曲线：第 t 巡（自己第 t 次摸牌之前）本局被他家胡掉的条件概率（自对弈实测，
    # 单家视角 ≈ 局结束风险 × 3/4）；设为 None 则退回常数 gamma。
    # v2 默认 = 真实排位拟合（原 @real_hazard 预设，773 局）：基线低（真人前期慢）、他家每副露 ×1.5。
    # 排位 A/B/C 六房：B/C(含此曲线) 均番 1.32 vs A 1.20；自对弈 4000 场 vs 旧曲线 +0.92±0.79（拒胡规则加入后才转正）。
    # 旧自对弈曲线保留为 @selfplay_hazard。
    "hazard_curve": [0.0, 0.003, 0.005, 0.022, 0.038, 0.047, 0.057, 0.067, 0.075, 0.074, 0.106,
                     0.101, 0.098, 0.107, 0.087, 0.122, 0.078, 0.078],
    "hazard_meld_scale": 0.5,   # 他家每个副露把风险曲线放大 50%
    "hazard_scale": 1.0,        # 风险曲线整体倍率（>1 = 假设桌面更快/更强，更看重速度；<1 = 更博番）
    # 向听 s 的典型进张数（近似层用；实测中位数），按副露数分别标定
    "typ_accept": [16.0, 21.0, 30.0, 45.0, 70.0, 85.0, 95.0],
    "typ_accept_by_melds": [[16.0, 21.0, 30.0, 45.0, 70.0, 85.0, 95.0],
                            [15.0, 20.0, 31.0, 62.0, 82.0, 92.0, 95.0],
                            [12.0, 20.0, 45.0, 74.0, 88.0, 92.0, 95.0],
                            [11.0, 24.0, 62.0, 80.0, 90.0, 92.0, 95.0],
                            [10.0, 24.0, 62.0, 80.0, 90.0, 92.0, 95.0]],
    "meld_lookahead": 1,   # 响应窗口（吃碰杠）精确展开层数。2 层 p99 214ms/最大 0.9s，1 秒窗口内太险；
                           # 1 层 p99 19ms，2000 场实测 +1.65±1.01 vs greedy（2 层 +0.97±0.81，噪声内）
                           # 修复版 2000 场：2 层 vs 1 层 +0.32±1.26（噪声内），且 1s 碰/吃窗口内 p99 214ms 太险 → 保持 1。
    "approx_scale": 0.78,  # 近似层相对精确层的系统性高估校正（实测中位数比 ≈0.77）
    "tempo_bonus": 0.0,    # 吃碰副露的节奏加成（自对弈实测 0 最优：先手价值已由 n_pass/n_after 精确体现）
    "meld_penalty": 0.0,   # 副露额外折扣（比例）
    "meld_margin": 0.0,    # 明杠要求 EV(杠) > EV(不动)×(1+margin)
    # v3 默认（2026-09-13）= 原 @meld_loose：向听下降的吃碰 EV 门槛 0，且允许向听不降但 EV 高 15% 的副露。
    # 证据：配对自对弈 33 候选中唯一稳健正向——2v2 +0.66±0.55（两次同种子重跑一致）、1v3 +0.75±0.61 / +1.08±0.70，
    # 增益 = 提速（胡率 +0.4pp、庄闲皆正、≥4番/百局 1.09→0.94、最差 5% 局不变）；线上 A/B 14 轮 41 房 B +0.436/局（t 1.37）、
    # 四条安全线（超时率/向听更差/分歧/一致率）全过、被胡率 −1.9pp、尾部不变差。只放宽庄局（@meld_loose_dealer）只拿到 2/3 增益。
    # v2 默认（高手口径：只在向听下降且 EV 提升 ≥10% 时吃碰）完整保留为 @expert_meld（审计 v2 期房间传 `--params @expert_meld`）。
    # 更早的贪心口径为 @greedy_meld。
    "meld_margin_drop": 0.0,   # 向听下降的吃碰：EV 门槛（-1 = 只要向听下降就做，贪心同款；v2 默认 0.1）
    "meld_margin_same": 0.15,  # 向听不降的吃碰（仅 meld_need_shanten=0 时允许）：EV 门槛（0.05～0.3 在噪声内等价）
    "meld_need_shanten": 0,  # 1：吃碰必须使向听数下降（贪心同款硬约束，v2 默认）；0：允许向听不降但 EV 更高的副露
    "meld_need_shanten_from": 3,  # 当前向听 ≥ 此值时仍强制要求向听下降（远处近似层不可靠）
    "piao_margin": 0.15,   # 财飘要求 EV(飘) ≥ EV(胡)×(1+margin)
    "piao_any_draw": 1,    # 1：爆头态摸到非白也考虑飘 + 吃/碰后（不能胡）出牌点也考虑飘（手 ≥2 白、打白后仍爆头）；
                           # 0：只在摸到白时考虑（旧行为）。自对弈 2v2 8000 场 ×2：+0.39±0.64 / +0.31±0.64；1v3 8000 场 +0.05±0.90（中性）
    "gang_margin": 0.0,
    "qidui_bonus": 1.0,
    # 出牌近似平手（EV 差 < tie_eps）时按高手习惯打孤张字牌/幺九。真实排位 10 位高手 5.7k 手出牌：
    # 一致率 73% → 82%；自对弈 4000 场 +0.3～1.1/场（不退化）。0 关闭。
    "tie_eps": 0.02,
    # 拒胡转爆头听的门槛：None 关闭；0.1 表示爆头听估值须高于立刻胡 10%。见 act_draw。
    # 来源：R1 排位 4 房，对手爆头率 25–45%、我方 11%；高手拒胡转爆头占其可胡局 8–10%。
    # 自对弈 4000 场 2v2：margin 0 → +0.98±0.78/场，0.1 → +0.79±0.76；vs laggy 1v3 +2.8/场。取 0.1 留安全余量。
    # 自对弈 4000 场：margin 0.05 → −0.07±0.88、0 → −0.01±0.88、−0.1 → +0.01±0.88，全部噪声内；线上修复版拒胡率 15.2%
    # 已与高手（15–20%）对齐，故保持 0.1。
    "hu_decline_margin": 0.1,
    # 输分侧（庄位效率实验）：每次胡额外加 win_bonus = loss_fan×(8 庄 / 10/3 闲)，让庄家更看重速度而非番。
    # 自对弈 4000 场：0.5 → −0.20±0.87、1.0 → −0.27±0.86、2.0 → −0.47±0.86，单调变差 → 保持 0（关闭）。
    # 结论：庄位「速度优先」不成立；风险曲线已经把被胡损失算进 gamma，再加一层是重复计价。
    "loss_fan": 0.0,
    # 连庄价值：胡牌者下局坐庄，庄局净分比闲局高 ≈1.2～1.9（自对弈/真实）。每次胡固定加此分，
    # 使「速胡」相对「博番」更值一点；庄闲同加（庄胡也连庄）。0 关闭。
    # 自对弈 4000 场（模拟器本身已按真实规则连庄：胡者坐庄/流局连庄，排位 1971 局实证一致）：
    # 1.5 → −0.10±0.87、3.0 → −0.09±0.88、6.0 → −0.13±0.87，全部噪声内 → 保持 0。
    "dealer_value": 1.67,  # 连庄价值：胡牌者下局坐庄。内网实测坐庄净分差 d=+1.25/局（2740 局），理论 4d/3≈1.67。
                           # T2 层化（21 房 v3）：庄位拒胡净收益 +25.0 vs 闲位 +7.9，庄位并未被压错；
                           # 自对弈 8000 场 +0.10±0.65（中性）；排位 1134 个可胡点只翻 2 个（都是边际）；audit 100%。
    # 喂牌惩罚（实验）：出牌候选（同向听）按「被碰 / 被下家吃」的经验概率扣分，单位 = 每次被吃碰的等价损失（分）。
    # 概率由真实排位 13.6 万次出牌标定：P(碰|未见 u)=0/0.16%/1.95%/9.97%（u=0..3）；
    # P(下家吃|搭子分 cs=Σ unseen[a]·unseen[b]/16)≈4.7%+5.2%·cs（封顶 20%）。0 关闭。
    # 结果（自对弈 2v2 8000 场 vs 默认）：2 → −0.88±0.66、5 → −1.63±0.66，单调变差；排位复盘 3690 个出牌点
    # 改牌 17%/20%，分歧点喂牌概率 20%→11%，但换来的牌效损失更大。纯自摸规则下「防喂」不成立 → 保持 0（关闭）。
    "feed_penalty": 0.0,
    # 剩余摸牌次数的量化平滑（2026-09-13 线上 4 处出牌不符全部源于此）：
    #   0：各层剩余次数 int(round(n−1/p))、巡目 int(round(t))——同一手牌两张候选的进张只差 1%，下一层却可能
    #      差 1 次摸牌（EV 差 ~20%），近似平手时排序随牌墙余数跳变（wall±1 即翻转）；
    #   1：各层用分数次数 n−1/p（_geom 支持分数 n：末项按比例计入）、巡目分数插值风险表；
    #      精确层的 n_next 取 1/4 步长（保持 cache 命中）。默认层数 n_draws 仍为整数 drawable//4。
    "n_smooth": 0,
    # 1：顶层 n_draws 用 drawable/4（分数）而非 drawable//4——牌墙每少 1 张估值平滑变化（吃碰杠本就打乱轮转，
    #    整数并不更"准"）。仅在 n_smooth=1 时有意义。
    "n_frac": 0,
    # 1：未见牌计数按服务端语义修正——被吃/碰/明杠的那张牌仍留在出牌者河里（1138 个线上快照实证，可见张数可达 5），
    #    河牌 + 副露会把它扣两次；补回一张。0 = 旧口径（重复扣减；14% 的快照里有牌被误判为已绝）。
    "river_keeps_claimed": 1,
    # v4 默认（2026-09-14）：听牌估值（value0）里，摸到后可「拒胡转爆头听」的胡牌张按番 ×(1+此值) 计（0 关闭 = v3）。
    # 依据：排位 2.3k 个拒胡转爆头点，E[终番]/当时番 ≈ 1.8（我方 1.82～1.91，高手 1.5～1.95）；
    # 旧 value0 只算立刻胡的番，低估「可转爆头」听型，导致换听/选听时不偏向可转形（高手首听可转比例 0.19 vs 我 0.17）。
    # 配对自对弈：种子 50000/6000 场 2v2 +0.56±0.38、1v3 +1.03±0.57；种子 91000/8000 场 2v2 +1.06±0.33、1v3 +1.26±0.51
    # （0.65/0.8 同量级）。增益主要在庄局（+0.15～+0.42/局），≥4番/百局 0.86→1.02，最差 5% 局不变（-21.2→-21.3）。
    # 排位回放 3686 个我方摸牌点只改 0.5%（持白听牌点 1.6%）；线上快照延迟 p99 8ms/13ms。
    "conv_fan_bonus": 0.5,
    # 近似层（向听≥1）持白的番系数斜率：god_fan = 1 + slope×白数。真实排位 s=1/2 时胡均番 g0 1.14、g1 1.31、g2 1.59
    # → 相对斜率 ≈0.15～0.2；0.35 偏高（但 2v2 6000 场 0.25 → +0.08±0.08，0.5 → 0.00±0.11，都很小）。
    "god_fan_slope": 0.35,
    # 1：链>0（杠后补牌 / 飘后）的出牌点，普通出牌与「不杠」基线按 chain=0 估值（打普通牌链即断——排位 492 局实证
    #    杠后打普通牌再自摸无一计杠开）；旧口径把 ×2^链 也算进拒胡/不杠的基线，71 个杠后补牌可胡点里 16 个误拒胡。
    #    v33 后杠后补牌点首次进入 act_draw（此前服务端直接结算），该修正才真正落到线上。
    "chain_break_on_discard": 1,
    # 明杠估值的时间预算（秒）。测试赛 9/17：明杠精确估值（34 张补牌 × 最佳打法）在 10 场并发 + 冷缓存下
    # p90 984ms / 最大 1672ms（本地冷回放最大 291ms，线上慢 3～6 倍），3 次错过 1s 碰窗口（杠→碰兜底也 409）。
    # 预算内先算「进张」补牌（精确），超预算后其余补牌按「补牌不进张 = 杠后 13 张原估值」合并计（近似，见 act_response）。
    # 32 个线上明杠点回放：近似 EV 与精确 EV 同向 32/32（决策一致），冷缓存 p50 30→? 见 test_gang_budget.py。
    "gang_budget": 0.25,
}

_P_PENG = (0.0, 0.0016, 0.0195, 0.0997)

# 牌种行号（wall_weight 表）：0 字、1 幺九、2 二八、3 中张、4 白
_KIND_ROW = [4 if k == GOD else (0 if k >= 27 else (1 if k % 9 in (0, 8) else (2 if k % 9 in (1, 7) else 3))) for k in range(34)]


def feed_table(unseen):
    """每张牌被吃碰的经验概率（出牌者视角未见张数）。"""
    f = [0.0] * 34
    for d in range(33):
        p = _P_PENG[min(3, unseen[d])]
        if d < 27:
            lo = d - d % 9; r = d % 9
            cs = 0.0
            for a, b in ((r - 2, r - 1), (r - 1, r + 1), (r + 1, r + 2)):
                if 0 <= a < 9 and 0 <= b < 9:
                    cs += unseen[lo + a] * unseen[lo + b]
            p += min(0.20, 0.047 + 0.052 * cs / 16.0)
        f[d] = p
    return f

# 命名预设（hz_eval --a brain:@名字 / hz_bot --params @名字）
PRESETS = {
    # 真实排位 773 局（10 房、26 名玩家）拟合的风险曲线：h(t,m) = curve[t]·(1 + 0.5·m)，m = 他家副露总数。
    # 与自对弈曲线相比：基线更低（真人前期慢），但每个副露的放大系数 0.5（自对弈 0.10）——真人副露≈快听。
    # 自 v2 起已是默认值；预设保留以兼容 `--params @real_hazard`（等价于默认）。
    "real_hazard": {
        "hazard_curve": [0.0, 0.003, 0.005, 0.022, 0.038, 0.047, 0.057, 0.067, 0.075, 0.074, 0.106,
                         0.101, 0.098, 0.107, 0.087, 0.122, 0.078, 0.078],
        "hazard_meld_scale": 0.5,
    },
    # 2026-09-12 用冻结引擎期 48 房（3833 席局，只用我方席位视角）重拟合：基线曲线整体比 773 局版低 10–20%，
    # 副露放大系数 0.4（原 0.5）。候选，未定为默认——先过 hz_paired 配对自对弈。
    "fit48": {
        "hazard_curve": [0.0, 0.003, 0.005, 0.02, 0.033, 0.043, 0.062, 0.074, 0.08, 0.09, 0.104,
                         0.099, 0.099, 0.117, 0.093, 0.096, 0.09, 0.09],
        "hazard_meld_scale": 0.4,
    },
    # 2026-09-13 用全部 228 房排位（我方席位视角 17,819 席局，fit48 的 4.6 倍样本）重拟合 h(t,m)=curve[t]·(1+k·m)：
    # k=0.4；曲线中段比默认低 5–15%、末段（t≥14）低 10–25%（默认曲线 773 局末段样本极少、锯齿明显）。
    # 加「单家最大副露数」项 j·x 只把 chi2 从 435 降到 426（j=0.1）→ 线性副露总数模型已足够，不加参数。
    # r11 配对 6000 场 vs v2 默认：2v2 +0.31±0.23 ✅、1v3 +0.13±0.33 —；叠在 @meld_loose 上：2v2 +0.75 vs +0.66、
    # 1v3 +1.04 vs +1.08 → 叠加增量 ≈0（±0.1）。单独用有微弱正向，与 loose 同源（都是"更看重速度"）。不改默认。
    "fit228": {
        "hazard_curve": [0.0, 0.003, 0.007, 0.018, 0.033, 0.046, 0.061, 0.072, 0.077, 0.088, 0.093,
                         0.098, 0.094, 0.095, 0.1, 0.098, 0.093, 0.093],
        "hazard_meld_scale": 0.4,
    },
    # 2026-09-12 配对自对弈（6000 场×3 场型）最强候选：放宽副露——向听下降的吃碰 EV 门槛 0（原 0.1），
    # 且允许向听不降但 EV 高 15% 的副露。vs v2 默认 +0.66±0.55、vs v2 +0.76±0.53（均 ✅），≥4番/百局 1.09→0.94。
    # 与线上观察一致：Astra-0 碰 0.61/局 vs 我方 0.37，爆头占胡 34% vs 25%。
    # 2026-09-13 起为 v3 默认；预设保留以兼容 `HZ_AB="@meld_loose"` / 审计传参（等价于默认）。
    "meld_loose": {
        "meld_margin_drop": 0.0,
        "meld_need_shanten": 0,
        "meld_margin_same": 0.15,
    },
    # 2026-09-13 r8 庄/闲拆分（2v2）：@meld_loose 的增益看似全在庄局 → 只在庄局放宽副露、闲局用 v2 口径。
    # r9/r10 结论：2v2 的"闲局 Δ0"是队友混杂；1v3（solo）配对 8000 场 vs v2：@meld_loose +0.75±0.61（庄 +0.13/闲 +0.08 每局）、
    # 本预设 +0.49±0.43 —— 只拿到 2/3 增益，闲局放宽同样有正收益。不推荐；保留作记录/对照。
    # （v3 默认已是 loose；本预设 = 闲局退回 v2 口径 + 庄局 loose，语义不变。）
    "meld_loose_dealer": {
        "meld_margin_drop": 0.1,
        "meld_need_shanten": 1,
        "meld_margin_same": 0.1,
        "meld_margin_drop_dealer": 0.0,
        "meld_need_shanten_dealer": 0,
        "meld_margin_same_dealer": 0.15,
    },
    # 反向：庄局比 v2 更激进、闲局比 v2 更保守（探索方向，仅用于配对扫描；2v2 −0.19±0.45）
    "meld_dealer_only_tight": {
        "meld_need_shanten": 1,
        "meld_margin_same": 0.1,
        "meld_margin_drop_dealer": -1.0,
        "meld_need_shanten_dealer": 0,
        "meld_margin_same_dealer": 0.1,
        "meld_margin_drop": 0.2,
    },
    # v2 默认（2026-09-11～09-13）= 高手副露口径：向听不降不吃碰（高手在向听不降时副露率 0.27，bot 0.36；
    # 向听下降时高手 0.75，bot 1.00），且向听下降的吃碰要求 EV 提升 ≥10%。
    # 排位 C 组 160 局 +2.24/局（A 组 +0.49，B 组 +0.14）、想副露 66/房 vs 102。
    # 审计/回放 v2 期（A 臂）房间时传 `--params @expert_meld`。
    "expert_meld": {
        "meld_need_shanten": 1,
        "meld_margin_drop": 0.1,
        "meld_margin_same": 0.1,
    },
    # v3 默认（2026-09-13～09-14）= loose 副露 + 旧听牌估值（不计可转爆头加成）。线上 A/B 基线臂传 `@v3`。
    # 注意：`@v3` 保留链口径修正（chain_break_on_discard=1）——那是 bug 修复不是策略，A/B 两臂都该带。
    # 审计 v3 期（9/13 18:xx～9/14 15:37，rev a8a17b6 之前）的房间要逐位对齐时传 `@pre_v4`。
    "v3": {
        "conv_fan_bonus": 0.0,
    },
    # = v3 期线上引擎的完整口径（旧听牌估值 + 旧链口径）。仅用于 hz_audit 复跑 v3 期房间；不作 A/B 臂。
    "pre_v4": {
        "conv_fan_bonus": 0.0,
        "chain_break_on_discard": 0,
    },
    # v1 默认（R1 基线 + A 组）：自对弈风险曲线 + 贪心副露。用于回归对照。
    "selfplay_hazard": {
        "hazard_curve": [0.0, 0.001, 0.005, 0.025, 0.06, 0.075, 0.125, 0.175, 0.20, 0.25, 0.29, 0.31,
                         0.33, 0.35, 0.38, 0.40, 0.40, 0.40],
        "hazard_meld_scale": 0.10,
    },
    "greedy_meld": {
        "meld_need_shanten": 0,
        "meld_margin_drop": -1.0,
        "meld_margin_same": 0.1,
    },
    # = R1/A 组当时的完整参数（v1）：用于 hz_audit 复跑 R1/A 组房间时传 `--params @v1`
    "v1": {
        "hazard_curve": [0.0, 0.001, 0.005, 0.025, 0.06, 0.075, 0.125, 0.175, 0.20, 0.25, 0.29, 0.31,
                         0.33, 0.35, 0.38, 0.40, 0.40, 0.40],
        "hazard_meld_scale": 0.10,
        "meld_need_shanten": 0,
        "meld_margin_drop": -1.0,
        "meld_margin_same": 0.1,
    },
    # 牌墙密度修正（hz_wall_density.py，真实排位 578,889 次摸牌）：观测/期望 比值表 [牌种][他家河里张数 d]。
    # 字牌 d=0 1.09→d=3 1.58、幺九 0.98→1.48、中张 0.92→1.12、白 0.89。自对弈里同样复现（字 1.10/1.39/1.46，中张 0.92）。
    "wall_real": {
        "wall_weight": [[1.092, 1.364, 1.490, 1.584],
                        [0.979, 1.162, 1.290, 1.476],
                        [0.942, 1.068, 1.149, 1.147],
                        [0.920, 0.989, 1.046, 1.121],
                        [0.886, 0.868, 0.838, 0.838]],
    },
}


def parse_params(spec):
    """解析参数覆盖串：`@预设,k=v,k=[1,2,3]`。逗号只在方括号/花括号外切分；值按 JSON 解析，失败则保留原串。
    hz_bot / hz_audit / hz_eval 共用，保证同一串在三处得到同一组参数。"""
    import json as _json
    params = {}
    if not spec:
        return params
    items, buf, depth = [], "", 0
    for ch in spec:
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            items.append(buf); buf = ""
        else:
            buf += ch
    items.append(buf)
    for item in items:
        item = item.strip()
        if not item:
            continue
        if item.startswith("@"):
            params.update(PRESETS[item[1:]])
            continue
        k, v = item.split("=", 1)
        try:
            v = _json.loads(v)
        except Exception:
            pass
        params[k.strip()] = v
    return params


class Ctx:
    __slots__ = ("unseen", "U", "gamma", "n_draws", "drawable", "mult", "chain", "params", "cache1", "cache0",
                 "surv", "turn", "bikao", "bikao_inner", "win_bonus", "piao", "feed")

    def __init__(self, unseen, gamma, n_draws, mult, chain, params, drawable=None, surv=None, turn=0, bikao=False, piao=0):
        self.unseen = unseen
        self.U = max(1, sum(unseen))
        self.gamma = gamma
        self.n_draws = max(1, n_draws)
        self.drawable = drawable if drawable is not None else n_draws * 4
        self.surv = surv          # 累计存活表（按巡），None 则用常数 gamma
        self.turn = turn          # 当前巡目（自己已摸牌次数）
        self.mult = mult
        self.chain = chain
        self.params = params
        self.bikao = bikao        # 有财必拷响：持财神且 chain==0 时只有爆头听才能胡
        self.bikao_inner = False  # 拷响一步展望的内层标记（防指数递归）
        self.win_bonus = 0.0      # 每次胡的等价加分（= 避免的期望损失），见 loss_fan
        self.piao = piao          # 链内已飘出的白板数（4 白板判定 = 手留白 + 链内飘出，官方 fan-calc 口径）
        self.feed = None          # 喂牌惩罚表（feed_penalty>0 时由 make_ctx 填；仅顶层出牌比较用，不进递归估值）
        self.cache1 = {}
        self.cache0 = {}


# ---------- 快速工具 ----------

def _relevant_tiles(counts):
    """可能改变向听的摸牌候选（同花色 ±2 邻近、手中已有的字牌、财神）。"""
    rel = [False] * 34
    for i in range(27):
        if counts[i]:
            lo = i - (i % 9)
            for j in range(max(lo, i - 2), min(lo + 9, i + 3)):
                rel[j] = True
    for i in range(27, 33):
        if counts[i]:
            rel[i] = True
    rel[GOD] = True
    return [i for i in range(34) if rel[i]]


@lru_cache(maxsize=1 << 16)
def _win_tiles_key(key, nm):
    counts = list(key)
    out = []
    for x in _relevant_tiles(counts):
        counts[x] += 1
        if is_win(counts, nm):
            out.append(x)
        counts[x] -= 1
    return tuple(out)


@lru_cache(maxsize=1 << 16)
def _useful_key(key, nm, base):
    counts = list(key)
    out = []
    for x in _relevant_tiles(counts):
        counts[x] += 1
        if shanten(counts, nm) < base:
            out.append(x)
        counts[x] -= 1
    return tuple(out)


def _fan_of(counts14, nm, baotou, chain, piao=0):
    fan, _ = calc_fan(counts14, nm, baotou, chain, piao)
    return fan


@lru_cache(maxsize=1 << 16)
def _can_convert_key(key14, nm):
    """14 张可胡形（持财神、非爆头）：是否存在一张非财神打出后成爆头听（即「拒胡转爆头」可行）。"""
    c = list(key14)
    for z in range(33):
        if c[z]:
            c[z] -= 1
            ok = is_baotou(c, nm)
            c[z] += 1
            if ok:
                return True
    return False


# ---------- 估值 ----------

def _surv_at(surv, t):
    """累计存活表在（可能是分数的）巡目 t 处的值：整数巡直接取表，分数巡线性插值（表尾取常数）。"""
    i = int(t)
    if i >= len(surv) - 1:
        return surv[-1]
    f = t - i
    if f <= 0.0:
        return surv[i]
    return surv[i] + (surv[i + 1] - surv[i]) * f


def _geom(p, gamma, n, t0=0, surv=None):
    """Σ_{k=1..n} S(t0,k) (1-p)^(k-1)，S = 活到第 k 次摸牌的概率。
    surv 为 None 时 S = γ^k；否则 S = Π_{j=t0+1..t0+k} (1 - hazard[j])（surv 是累计存活表）。
    n / t0 允许为分数（n_smooth）：末项按小数部分比例计入、巡目在存活表上线性插值；整数入参与旧实现逐位一致。"""
    if surv is None:
        q = (1 - p) * gamma
        if q >= 0.999999:
            return gamma * n
        return gamma * (1 - q ** n) / (1 - q)
    total = 0.0
    q = 1.0
    ni = int(n)
    frac = n - ni
    if t0 == int(t0):
        t0 = int(t0)
        L = len(surv) - 1
        base = surv[min(t0, L)]
        if base <= 0:
            return 0.0
        for k in range(1, ni + 1):
            total += surv[min(t0 + k, L)] / base * q
            q *= (1 - p)
            if q < 1e-6:
                return total
        if frac > 0:
            total += frac * surv[min(t0 + ni + 1, L)] / base * q
        return total
    base = _surv_at(surv, t0)
    if base <= 0:
        return 0.0
    for k in range(1, ni + 1):
        total += _surv_at(surv, t0 + k) / base * q
        q *= (1 - p)
        if q < 1e-6:
            return total
    if frac > 0:
        total += frac * _surv_at(surv, t0 + ni + 1) / base * q
    return total


def _bikao_step_value(counts13, nm, ctx, n):
    """有财必拷响下、持财神的非爆头听：本次不能胡，摸一张后重新选打（可能转成爆头听或换听）。
    相关牌逐一展开；非相关牌视为「打掉它、手不变」（同手 n-1 递归）。"""
    if n <= 1 or ctx.bikao_inner:
        return 0.0          # 内层：一步展望到此为止（再往下的非爆头持财听按 0 计，保守）
    inner = Ctx(ctx.unseen, ctx.gamma, ctx.n_draws, ctx.mult, ctx.chain, ctx.params, ctx.drawable, ctx.surv, ctx.turn, True, ctx.piao)
    inner.bikao_inner = True
    inner.win_bonus = ctx.win_bonus
    c = list(counts13)
    total = 0.0
    A = 0
    for x in _relevant_tiles(counts13):
        u = ctx.unseen[x]
        if u <= 0:
            continue
        c[x] += 1
        v = best_discard_value(c, nm, inner, None, n - 1)[1]
        c[x] -= 1
        total += u * v
        A += u
    rest = ctx.U - A
    if rest > 0:
        # 非相关牌：打掉它、手不变，下一巡再看（同手递减 n）——用外层 ctx 保留一步展望
        total += rest * value0(counts13, nm, ctx, n - 1)
    t = ctx.turn + (ctx.n_draws - n)
    s = 1.0
    if ctx.surv is not None:
        a = ctx.surv[min(t, len(ctx.surv) - 1)]
        s = ctx.surv[min(t + 1, len(ctx.surv) - 1)] / a if a > 0 else 0.0
    else:
        s = ctx.gamma
    return s * total / ctx.U


def value0(counts13, nm, ctx, n=None):
    """听牌估值：期望得分。n = 剩余可摸次数（默认 ctx.n_draws）。"""
    if n is None:
        n = ctx.n_draws
    key = (tuple(counts13), nm, n)
    v = ctx.cache0.get(key)
    if v is not None:
        return v
    wins = _win_tiles_key(key[0], nm)
    W = 0
    F = 0.0
    if wins:
        baotou = counts13[GOD] > 0 and len(wins) == len(_relevant_tiles(counts13))
        # 爆头判定：含财神且所有牌皆胡（relevant 之外的牌加入后必然也胡：财神顶替）
        if baotou:
            baotou = is_baotou(counts13, nm)
        if ctx.bikao and counts13[GOD] > 0 and ctx.chain == 0 and not baotou:
            # 有财必拷响：持财神的非爆头听不能平胡（杠开 chain>0 除外）→ 按「再进一张转爆头听 / 打掉财神转普通听」估值
            v = _bikao_step_value(counts13, nm, ctx, n)
            ctx.cache0[key] = v
            return v
        c = list(counts13)
        cb = ctx.params.get("conv_fan_bonus") or 0.0
        conv = cb > 0 and not baotou and counts13[GOD] > 0 and ctx.chain == 0 and n >= 3
        for x in wins:
            u = ctx.unseen[x]
            if u <= 0:
                continue
            c[x] += 1
            f = _fan_of(c, nm, baotou, ctx.chain, ctx.piao)
            if conv and _can_convert_key(tuple(c), nm):
                # 摸到这张后可「拒胡转爆头听」（act_draw 会这么做）：真实排位 2.3k 次拒胡点 E[终番]/当时番 ≈1.8
                f *= 1.0 + cb
            c[x] -= 1
            W += u
            F += u * f
        if baotou:
            # 爆头：relevant 之外的牌（其他花色/未持字牌）同样胡，且番与「财神顶替」的最低番一致
            # ——此前漏算，使爆头听胜率被低估 30%～50%（真实 p=1）。
            rel = set(wins)
            f_other = None
            for x in range(34):
                if x in rel:
                    continue
                u = ctx.unseen[x]
                if u <= 0:
                    continue
                if f_other is None:
                    c[x] += 1
                    f_other = _fan_of(c, nm, True, ctx.chain, ctx.piao)
                    c[x] -= 1
                W += u
                F += u * f_other
    if W == 0:
        v = 0.0
    else:
        p = W / ctx.U
        v = _geom(p, ctx.gamma, n, ctx.turn + (ctx.n_draws - n), ctx.surv) * p * ((F / W) * ctx.mult + ctx.win_bonus)
    ctx.cache0[key] = v
    return v


@lru_cache(maxsize=1 << 16)
def _useful_normal_key(key, nm, base):
    counts = list(key)
    out = []
    for x in _relevant_tiles(counts):
        counts[x] += 1
        if shanten_normal(counts, nm) < base:
            out.append(x)
        counts[x] -= 1
    return tuple(out)


def _route_value(A, U, s, typ, fan_est, ctx, n=None):
    """自上而下：先等实际进张（p=A/U），随后每层用典型进张常数；每层的起始巡目按期望等待推进。"""
    v = fan_est * ctx.mult + ctx.win_bonus
    if n is None:
        n = ctx.n_draws
    t0 = ctx.turn + (ctx.n_draws - n)
    p = min(1.0, A / U)
    v *= _geom(p, ctx.gamma, n, t0, ctx.surv) * p
    wait = min(n, 1.0 / max(p, 1e-6))
    n_left = n - wait
    t = t0 + wait
    smooth = ctx.params.get("n_smooth")
    for k in range(s - 1, -1, -1):
        Ak = typ[k] if k < len(typ) else typ[-1]
        p = min(1.0, Ak / U)
        if smooth:
            # 分数剩余次数 / 分数巡目：估值随牌墙余数连续变化，不再在 round() 边界跳变
            nn = max(1.0, n_left)
            v *= _geom(p, ctx.gamma, nn, t, ctx.surv) * p
        else:
            nn = max(1, int(round(n_left)))
            v *= _geom(p, ctx.gamma, nn, int(round(t)), ctx.surv) * p
        wait = min(nn, 1.0 / p)
        n_left -= wait
        t += wait
    return v


def value_approx(counts13, nm, s, ctx, n=None):
    """向听 s≥1 的解析近似：按「普通型」「七对」两条路线分别估算（各自的实际进张 +
    更深层典型进张常数），取较大者。"""
    if n is None:
        n = ctx.n_draws
    tab = ctx.params.get("typ_accept_by_melds")
    typ = tab[min(nm, len(tab) - 1)] if tab else ctx.params["typ_accept"]
    key = tuple(counts13)
    U = ctx.U
    g = counts13[GOD]
    god_fan = 1.0 + ctx.params.get("god_fan_slope", 0.35) * g if g else 1.0
    chain_fan = (1 << ctx.chain) if ctx.chain else 1.0
    best = 0.0
    # 普通型
    s_n = shanten_normal(counts13, nm)
    if s_n <= 6:
        useful = _useful_normal_key(key, nm, s_n) if s_n > 0 else _win_tiles_key(key, nm)
        A = sum(ctx.unseen[x] for x in useful)
        if A > 0:
            if s_n <= 0:
                best = value0(counts13, nm, ctx, n)
            else:
                best = _route_value(A, U, s_n, typ, god_fan * chain_fan, ctx, n)
    # 七对（无副露）
    if nm == 0:
        s_q = shanten_qidui(counts13)
        if 0 < s_q <= 4:
            singles = [i for i in range(33) if counts13[i] == 1]
            A = sum(ctx.unseen[x] for x in singles)
            if A > 0:
                # 七对每步进张 ≈ 3 × 单张数（单张数 = 1 + 2k + g），未见比例 ~0.8
                typ_q = [max(2.0, 2.4 * (1 + 2 * k + g)) for k in range(s_q)]
                fan_q = 2.0 * ctx.params["qidui_bonus"] * god_fan * chain_fan
                vq = _route_value(A, U, s_q, typ_q, fan_q, ctx, n)
                if vq > best:
                    best = vq
        elif s_q == 0:
            vq = value0(counts13, nm, ctx, n)
            if vq > best:
                best = vq
    return best


def value(counts13, nm, ctx, s=None, n=None):
    """13 张形手牌估值（任意向听）。n = 剩余可摸次数。"""
    if n is None:
        n = ctx.n_draws
    if n <= 0:
        return 0.0
    if s is None:
        s = shanten(counts13, nm)
    if s <= 0:
        return value0(counts13, nm, ctx, n)
    if s > ctx.params["lookahead"]:
        return value_approx(counts13, nm, s, ctx, n) * ctx.params["approx_scale"]
    key = (tuple(counts13), nm, n)
    v = ctx.cache1.get(key)
    if v is not None:
        return v
    useful = _useful_key(key[0], nm, s)
    c = list(counts13)
    total = 0.0
    A = 0
    # 进张平均要等 1/p 次摸牌；用「期望等待后剩余次数」近似下一层的 n
    for x in useful:
        u = ctx.unseen[x]
        if u <= 0:
            continue
        A += u
    if A == 0:
        ctx.cache1[key] = 0.0
        return 0.0
    p = A / ctx.U
    wait = min(n, 1.0 / p)
    if ctx.params.get("n_smooth"):
        # 分数剩余次数（1/4 步长：兼顾平滑与下层 cache 命中）——避免两张进张只差 1% 的候选在下一层差整整 1 次摸牌
        n_next = max(1.0, round((n - wait) * 4) / 4)
    else:
        n_next = max(1, int(round(n - wait)))
    for x in useful:
        u = ctx.unseen[x]
        if u <= 0:
            continue
        c[x] += 1
        best = best_discard_value(c, nm, ctx, s - 1, n_next)[1]
        c[x] -= 1
        total += u * best
    v = _geom(p, ctx.gamma, n, ctx.turn + (ctx.n_draws - n), ctx.surv) * (total / ctx.U)
    ctx.cache1[key] = v
    return v


def best_discard_value(counts14, nm, ctx, target_s=None, n=None, feed=None):
    """14 张形：返回 (最佳打出牌索引, 估值)。只考虑打出后向听 = 最小向听的候选；财神不打。
    feed：喂牌惩罚表（仅顶层出牌传入）：候选估值 −= P(被吃碰)×feed_penalty。"""
    c = counts14
    cands = []
    min_s = 99
    # 有财必拷响下持财神：打掉财神转普通听也是合法选项（否则永远不能平胡）
    n_cand = 34 if (ctx.bikao and c[GOD] > 0 and ctx.chain == 0) else 33
    for d in range(n_cand):
        if c[d] <= 0:
            continue
        c[d] -= 1
        s = shanten(c, nm)
        c[d] += 1
        cands.append((d, s))
        if s < min_s:
            min_s = s
    if not cands:
        return None, 0.0
    if target_s is not None and min_s > target_s:
        target_s = min_s
    best_d, best_v = None, -1.0
    eps = ctx.params.get("tie_eps", 0.0)
    fp = ctx.params.get("feed_penalty") or 0.0
    scored = []
    for d, s in cands:
        if s != min_s:
            continue
        c[d] -= 1
        v = value(c, nm, ctx, s, n)
        c[d] += 1
        if feed is not None and fp:
            v -= feed[d] * fp
        scored.append((v, d))
        if v > best_v:
            best_v, best_d = v, d
    if eps > 0 and best_v > 0:
        # 近似平手（EV 差 < eps）时按高手习惯打：孤张字牌 > 孤张幺九 > 其他（真实排位 5.7k 手出牌分歧的主因）
        def pref(d):
            if d >= 27:
                return 0 if c[d] == 1 else 3
            r = d % 9 + 1
            lo = d - (d % 9)
            iso = c[d] == 1 and all(c[j] == 0 for j in range(max(lo, d - 2), min(lo + 9, d + 3)) if j != d)
            return (1 if r in (1, 9) else 2) if iso else 3
        cut = best_v * (1 - eps)
        near = [(pref(d), -v, d) for v, d in scored if v >= cut]
        near.sort()
        best_d = near[0][2]
    return best_d, best_v


# ---------- Agent ----------

class Brain:
    name = "brain"

    def __init__(self, params=None, name=None):
        self.params = dict(DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        if name:
            self.name = name
        self.last_info = {}

    # ---- 局面 → Ctx ----
    def make_ctx(self, snap, chain=None, piao=None):
        seat = snap["seat"]
        my = snap["my_hand"]
        unseen = [4] * 34
        for t in my:
            unseen[TILE_INDEX[t]] -= 1
        for d in snap["discards"]:
            for t in d:
                unseen[TILE_INDEX[t]] -= 1
        p_river = self.params.get("river_keeps_claimed", 1)
        opp_melds = 0
        for s, ms in enumerate(snap["melds"]):
            for m in ms:
                for t in m["tiles"]:
                    unseen[TILE_INDEX[t]] -= 1
                # 服务端语义（1138 个快照实证：被吃碰的牌仍留在出牌者河里，可见张数会到 5）：河牌 + 副露会把被吃/碰/明杠的
                # 那一张扣两次。补回一张（吃：tiles[-1]；碰/明杠/补杠：tiles[0]；暗杠不来自河）。平均每快照 0.21 张、
                # 14% 的快照里有牌被误判为「已绝」。模拟器 hz_sim 同样保留河牌，口径一致。
                k = m.get("kind")
                if p_river:
                    if k == "chi":
                        unseen[TILE_INDEX[m["tiles"][-1]]] += 1
                    elif k in ("peng", "gang_ming", "gang_bu", "gang"):
                        unseen[TILE_INDEX[m["tiles"][0]]] += 1
                if s != seat:
                    opp_melds += 1
        unseen = [max(0, x) for x in unseen]
        unseen_raw = unseen
        ww = self.params.get("wall_weight")
        if ww:
            # 牌墙密度修正：自摸规则下进张概率 = P(牌在墙里)，而 unseen/U 把「他家手里」与「墙里」混同。
            # 他家早打字牌/幺九、留中张 → 未见的字牌/幺九更可能在墙里，中张/白更可能在他家手里。
            # 权重表 [牌种][他家河里该牌张数 d(封顶3)]，牌种行序 字/幺九/28/中张/白；按 U 不变重新归一。
            odc = [0] * 34
            for s, d in enumerate(snap["discards"]):
                if s != seat:
                    for t in d:
                        odc[TILE_INDEX[t]] += 1
            tot = sum(unseen)
            wt = [u * ww[_KIND_ROW[k]][min(3, odc[k])] for k, u in enumerate(unseen)]
            ws = sum(wt)
            if ws > 0 and tot > 0:
                f = tot / ws
                unseen = [x * f for x in wt]
        wall = snap["wall_remaining"]
        reserve = snap.get("rules", {}).get("wall_reserve", 20)
        drawable = max(0, wall - reserve)
        # 我下一次摸牌前还有几家要摸：出牌阶段（刚摸完）= 3 家；响应窗口由调用方另算
        p = self.params
        if p.get("n_smooth") and p.get("n_frac"):
            # 分数剩余次数：牌墙每少 1 张估值连续变化（整数 //4 在边界处一次跳 1 次摸牌 ≈ EV 跳 10–20%）
            n_draws = max(1.0, drawable / 4.0) if snap.get("phase") == "draw" else max(1.0, (drawable + 3) / 4.0)
        else:
            n_draws = max(1, drawable // 4) if snap.get("phase") == "draw" else max(1, (drawable + 3) // 4)
        turn_no = len(snap["discards"][seat]) + (1 if snap.get("phase") == "draw" else 0)
        hazard = p["hazard_base"] + p["hazard_meld"] * opp_melds + p["hazard_turn"] * turn_no
        gamma = max(0.5, 1.0 - hazard)
        surv = None
        curve = p.get("hazard_curve")
        if curve:
            # 经验风险曲线（按巡）× 他家副露强度系数 → 累计存活表
            scale = (1.0 + p["hazard_meld_scale"] * opp_melds) * (p.get("hazard_scale") or 1.0)
            surv = [1.0]
            for t in range(1, 40):
                h = curve[min(t, len(curve) - 1)] * scale
                surv.append(surv[-1] * max(0.0, 1.0 - min(0.95, h)))
        dealer = snap["dealer"] == seat
        mult = 24.0 if dealer else 10.0
        if chain is None:
            chain = snap["god"]["chain_count"]
        if piao is None:
            # 链内已飘出的白板数：快照若不带 piao_count 则按「链全是飘」近似（明杠/暗杠会高估 4 白板判定，极罕见）
            piao = snap["god"].get("piao_count")
            if piao is None:
                piao = chain
        bikao = bool(snap.get("rules", {}).get("YouCaiBiKao"))
        ctx = Ctx(unseen, gamma, n_draws, mult, chain, p, drawable, surv, turn_no, bikao, piao)
        lf = p.get("loss_fan") or 0.0
        if lf:
            # 被他家胡的期望损失：庄家付 8×番；闲家付 1×番（闲胡）或 8×番（庄胡），按 1/3 庄胡 ≈ 10/3
            ctx.win_bonus = lf * (8.0 if dealer else 10.0 / 3.0)
        # 连庄价值：胡牌者下局坐庄（庄胡连庄、流局连庄，无倍数递增——排位 1971 局实证）。
        # 坐庄一局相对闲家的净分差：真实排位我方 +1.9、自对弈 +1.2。作为每次胡的固定加分。
        ctx.win_bonus += p.get("dealer_value") or 0.0
        if p.get("feed_penalty"):
            ctx.feed = feed_table(unseen_raw)
        return ctx

    @staticmethod
    def can_hu_now(snap, c, nm):
        """当前 14 张能否胡：只能自摸 + 有财必拷响（持财神且 chain==0 时必须已处于爆头态）。"""
        if snap.get("drawn_tile") is None or not is_win(c, nm):
            return False
        god = snap["god"]
        if snap.get("rules", {}).get("YouCaiBiKao") and c[GOD] > 0 and not god.get("chain_count") and not god.get("baotou"):
            return False
        return True

    # ---- 摸牌后 ----
    def act_draw(self, snap):
        seat = snap["seat"]
        hand = snap["my_hand"]
        c = hand_to_counts(hand)
        melds = snap["melds"][seat]
        nm = len(melds)
        drawn = snap.get("drawn_tile")
        god = snap["god"]
        # 只能自摸：吃/碰之后（没有摸牌）即使 3N+2 成形也不能胡，只能打牌听牌
        can_hu = self.can_hu_now(snap, c, nm)
        ctx = self.make_ctx(snap)
        # 链在「打出普通牌」时断（服务端实证：杠后补牌→打普通牌→再自摸，492 局无一计杠开）。链>0 的出牌点
        # （杠后补牌 / 飘后）估普通出牌必须用 chain=0 的 ctx，否则拒胡/续杠基线被 ×2^链 高估
        # （排位 71 个杠后补牌可胡点，旧口径 16 个误拒胡：ev_decline 38.9 vs 真实 19.4 < 立刻胡 21.7）。
        # 飘（打白后仍爆头）走 chain+1 的 ctx2，不受影响。
        ctx_d = ctx
        if int(god.get("chain_count") or 0) and self.params.get("chain_break_on_discard", 1):
            ctx_d = self.make_ctx(snap, chain=0, piao=0)
        # 财飘判定：爆头态 + 能胡 + 手里 ≥2 白 → 打一张白后 13 张仍是爆头听则链 +1（×2）。
        # 摸到的不必是白板：爆头态摸任意牌都能胡，只要打白后仍爆头就成飘（排位 36 个飘点里 14 个摸的是非白，
        # 高手同样飘：铳一色14 摸 8w 飘 +96、乌鸦坐飞机 摸 2w 飘 +40）。旧代码只在摸白时考虑，漏掉了这类点。
        if can_hu:
            if god.get("baotou") and c[GOD] >= 2 and not god.get("catch_play") \
                    and (drawn == TILE_NAMES[GOD] or self.params.get("piao_any_draw")):
                fan_now, _ = calc_fan(c, nm, True, god["chain_count"], ctx.piao)
                ev_now = fan_now * ctx.mult + ctx.win_bonus
                c13 = list(c); c13[GOD] -= 1
                if is_baotou(c13, nm):
                    # 飘出的白仍计入「4 白板」（手留 + 链内飘出）：4 白飘后仍 ×2，不是丢掉一倍
                    ctx2 = self.make_ctx(snap, chain=god["chain_count"] + 1, piao=ctx.piao + 1)
                    ev_piao = value0(c13, nm, ctx2)
                    if ev_piao >= ev_now * (1 + self.params["piao_margin"]) and ctx.n_draws >= 3:
                        self.last_info = {"piao": True, "ev_now": ev_now, "ev_piao": ev_piao}
                        return {"action": "discard", "tile": TILE_NAMES[GOD]}
            # 拒胡转爆头听：手里有财神、打一张后成「摸什么都胡」的爆头听（番 ×2），且估值高于立刻胡。
            # 真实排位高手（Fable-6/宝宝疯了/Astra-0 等）都这么做；只允许转爆头，不允许转普通听。
            hdm = self.params.get("hu_decline_margin")
            if hdm is not None and c[GOD] >= 1 and ctx.n_draws >= 3 and not god.get("catch_play"):
                fan_now, _ = calc_fan(c, nm, bool(god.get("baotou")), god.get("chain_count", 0), ctx.piao)
                ev_now = fan_now * ctx.mult + ctx.win_bonus
                d, v = best_discard_value(c, nm, ctx_d)
                if d is not None and v > ev_now * (1 + hdm):
                    c13 = list(c); c13[d] -= 1
                    if is_baotou(c13, nm):
                        self.last_info = {"decline_hu": True, "ev_now": ev_now, "ev_decline": v}
                        return {"action": "discard", "tile": TILE_NAMES[d]}
            return {"action": "hu"}
        if god.get("catch_play") and drawn:
            return {"action": "discard", "tile": drawn}
        # 暗杠 / 补杠
        if snap["wall_remaining"] >= snap.get("rules", {}).get("gang_min_wall", 21):
            g = self._consider_gang(c, nm, melds, ctx, god, ctx_d)
            if g:
                return g
        d, v = best_discard_value(c, nm, ctx_d, feed=ctx.feed)
        if d is None:
            d = GOD  # 只剩财神（极罕见）
        # 吃/碰后（不能胡）的出牌：手里 ≥2 白、打白后仍是爆头听 → 飘（链 +1，×2）。排位里 12 个此类点
        # 高手飘了 3 次（腾蛇 +80 4白飘）；其余打普通牌保留白。按 EV 比较（飘的 value0 用 chain+1 的 ctx）。
        if self.params.get("piao_any_draw") and c[GOD] >= 2 and not god.get("catch_play") and ctx.n_draws >= 3 \
                and d != GOD:
            c13 = list(c); c13[GOD] -= 1
            if is_baotou(c13, nm):
                ctx2 = self.make_ctx(snap, chain=int(god.get("chain_count") or 0) + 1, piao=ctx.piao + 1)
                vp = value0(c13, nm, ctx2)
                if vp >= v * (1 + self.params["piao_margin"]):
                    self.last_info = {"piao": True, "postmeld": True, "ev_now": v, "ev_piao": vp}
                    return {"action": "discard", "tile": TILE_NAMES[GOD]}
        self.last_info = {"value": v}
        return {"action": "discard", "tile": TILE_NAMES[d]}

    def _consider_gang(self, c, nm, melds, ctx, god, ctx_d=None):
        """ctx：链口径（杠 = 链动作 +1，补牌胡按 chain+1 计番）；ctx_d：普通出牌口径（链断 → chain=0），
        用于「不杠直接打牌」的基线与补牌后不能胡时的出牌估值。"""
        if ctx_d is None:
            ctx_d = ctx
        options = []
        for t in range(33):
            if c[t] == 4:
                options.append((t, "an"))
            elif c[t] >= 1:
                for m in melds:
                    if m["kind"] == "peng" and TILE_INDEX[m["tiles"][0]] == t:
                        options.append((t, "bu"))
        if not options:
            return None
        base_d, base_v = best_discard_value(c, nm, ctx_d)
        best = None
        for t, kind in options:
            c2 = list(c)
            nm2 = nm
            if kind == "an":
                c2[t] -= 4
                nm2 = nm + 1
            else:
                c2[t] -= 1
            # 补牌：期望
            baotou = is_baotou(c2, nm2)
            total = 0.0
            for x in range(34):
                u = ctx.unseen[x]
                if u <= 0:
                    continue
                c2[x] += 1
                if is_win(c2, nm2):
                    v = _fan_of(c2, nm2, baotou, ctx.chain + 1, ctx.piao) * ctx.mult + ctx.win_bonus
                else:
                    v = best_discard_value(c2, nm2, ctx_d)[1]
                c2[x] -= 1
                total += u * v
            ev = total / ctx.U
            if ev > base_v * (1 + self.params["gang_margin"]) and (best is None or ev > best[0]):
                best = (ev, t)
        if best:
            return {"action": "gang", "tile": TILE_NAMES[best[1]]}
        return None

    # ---- 响应窗口 ----
    def act_response(self, snap):
        t_start = time.perf_counter()
        seat = snap["seat"]
        hand = snap["my_hand"]
        c = hand_to_counts(hand)
        melds = snap["melds"][seat]
        nm = len(melds)
        last = snap["last_discard"]
        if last is None or last == TILE_NAMES[GOD] or snap["god"].get("catch_play"):
            return {"action": "pass"}
        ti = TILE_INDEX[last]
        ctx = self.make_ctx(snap, chain=0)
        if self.params["meld_lookahead"] != self.params["lookahead"]:
            ctx.params = dict(ctx.params)
            ctx.params["lookahead"] = self.params["meld_lookahead"]
        # 副露不消耗牌墙：我之后的摸牌次数基本不变（吃仅在牌墙余数 ≡1 mod 4 时少 1 次，
        # 平均 −0.25），却白得一张「正好要的牌」；碰/杠还抢到先手（中间几家少摸一轮）→ tempo_bonus。
        src = snap.get("last_discard_seat")
        if src is None:
            src = snap.get("turn")
        skipped = (seat - src - 1) % 4 if src is not None else 1
        drawable = ctx.drawable
        n_pass = max(1, (drawable - skipped + 3) // 4)   # 不动：前面 skipped 家先摸
        n_after = max(1, drawable // 4)                  # 副露后我先出牌，之后三家摸完才轮到我
        base = value(c, nm, ctx, None, n_pass)
        bonus = 1.0 + self.params["tempo_bonus"] * skipped
        pen = 1.0 - self.params["meld_penalty"]
        gate = base * (1 + self.params["meld_margin"])
        best = None
        cand_log = []
        gang_info = {}
        phase = snap["phase"]
        cur_s = shanten(c, nm)
        # 副露口径可按庄/闲分开（*_dealer 键，缺省 = 与闲家同）。r8 2v2 曾显示 @meld_loose 的增益全在庄局，
        # 但 r10 1v3 证明闲局同样有正收益（庄 +0.13/闲 +0.08 每局）；@meld_loose_dealer 只拿到 2/3 增益 → 机制保留、不推荐。
        mp = self.params
        if snap["dealer"] == seat:
            mp = {k: (self.params[k + "_dealer"] if self.params.get(k + "_dealer") is not None else self.params[k])
                  for k in ("meld_need_shanten", "meld_margin_drop", "meld_margin_same", "meld_need_shanten_from")}
        need_s = mp["meld_need_shanten"]

        def ok_shanten(c_after, nm_after):
            # 副露后（3N+2 形）打一张的最小向听 必须 < 当前向听（need_s=1）或 ≤（need_s=0）
            # meld_need_shanten_from：当前向听 ≥ 此值时（近似层估值不够准）强制要求向听下降
            if not need_s and cur_s < mp["meld_need_shanten_from"]:
                return True
            # 3N+2 形的向听 = 打出最佳一张后的向听；副露直接成「完成形」(−1) 也算下降
            return shanten(c_after, nm_after) < cur_s

        def meld_ev(c_after, nm_after):
            """副露候选的 EV；向听下降的副露按 margin_drop 门槛，不下降的按 margin_same 门槛。"""
            if not ok_shanten(c_after, nm_after):
                return -1.0
            v = best_discard_value(c_after, nm_after, ctx, None, n_after)[1] * bonus * pen
            drop = shanten(c_after, nm_after) < cur_s
            m = mp["meld_margin_drop"] if drop else mp["meld_margin_same"]
            return v if v > base * (1 + m) else -1.0
        if phase == "response_peng" and c[ti] >= 2:
            if c[ti] >= 3 and snap["wall_remaining"] >= snap.get("rules", {}).get("gang_min_wall", 21):
                c[ti] -= 3
                # 明杠：补牌期望（杠开 ×2）。链不因明杠断：已有链（刚飘过）时是 ctx.chain+1，与暗杠口径一致
                # （排位 1971 局里明杠前链>0 的样本为 0，纯正确性修正，无实际影响）。
                baotou = is_baotou(c, nm + 1)
                chain_g = int(snap["god"].get("chain_count") or 0) + 1
                s_after = shanten(c, nm + 1)
                # 时间预算（gang_budget）：先精确算「进张」补牌（价值主体），再算其余补牌；超预算后剩余补牌按
                # 「补到无用张 ≈ 杠后 13 张原估值」合并近似（略低估 → 偏保守，不会多杠）。
                useful = set(_useful_key(tuple(c), nm + 1, s_after))
                order = sorted(range(34), key=lambda x: (x not in useful, x))
                budget = self.params.get("gang_budget") or 0.0
                total = 0.0
                u_rest = 0
                n_approx = 0
                for x in order:
                    u = ctx.unseen[x]
                    if u <= 0:
                        continue
                    if budget:
                        el = time.perf_counter() - t_start
                        # 软预算：无用张改近似；硬上限（2×）：进张也改近似（进张按 13 张估值近似会低估杠开 → 偏向碰/不动，
                        # 但仍能在 1s 碰窗口内提交；线上 CPU 争用下本地 150ms 的点曾跑到 984ms）
                        if el > budget * 2 or (el > budget and x not in useful):
                            u_rest += u
                            n_approx += 1
                            continue
                    c[x] += 1
                    if is_win(c, nm + 1):
                        v = _fan_of(c, nm + 1, baotou, chain_g, ctx.piao) * ctx.mult + ctx.win_bonus
                    else:
                        v = best_discard_value(c, nm + 1, ctx, None, n_after)[1]
                    c[x] -= 1
                    total += u * v
                if u_rest:
                    total += u_rest * value(c, nm + 1, ctx, s_after, n_after)
                c[ti] += 3
                ev = total / ctx.U * bonus * pen
                gang_info = {"gang_ev": ev, "gang_approx": n_approx}
                if ev > gate and (best is None or ev > best[0]):
                    best = (ev, {"action": "gang", "tile": last})
            c[ti] -= 2
            ev = meld_ev(c, nm + 1)
            c[ti] += 2
            cand_log.append(("peng", ev))
            if ev > 0 and (best is None or ev > best[0]):
                best = (ev, {"action": "peng", "tile": last})
        elif phase == "response_chi" and is_suited(ti):
            chi_used = sum(1 for m in melds if m["kind"] == "chi")
            if chi_used < snap.get("rules", {}).get("max_chi", 2):
                r = rank_of(ti); basei = ti - (r - 1)
                for lo in (r - 2, r - 1, r):
                    if lo < 1 or lo + 2 > 9:
                        continue
                    need = [basei + lo - 1 + k for k in range(3) if basei + lo - 1 + k != ti]
                    if all(c[x] >= 1 for x in need):
                        for x in need: c[x] -= 1
                        ev = meld_ev(c, nm + 1)
                        for x in need: c[x] += 1
                        cand_log.append(("chi", [TILE_NAMES[x] for x in need], ev))
                        if ev > 0 and (best is None or ev > best[0]):
                            best = (ev, {"action": "chi", "tile": last, "tiles": [TILE_NAMES[x] for x in need]})
        if best:
            self.last_info = {"meld_ev": best[0], "base": base, **gang_info}
            return best[1]
        self.last_info = {"meld_ev": None, "base": base, "cands": cand_log, **gang_info}
        return {"action": "pass"}


if __name__ == "__main__":
    import sys, io, time
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    from hz_sim import Game, GreedyAgent
    t0 = time.time()
    agents = [Brain(), GreedyAgent(), Brain(), GreedyAgent()]
    g = Game(agents, seed=5, verbose=True)
    g.play()
    print("scores", g.scores, "stats", g.stats, f"{time.time()-t0:.1f}s")
