# 杭州麻将 AI 竞技赛 · 参赛程序（Carrion）

完整可运行的参赛源码。接入与启动方式见 **`使用说明.md`**。

| 文件 | 作用 |
|---|---|
| `scripts/hz_bot.py` | **入口**：官方平台接入 + 对局循环（参赛主程序）|
| `scripts/hz_brain.py` | **决策引擎**：向听搜索 + 期望值评估 |
| `scripts/hz_core.py` | 牌型解析 / 算番 / 规则 |
| `scripts/hz_sim.py` | 局面模拟与候选评估 |
| `scripts/hz_fakeserver.py` | 本地假服务器（**自测用**，非参赛必需）|
| `scripts/verify_tournament_e2e2.py` | 端到端自测（无需真实平台）|

## 启动

```bash
python scripts/hz_bot.py --token <令牌> --server https://10.240.169.190:18080 --name <昵称>
```

## 自测（本地假服务器，无需真实平台）

```bash
python scripts/verify_tournament_e2e2.py --budget 420
```

> ⚠️ 自测里的 `stage_open/确认` 依赖阶段推进落在预算内，**预算不足会假失败**（实测 300s 偶发 5~6/7、420~540s 稳定 7/7）⇒ 建议 `--budget 420` 以上。

环境：**Python 3.8+，纯标准库，无第三方依赖**。

## 许可

[MIT](LICENSE)。

> 本仓库原为比赛提交仓库（赛事已结束，现转为公开）。仓库内的 `round1` / `round2` / `round3` / `final`
> 四个 tag **全部指向同一个提交** `8ff249f` —— 即第 1～4 轮**实际参赛所用、逐字节未改动**的那份代码；
> 之后新增的提交（如本文件与 `LICENSE`）**不含任何参赛代码改动**。
