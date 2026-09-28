#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端验证锦标赛模式（假服务器）——完整跑完多阶段，不误杀。

判定（对齐开发 agent 冻结清单「9/17 测试赛要看的」）：
  □ 报名期 register/ready 均成功
  □ 赛事状态出现 running（跟到阶段切换）
  □ 新场次被自动接入（「新场次」行）
  □ 有实际出牌/副露动作
  □ 阶段结束/下一阶段确认（stage_done / stage_open / 再 ready）
  □ 无 赛事轮询异常 / 对局线程异常 / 线程已退出
用法: python verify_tournament_e2e.py [--fault-game-storm] [--budget 300]
"""
import os, subprocess, sys, time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PORT = 18903
OUT = os.path.join(REPO, "collab", "data", "tournament_e2e.md")
ENV = {**os.environ, "NO_PROXY": "*", "HTTP_PROXY": "", "HTTPS_PROXY": "",
       "http_proxy": "", "https_proxy": ""}
BUDGET = 300
if "--budget" in sys.argv:
    BUDGET = int(sys.argv[sys.argv.index("--budget") + 1])


def run_case(storm, budget):
    fs_cmd = [sys.executable, "-u", os.path.join(HERE, "hz_fakeserver.py"),
              "--port", str(PORT), "--tournament", "--games", "2", "--rounds", "3",
              "--reg-sec", "2", "--gap-sec", "2", "--stages", "2"]
    if storm:
        fs_cmd += ["--fault-game-storm"]
    fs = subprocess.Popen(fs_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", env=ENV)
    time.sleep(2)
    bot = subprocess.Popen([sys.executable, "-u", os.path.join(HERE, "hz_bot.py"),
                            "--token", "T", "--server", f"http://127.0.0.1:{PORT}",
                            "--name", "E2E", "--dump", os.path.join(REPO, "collab", "data", "e2e_dump")],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, encoding="utf-8", errors="replace", env=ENV)
    t0 = time.time()
    exited = False
    try:
        out, _ = bot.communicate(timeout=budget)
        exited = True
    except subprocess.TimeoutExpired:
        bot.kill()
        out, _ = bot.communicate()
    dur = time.time() - t0
    try:
        fs_out, _ = fs.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        fs.kill(); fs_out, _ = fs.communicate()
    lines = (out or "").splitlines()
    ck = {
        "register OK": any("register OK" in l for l in lines),
        "ready OK": any("ready OK" in l for l in lines),
        "赛事状态 running": any("赛事状态 running" in l for l in lines),
        "新场次": any("新场次" in l for l in lines),
        "实际动作": any(("→ chi" in l or "→ peng" in l or "→ hu" in l or "→ gang" in l) for l in lines),
        "stage_open/确认": any(("stage_open" in l or "确认期" in l or "阶段" in l) for l in lines),
        "无异常行": not any(("赛事轮询异常" in l or "对局线程异常" in l or "线程已退出" in l) for l in lines),
    }
    return {"storm": storm, "dur": dur, "exited": exited, "rc": bot.returncode,
            "ck": ck, "lines": lines, "fs_lines": (fs_out or "").splitlines()}


def main():
    cases = [run_case(False, BUDGET)]
    if "--fault-game-storm" in sys.argv:
        cases.append(run_case(True, BUDGET))
    L = ["# 锦标赛模式端到端验证（假服务器）", ""]
    for c in cases:
        L += [f"## {'带故障风暴' if c['storm'] else '无故障'}｜{c['dur']:.0f}s｜bot {'正常退出' if c['exited'] else '被我方超时终止'}（rc={c['rc']}）", "",
              "| 检查项 | 结果 |", "|---|---|"]
        for k, v in c["ck"].items():
            L.append(f"| {k} | {'✅' if v else '❌'} |")
        st = sum(1 for v in c["ck"].values() if v)
        L.append(f"\n**{st}/7 项通过**")
        L += ["", "<details><summary>bot 输出（尾部 40 行）</summary>", "", "```"]
        L += c["lines"][-40:]
        L += ["```", "", "</details>", "", "<details><summary>假服务器输出（尾部 25 行）</summary>", "", "```"]
        L += c["fs_lines"][-25:]
        L += ["```", "", "</details>", ""]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:40]))
    print(f"\n已写入 {OUT}")


if __name__ == "__main__":
    main()
