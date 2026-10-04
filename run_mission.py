"""
Run the full pipeline on your laptop:
    Perception -> Planner -> 2 PPO Robots -> Stats -> Gemini Report -> Email alerts

Outputs: demo.mp4, trajectory.json, stats.json, report.md  (these go to the Streamlit app)

Usage:
    python run_mission.py                 # full run (sends emails if configured)
    python run_mission.py --no-email      # everything except sending emails
    python run_mission.py --skip-baseline # skip the single-robot comparison run
    python run_mission.py --inject-fault  # test run: Robot 2's steering is sabotaged, Supervisor must fix it
"""

import argparse
import json
import os
from datetime import datetime

import imageio

from multi_agent import run_episode
from business_automation import (load_env, compute_metrics, generate_report,
                                 decide_alerts, send_email)
MODEL_PATH = "weeding_ppo_model.zip"

FPS = 12
FAULT = {"robot": 1, "start": 40, "end": 75}   # used only with --inject-fault


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-email", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--inject-fault", action="store_true",
                        help="deliberately corrupt Robot 2's steering so the Supervisor must correct it (test)")
    args = parser.parse_args()

    load_env()

    model, mode = None, "RULE"
    if os.path.exists(MODEL_PATH):
        from stable_baselines3 import PPO
        model, mode = PPO.load(MODEL_PATH), "PPO"
    else:
        print(f"{MODEL_PATH} not found -> using rule-based controllers.")
    print(f"Controller mode: {mode}")

    baseline = None
    if not args.skip_baseline:
        print("1/4 Running single-robot baseline...")
        baseline, _ = run_episode(1, model, record=False)
        baseline.pop("trajectory", None)
        print(f"    baseline: {baseline['weeds_removed']}/{baseline['total_weeds']} weeds "
              f"in {baseline['steps']} steps")

    print("2/4 Running 2-robot multi-agent mission (recording video)...")
    fault = FAULT if args.inject_fault else None
    team, frames = run_episode(2, model, record=True, fault=fault)
    trajectory = team.pop("trajectory")
    with open("trajectory.json", "w", encoding="utf-8") as f:
        json.dump(trajectory, f)
    imageio.mimsave("demo.mp4", frames, fps=FPS, macro_block_size=1)
    print(f"    team: {team['weeds_removed']}/{team['total_weeds']} weeds in {team['steps']} steps, "
          f"{team['crop_touches']} crop touches")
    sup = team.get("supervisor", {})
    print(f"    supervisor corrections: {sup.get('interventions', 0)}"
          + (" (fault test ON)" if fault else ""))
    for e in sup.get("events", []):
        print(f"      step {e['start']}: {e['robot_name']} - {e['reason']}"
              + (" [injected fault]" if e["injected"] else ""))

    print("3/4 Writing AI report...")
    metrics = compute_metrics(team, baseline)
    report, source = generate_report(team, metrics, mode)
    with open("report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("4/4 Alerts...")
    alerts = decide_alerts(team, report)
    alert_log = []
    for a in alerts:
        if args.no_email:
            ok, detail = False, "skipped (--no-email)"
        else:
            ok, detail = send_email(a["subject"], a["body"])
        print(f"    [{a['type']}] {a['subject']} -> {detail}")
        alert_log.append({"type": a["type"], "subject": a["subject"],
                          "sent": ok, "detail": detail})

    result = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "controller": mode,
        "fault_test": bool(fault),
        "team": team,
        "baseline": baseline,
        "metrics": metrics,
        "report": report,
        "report_source": source,
        "alerts": alert_log,
    }
    with open("stats.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("Done. Saved demo.mp4, trajectory.json, stats.json, report.md")


if __name__ == "__main__":
    main()