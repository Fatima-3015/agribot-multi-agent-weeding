"""
AI Business Automation
------------------------------------------------------
1. compute_metrics  -> business numbers from the mission stats
2. generate_report  -> Gemini writes a Farm Report (template fallback if it fails)
3. decide_alerts    -> which emails should go out
4. send_email       -> Gmail SMTP

Secrets are read from environment variables or a local .env file (never hard-code):
    GEMINI_API_KEY, GMAIL_ADDRESS, GMAIL_APP_PASSWORD, ALERT_TO (optional)
    GEMINI_MODEL (optional, to force a model)
"""

import json
import os
import smtplib
import urllib.error
import urllib.request
from email.message import EmailMessage

# ----------------------------------------------------------------------
# ASSUMPTIONS for the business estimate (edit these freely).
# They are illustrative numbers, NOT measured data.
# ----------------------------------------------------------------------
ASSUMPTIONS = {
    "manual_minutes_per_weed": 2.0,       # a labourer needs ~2 min per weed
    "labour_rate_pkr_per_hour": 250,      # assumed hourly wage (PKR)
    "crop_loss_pkr_per_touch": 50,        # assumed loss each time a crop is hit (PKR)
}

GEMINI_MODELS = ["gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-3-flash-preview"]


def load_env(path=".env"):
    """Tiny .env loader (no extra package needed)."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ----------------------------------------------------------------------
def compute_metrics(team, baseline=None):
    a = ASSUMPTIONS
    manual_minutes = team["weeds_removed"] * a["manual_minutes_per_weed"]
    manual_cost = manual_minutes / 60 * a["labour_rate_pkr_per_hour"]
    crop_loss = team["crop_touches"] * a["crop_loss_pkr_per_touch"]

    metrics = {
        "manual_labour_minutes_equivalent": round(manual_minutes, 1),
        "manual_labour_cost_pkr_equivalent": round(manual_cost, 1),
        "estimated_crop_loss_pkr": round(crop_loss, 1),
        "baseline_single_robot_steps": None,
        "time_saved_vs_single_robot_pct": None,
        "assumptions": a,
    }
    if baseline and baseline["completed"] and team["completed"] and baseline["steps"] > 0:
        metrics["baseline_single_robot_steps"] = baseline["steps"]
        metrics["time_saved_vs_single_robot_pct"] = round(
            100 * (baseline["steps"] - team["steps"]) / baseline["steps"], 1)
    return metrics


# ----------------------------------------------------------------------
def _prompt(team, metrics, controller):
    data = {"controller": controller, "team_result": team, "business_metrics": metrics}
    return (
        "You are an agriculture operations analyst. A team of 2 autonomous weeding "
        "robots (simulation) just finished a mission on a fixed field with 16 crop "
        "plants and 12 weeds. Write a short Farm Report for the farm owner from the "
        "JSON data below.\n\n"
        "Rules:\n"
        "- Use only the numbers in the JSON. Do not invent any figures.\n"
        "- The business metrics are based on ASSUMPTIONS (listed in the JSON); say "
        "clearly that they are estimates.\n"
        "- If time_saved_vs_single_robot_pct is null, do not mention it.\n"
        "- team_result.supervisor describes the Supervisor Agent (a third robot that watches the "
        "workers and corrects them). Mention how many corrections it made and why. If "
        "fault_injected is true, say clearly that a fault was deliberately injected as a test.\n"
        "- Use simple markdown: '##' headings and '-' bullets, no tables.\n"
        "- Sections: Mission Summary, Robot Performance, Crop Safety, Business Impact, "
        "Recommendations (3 bullets).\n"
        "- Keep it under 250 words in simple English.\n"
        "- End with a 2-line summary in Roman Urdu under the heading '## Khulasa'.\n\n"
        f"DATA:\n{json.dumps(data, indent=2)}"
    )


def _call_gemini(prompt, model, key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4},
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.load(resp)
    parts = data["candidates"][0]["content"]["parts"]
    text = "".join(part.get("text", "") for part in parts).strip()
    if not text:
        raise ValueError("empty response")
    return text


def _template_report(team, metrics):
    lines = [
        "## Mission Summary",
        f"- Weeds removed: {team['weeds_removed']}/{team['total_weeds']} in {team['steps']} steps.",
        f"- Mission completed: {'yes' if team['completed'] else 'no'}.",
        "## Robot Performance",
    ]
    for r in team["per_robot"]:
        lines.append(f"- {r['robot']}: {r['weeds_removed']} weeds, {r['crop_touches']} crop touches.")
    if metrics["time_saved_vs_single_robot_pct"] is not None:
        lines.append(f"- Two robots were {metrics['time_saved_vs_single_robot_pct']}% faster than one robot.")
    lines += [
        "## Supervisor",
        f"- Supervisor corrections: {team.get('supervisor', {}).get('interventions', 0)}"
        + (" (a fault was deliberately injected as a test)."
           if team.get("supervisor", {}).get("fault_injected") else "."),
        "## Crop Safety",
        f"- Total crop touches: {team['crop_touches']}.",
        "## Business Impact (estimates based on assumptions)",
        f"- Equivalent manual labour: {metrics['manual_labour_minutes_equivalent']} min "
        f"(about PKR {metrics['manual_labour_cost_pkr_equivalent']}).",
        f"- Estimated crop loss: PKR {metrics['estimated_crop_loss_pkr']}.",
        "## Note",
        "- This report was generated from a template because the Gemini API was not available.",
    ]
    return "\n".join(lines)


def generate_report(team, metrics, controller):
    """Returns (report_text, source) where source is the Gemini model or 'template'."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("GEMINI_API_KEY not set -> using template report.")
        return _template_report(team, metrics), "template"

    models = GEMINI_MODELS[:]
    if os.environ.get("GEMINI_MODEL"):
        models.insert(0, os.environ["GEMINI_MODEL"])
    prompt = _prompt(team, metrics, controller)
    for m in models:
        try:
            text = _call_gemini(prompt, m, key)
            print(f"Report written by {m}")
            return text, m
        except urllib.error.HTTPError as e:
            print(f"Gemini model {m} failed: HTTP {e.code}")
        except Exception as e:
            print(f"Gemini model {m} failed: {e}")
    print("All Gemini models failed -> using template report.")
    return _template_report(team, metrics), "template"


# ----------------------------------------------------------------------
def decide_alerts(team, report_text):
    alerts = []
    ok = team["completed"]
    alerts.append({
        "type": "mission_report",
        "subject": (f"Weeding mission complete: {team['weeds_removed']}/{team['total_weeds']} weeds removed"
                    if ok else
                    f"Weeding mission finished: {team['weeds_removed']}/{team['total_weeds']} weeds removed"),
        "body": report_text,
    })
    if team["crop_touches"] > 0:
        detail = "\n".join(f"- {r['robot']}: {r['crop_touches']} touch(es)"
                           for r in team["per_robot"] if r["crop_touches"] > 0)
        alerts.append({
            "type": "crop_damage",
            "subject": f"URGENT: crop damaged ({team['crop_touches']} touches)",
            "body": ("Crop damage detected during the weeding mission.\n\n"
                     f"{detail}\n\nPlease inspect the affected crop rows."),
        })
    if team["weeds_left"] > 0:
        alerts.append({
            "type": "mission_incomplete",
            "subject": f"Mission incomplete: {team['weeds_left']} weeds left",
            "body": (f"The robots used all {team['steps']} steps but {team['weeds_left']} of "
                     f"{team['total_weeds']} weeds are still in the field.\n"
                     "Another run or manual weeding is needed."),
        })
    return alerts


def send_email(subject, body):
    addr = os.environ.get("GMAIL_ADDRESS")
    pwd = os.environ.get("GMAIL_APP_PASSWORD")
    to = os.environ.get("ALERT_TO") or addr
    if not (addr and pwd):
        return False, "GMAIL_ADDRESS / GMAIL_APP_PASSWORD not set"
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = addr, to, subject
    msg.set_content(body)
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as s:
            s.login(addr, pwd.replace(" ", ""))
            s.send_message(msg)
        return True, f"sent to {to}"
    except Exception as e:
        return False, str(e)