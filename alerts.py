"""
Sends a Slack alert via an incoming webhook when a regression is
detected. This is what turns "the build silently went red" into
"someone on the team actually gets pinged."

Setup:
    1. In Slack: create an Incoming Webhook for the channel you want
       alerts in (Slack -> Apps -> search "Incoming Webhooks" -> Add
       to Slack -> choose channel -> copy the webhook URL).
    2. Add it to your .env file:
       SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

If SLACK_WEBHOOK_URL isn't set, alerts are skipped silently (printed
to console instead) — this is optional, not required for the
pipeline to run.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")


def send_slack_alert(text: str):
    if not SLACK_WEBHOOK_URL:
        print("[alerts] SLACK_WEBHOOK_URL not set — skipping Slack alert. Message would have been:")
        print(text)
        return

    try:
        response = requests.post(
            SLACK_WEBHOOK_URL,
            json={"text": text},
            timeout=10,
        )
        if response.status_code == 200:
            print("[alerts] Slack alert sent.")
        else:
            print(f"[alerts] Slack alert failed: {response.status_code} {response.text}")
    except requests.RequestException as e:
        print(f"[alerts] Slack alert failed: {e}")


def format_regression_alert(strong_pass: int, strong_total: int, weak_pass: int, weak_total: int, p_value: float, u_pvalue: float = None) -> str:
    lines = [
        f":rotating_light: *Agent reliability regression detected*",
        f"Strong agent pass rate: {strong_pass}/{strong_total}",
        f"Weak agent pass rate: {weak_pass}/{weak_total}",
        f"Fisher's exact test p-value: {p_value:.4f} (significant at p < 0.05)",
    ]
    if u_pvalue is not None:
        lines.append(f"Mann-Whitney U test (raw scores) p-value: {u_pvalue:.4f}")
    lines.append("Review before shipping this change.")
    return "\n".join(lines)