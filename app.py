"""
Small backend that holds the Groq API key privately (via environment
variable / Replit Secret) and proxies requests from the frontend.
The browser never sees the key.

Also serves /api/data, which reads judged_traces/ and
judged_traces_weak/ directly off disk on every request — no more
manually running export_for_dashboard.py and re-uploading a JSON
file every time the results change. Reload the page, get current
data.

Run locally:
    python store.py               (creates + seeds the database, once)
    set GROQ_API_KEY in .env
    python app.py

On Replit:
    1. Create a Python Repl, upload the project
    2. Go to the "Secrets" tab (lock icon) and add: GROQ_API_KEY = your key
    3. Click Run — Replit gives you a public URL
"""

import json
import os

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

app = Flask(__name__)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

STRONG_FOLDER = "judged_traces"
WEAK_FOLDER = "judged_traces_weak"


def load_judged_traces(folder: str) -> dict:
    results = {}
    if not os.path.isdir(folder):
        return results
    for filename in os.listdir(folder):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(folder, filename)) as f:
            trace = json.load(f)
        scenario_id = trace["scenario_id"]
        results[scenario_id] = trace
    return results


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    strong = load_judged_traces(STRONG_FOLDER)
    weak = load_judged_traces(WEAK_FOLDER)
    return jsonify({"strong": strong, "weak": weak})


@app.route("/api/ask", methods=["POST"])
def ask():
    if not GROQ_API_KEY:
        return jsonify({"error": "Server is missing GROQ_API_KEY. Set it as a Secret/env var."}), 500

    payload = request.get_json(force=True) or {}
    system_prompt = payload.get("system", "")
    question = payload.get("question", "")

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openai/gpt-oss-120b",
                "max_tokens": 600,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
            },
            timeout=30,
        )
        return jsonify(resp.json()), resp.status_code
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)