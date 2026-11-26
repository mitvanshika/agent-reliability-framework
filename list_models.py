"""
List the models your Groq key can use. Also a free key check: listing models costs no tokens.
    python list_models.py              # uses GROQ_API_KEY from .env
    python list_models.py path/to/.env # check another .env (e.g. a plugged-in system's own key)
"""

import os
import sys

import requests
from dotenv import load_dotenv

URL = "https://api.groq.com/openai/v1/models"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    env_file = argv[0] if argv else ".env"
    from_file = os.path.isfile(env_file)
    load_dotenv(env_file)
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        print(f"No GROQ_API_KEY found (looked in {env_file} and the environment).")
        return 1
    # Never print the key; its start and length are enough to tell which one is loaded.
    print(f"Key from {env_file if from_file else 'the environment'}: starts with {api_key[:4]}..., {len(api_key)} characters")
    try:
        response = requests.get(URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
    except requests.RequestException as e:
        print(f"Couldn't reach Groq: {type(e).__name__}: {e}")
        return 1
    try:
        body = response.json()
    except ValueError:
        body = {}
    if response.status_code != 200 or "data" not in body:
        err = body.get("error") if isinstance(body, dict) else None
        msg = err.get("message") if isinstance(err, dict) else (response.text[:300] or "no details")
        hint = {401: "the key is wrong, revoked or has a typo (check for quotes or spaces in .env)",
                403: "the key isn't allowed to do this",
                429: "rate limit hit; wait a minute and try again"}.get(response.status_code, "")
        print(f"Groq said {response.status_code}: {msg}" + (f"\n  -> {hint}" if hint else ""))
        return 1
    model_ids = sorted(m["id"] for m in body["data"])
    print(f"The key works: {len(model_ids)} models\n")
    for model_id in model_ids:
        print(model_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
