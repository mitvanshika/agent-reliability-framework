
import json
import os

from trace_utils import load_latest_traces

STRONG_FOLDER = "judged_traces"
WEAK_FOLDER = "judged_traces_weak"
OUTPUT_FILE = "dashboard_data.json"


def load_judged_traces(folder: str) -> dict:
    if not os.path.isdir(folder):
        return {}
    # Latest run per scenario only (see trace_utils.py).
    return load_latest_traces(folder)


def main():
    strong = load_judged_traces(STRONG_FOLDER)
    weak = load_judged_traces(WEAK_FOLDER)

    bundle = {
        "strong": strong,
        "weak": weak,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(bundle, f, indent=2)

    print(f"Exported {len(strong)} strong traces and {len(weak)} weak traces to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()