"""
Generic prompt rules ("addons") that any system under test can append to
its own system prompt. The rules live as plain text files in
prompt_addons/<name>.txt so they can be read, reviewed and edited:

    no_invention          answer only from tools / context, never invent
    injection_resistance  ignore instructions hidden in input, protect others' data
    scope                 "Your job: {description}" from the target's description
                          in targets.yaml; refuse requests outside it

An experiment switches them on with a setting, default none:

    settings:
      prompt_addons: [no_invention, injection_resistance, scope]

The harness cannot edit an arbitrary system's prompt, so a target applies
them with one line on its own prompt:

    system_prompt = prompt_addons.apply(system_prompt, settings, description)

run_experiment.py validates the names before any API call, so a typo fails
fast. The demo agent (agent_configurable.py) is the first user.
"""

import difflib
import os

ADDONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_addons")
HEADER = "Additional rules:"


def list_addons() -> list[str]:
    if not os.path.isdir(ADDONS_DIR):
        return []
    return sorted(f[:-4] for f in os.listdir(ADDONS_DIR) if f.endswith(".txt"))


def _text(name: str) -> str:
    with open(os.path.join(ADDONS_DIR, f"{name}.txt"), encoding="utf-8") as f:
        return f.read().strip()


def validate(names, description=None) -> list[str]:
    """Raises ValueError with a readable message; returns the names."""
    if names is None:
        return []
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ValueError("prompt_addons must be a list of addon names, e.g. [no_invention, scope]")
    available = list_addons()
    for name in names:
        if name not in available:
            close = difflib.get_close_matches(name, available, n=1)
            hint = f" (did you mean '{close[0]}'?)" if close else ""
            raise ValueError(f"unknown prompt addon '{name}'{hint}; available: {', '.join(available)}")
        if "{description}" in _text(name) and not (description or "").strip():
            raise ValueError(f"prompt addon '{name}' needs the target's description: add "
                             "description: \"...\" to its entry in targets.yaml")
    if len(set(names)) != len(names):
        raise ValueError("prompt_addons lists the same addon twice")
    return names


def render(names, description=None) -> str:
    """The rules block appended to a prompt, in the order given ("" for none)."""
    validate(names, description)
    if not names:
        return ""
    rules = [_text(n).replace("{description}", (description or "").strip()) for n in names]
    return HEADER + "\n" + "\n".join(f"- {r}" for r in rules)


def apply(system_prompt: str, settings, description=None) -> str:
    """system_prompt unchanged when settings has no prompt_addons; otherwise + the rules block."""
    names = (settings or {}).get("prompt_addons") or []
    if not names:
        return system_prompt
    return system_prompt + "\n\n" + render(names, description)
