"""
Loads experiment settings files (experiments/<name>.yaml).

An experiment describes one version of the system being tested:

    name: exp_01_weak_prompt            # must equal the file name
    description: "Same as baseline but with the weak system prompt"
    extends: baseline                   # optional: experiments/baseline.yaml
    target: configurable                # a name in targets.yaml
    settings:                           # passed to the target untouched
      prompt: prompts/weak.txt

"extends" starts from the parent experiment and overrides only what this
file sets. Nested dicts in settings merge key by key; anything else
(strings, numbers, lists) is replaced. Chains are allowed. name and
description are never inherited; target is, unless overridden.

The harness deliberately does NOT look inside settings: each target
validates its own (see agent_configurable.py), so any plugged-in system
can define whatever settings it needs.

load_experiment() also returns "changes_vs_parent", e.g.
    ["settings.prompt: prompts/v1.txt -> prompts/weak.txt"]
which reports show as "What changed".
"""

import copy
import difflib
import hashlib
import json
import os
import re

import yaml

TOP_LEVEL_KEYS = ("name", "description", "extends", "target", "settings")
NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")   # used in folder names, so keep it simple
_UNSET = object()


class ExperimentConfigError(ValueError):
    """A problem in an experiment file, with a message meant for a human."""


def _read(path: str) -> dict:
    if not os.path.isfile(path):
        raise ExperimentConfigError(f"Experiment file not found: {path}")
    with open(path, encoding="utf-8") as f:
        try:
            raw = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ExperimentConfigError(f"{path} is not valid YAML: {e}") from e
    fname = os.path.basename(path)
    if not isinstance(raw, dict):
        raise ExperimentConfigError(f"{fname} must be a mapping of keys (name, target, settings, ...).")

    for key in raw:
        if key not in TOP_LEVEL_KEYS:
            close = difflib.get_close_matches(str(key), TOP_LEVEL_KEYS, n=1)
            hint = f" (did you mean '{close[0]}'?)" if close else ""
            raise ExperimentConfigError(
                f"Unknown key '{key}' in {fname}{hint}. Allowed: {', '.join(TOP_LEVEL_KEYS)}.")

    name = raw.get("name")
    if not isinstance(name, str) or not NAME_PATTERN.match(name):
        raise ExperimentConfigError(
            f"{fname}: 'name' is required and may only contain letters, digits, '_', '-', '.'.")
    stem = os.path.splitext(fname)[0]
    if name != stem:
        raise ExperimentConfigError(
            f"{fname}: name '{name}' must equal the file name '{stem}', so that 'extends: {stem}', "
            f"runs/{stem}_... and experiments/{fname} always refer to the same experiment.")
    for key in ("description", "extends", "target"):
        if raw.get(key) is not None and not isinstance(raw[key], str):
            raise ExperimentConfigError(f"{fname}: '{key}' must be a string.")
    if raw.get("settings") is not None and not isinstance(raw["settings"], dict):
        raise ExperimentConfigError(f"{fname}: 'settings' must be a mapping.")
    return raw


def deep_merge(base: dict, override: dict) -> dict:
    """New dict: base with override applied. Dicts merge recursively, anything else replaces."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _flatten(value, prefix: str) -> dict:
    if isinstance(value, dict) and value:
        out = {}
        for k, v in value.items():
            out.update(_flatten(v, f"{prefix}.{k}"))
        return out
    return {prefix: value}


def _fmt(value) -> str:
    if value is _UNSET:
        return "(unset)"
    return value if isinstance(value, str) else json.dumps(value)


def diff_configs(parent: dict, child: dict) -> list[str]:
    """Human-readable differences in target/settings, one dotted path per line."""
    before = {**_flatten(parent.get("target"), "target"), **_flatten(parent.get("settings", {}), "settings")}
    after = {**_flatten(child.get("target"), "target"), **_flatten(child.get("settings", {}), "settings")}
    changes = []
    for path in sorted(set(before) | set(after)):
        old, new = before.get(path, _UNSET), after.get(path, _UNSET)
        if old != new or (old is _UNSET) != (new is _UNSET):
            changes.append(f"{path}: {_fmt(old)} -> {_fmt(new)}")
    return changes


def _resolve(path: str, seen: list[str]) -> dict:
    raw = _read(path)
    name = raw["name"]
    if name in seen:
        raise ExperimentConfigError("Cycle in 'extends': " + " -> ".join(seen + [name]))

    parent_name = raw.get("extends")
    if parent_name:
        parent_name = parent_name[:-5] if parent_name.endswith(".yaml") else parent_name
        parent_path = os.path.join(os.path.dirname(path), f"{parent_name}.yaml")
        if not os.path.isfile(parent_path):
            raise ExperimentConfigError(
                f"{name} extends '{parent_name}', but {parent_path} does not exist.")
        parent = _resolve(parent_path, seen + [name])
        base = {"target": parent["target"], "settings": parent["settings"]}
        chain = parent["chain"] + [name]
    else:
        parent = None
        base = {"target": None, "settings": {}}
        chain = [name]

    resolved = {
        "name": name,
        "description": raw.get("description") or "",
        "file": path,
        "parent": parent["name"] if parent else None,
        "chain": chain,
        "target": raw["target"] if raw.get("target") else base["target"],
        "settings": deep_merge(base["settings"], raw.get("settings") or {}),
    }
    resolved["changes_vs_parent"] = diff_configs(base, resolved) if parent else []
    return resolved


def load_experiment(path: str) -> dict:
    """Fully resolved experiment:
    {name, description, file, parent, chain, target, settings, changes_vs_parent}."""
    resolved = _resolve(path, [])
    if not resolved["target"]:
        raise ExperimentConfigError(
            f"{resolved['name']}: no 'target' set (set it here or in a parent experiment).")
    return resolved


def config_hash(resolved: dict) -> str:
    """Short fingerprint of what actually runs (target + settings), e.g. to tell
    whether a noise calibration still describes the current baseline."""
    blob = json.dumps({"target": resolved.get("target"), "settings": resolved.get("settings") or {}},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def format_changes(resolved: dict) -> str:
    """One-line "What changed" text for reports."""
    if resolved["parent"] is None:
        return "(baseline: no parent)"
    if not resolved["changes_vs_parent"]:
        return f"(nothing vs {resolved['parent']})"
    return "; ".join(resolved["changes_vs_parent"]) + f"   (vs {resolved['parent']})"
