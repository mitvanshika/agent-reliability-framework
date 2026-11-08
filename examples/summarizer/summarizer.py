"""
Example system under test: a deliberately simple, deterministic summarizer.
No LLM, no tools: it returns the first two sentences of the text. It is a
stand-in for YOUR system - replace summarize() with a call to your model and
nothing else in the framework has to change.

Plugged in through targets.yaml (entry `summarizer`), see the README section
"Plug in your system in 4 steps".
"""

import re

PREFIX = re.compile(r"^\s*summari[sz]e\s*:\s*", re.IGNORECASE)


def summarize(text: str) -> str:
    """The first two sentences of the text after an optional 'Summarise:' prefix."""
    body = PREFIX.sub("", text or "").strip()
    if not body:
        return "There is no text to summarise."
    sentences = re.split(r"(?<=[.!?])\s+", body)
    return " ".join(sentences[:2])
