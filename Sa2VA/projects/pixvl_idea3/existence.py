"""Shared target-existence answer parsing for training and evaluation."""

from __future__ import annotations

import re


_NO_TARGET_PATTERNS = (
    re.compile(r"^\s*no(?:\s|[.,:;!?]|$)", re.IGNORECASE),
    re.compile(r"\bno\s+target\b", re.IGNORECASE),
    re.compile(r"\b(?:does\s+not|doesn't|is\s+not|isn't)\s+(?:exist|present|visible)\b", re.IGNORECASE),
    re.compile(r"\bnot\s+present\b", re.IGNORECASE),
    re.compile(r"\btarget\s+(?:is\s+)?(?:absent|missing)\b", re.IGNORECASE),
)


def predicts_target_exists(text: str) -> bool:
    """Return False for explicit abstention/no-target answers."""

    normalized = str(text).strip()
    return not any(pattern.search(normalized) for pattern in _NO_TARGET_PATTERNS)
