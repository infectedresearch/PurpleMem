from __future__ import annotations

import re


_TEMPORAL_PATTERNS: list[tuple[re.Pattern, float, float]] = [
    (re.compile(r"\brecently\b", re.I), 3, 7),
    (re.compile(r"\bjust now\b|\bjust\b.*\bsaid\b", re.I), 0.5, 1),
    (re.compile(r"\byesterday\b", re.I), 1, 2),
    (re.compile(r"\ba few days ago\b|\bcouple days ago\b", re.I), 3, 4),
    (re.compile(r"\blast week\b|\ba week ago\b", re.I), 7, 5),
    (re.compile(r"\b(\d+)\s+days?\s+ago\b", re.I), -1, -1),
    (re.compile(r"\b(\d+)\s+weeks?\s+ago\b", re.I), -7, -1),
    (re.compile(r"\blast month\b|\ba month ago\b", re.I), 30, 15),
    (re.compile(r"\b(\d+)\s+months?\s+ago\b", re.I), -30, -1),
    (re.compile(r"\ba while ago\b|\bsome time ago\b", re.I), 14, 14),
    (re.compile(r"\blast year\b|\ba year ago\b", re.I), 365, 60),
]


def parse_temporal_reference(query: str) -> tuple[float, float] | None:
    if not query:
        return None
    for pattern, base_offset, base_window in _TEMPORAL_PATTERNS:
        match = pattern.search(query)
        if match:
            if base_offset < 0:
                n = int(match.group(1))
                days_offset = n * abs(base_offset)
                window_days = max(2, days_offset * 0.3)
            else:
                days_offset = base_offset
                window_days = base_window
            return (days_offset, window_days)
    return None
