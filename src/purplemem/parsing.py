from __future__ import annotations

import re


_POSITIVE_PREFERENCE_VERBS = {
    "likes", "loves", "prefers", "enjoys",
    "is a fan of", "is into", "is obsessed with", "is passionate about",
}
_NEGATIVE_PREFERENCE_VERBS = {
    "hates", "dislikes", "avoids", "can't stand",
    "doesn't like", "is not a fan of", "despises",
}


def normalize_username(username: str | None) -> str | None:
    if username is None:
        return None
    return username.strip().lower()


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def expand_query_variants(query: str) -> list[str]:
    normalized_query = normalize_text(query)
    if not normalized_query:
        return []
    return [query, normalized_query]


_PREFERENCE_PATTERNS = [
    (
        re.compile(
            r"^(?P<subject>[a-z0-9_']+)\s+(?P<verb>likes|loves|prefers|enjoys|hates|dislikes|avoids)\s+(?P<object>.+)$"
        ),
        None,
    ),
    (
        re.compile(
            r"^(?P<subject>[a-z0-9_']+)\s+(?:doesn't|does\s+not|doesnt)\s+(?:like|enjoy|prefer)\s+(?P<object>.+)$"
        ),
        -1,
    ),
    (
        re.compile(
            r"^(?P<subject>[a-z0-9_']+)\s+(?:can't\s+stand|cant\s+stand|cannot\s+stand)\s+(?P<object>.+)$"
        ),
        -1,
    ),
    (
        re.compile(
            r"^(?P<subject>[a-z0-9_']+)\s+is\s+(?:a\s+fan\s+of|into|obsessed\s+with|passionate\s+about)\s+(?P<object>.+)$"
        ),
        1,
    ),
    (
        re.compile(
            r"^(?P<subject>[a-z0-9_']+)\s+is\s+not\s+a\s+fan\s+of\s+(?P<object>.+)$"
        ),
        -1,
    ),
    (
        re.compile(r"^(?P<subject>[a-z0-9_']+)\s+(?P<verb>despises)\s+(?P<object>.+)$"),
        -1,
    ),
]


def parse_preference_memory(
    text: str, default_username: str | None = None
) -> dict | None:
    normalized = normalize_text(text)
    for pattern, forced_polarity in _PREFERENCE_PATTERNS:
        match = pattern.match(normalized)
        if not match:
            continue
        subject = normalize_username(match.group("subject"))
        object_text = match.group("object").strip(" .,!?:;\"'")
        if not object_text:
            continue
        try:
            predicate = match.group("verb")
        except IndexError:
            predicate = "prefers" if forced_polarity == 1 else "dislikes"
        if forced_polarity is not None:
            polarity = forced_polarity
        else:
            polarity = 1 if predicate in _POSITIVE_PREFERENCE_VERBS else -1
        if default_username and subject != normalize_username(default_username):
            return None
        return {
            "subject": subject,
            "predicate": predicate,
            "object_text": object_text,
            "polarity": polarity,
            "memory_type": "preference",
        }
    return None


_NATURE_MARKERS: dict[str, list[re.Pattern]] = {
    "decision": [
        re.compile(p, re.I)
        for p in [
            r"\blet'?s (use|go with|try|pick|choose|switch to)\b",
            r"\bwe (should|decided|chose|went with|picked|settled on)\b",
            r"\binstead of\b",
            r"\btrade-?off\b",
            r"\bdecided (to|on|against)\b",
            r"\bbecause\b.*\b(better|faster|simpler|easier|safer|cleaner)\b",
        ]
    ],
    "milestone": [
        re.compile(p, re.I)
        for p in [
            r"\bit works\b",
            r"\bit worked\b",
            r"\bfixed\b",
            r"\bsolved\b",
            r"\bbreakthrough\b",
            r"\bfinally\b",
            r"\bfirst time\b",
            r"\bbuilt\b",
            r"\bshipped\b",
            r"\bdeployed\b",
        ]
    ],
    "problem": [
        re.compile(p, re.I)
        for p in [
            r"\b(bug|error|crash|fail|broke|broken|issue|problem)\b",
            r"\bdoesn'?t work\b",
            r"\bnot working\b",
            r"\broot cause\b",
            r"\bworkaround\b",
            r"\bpatched\b",
        ]
    ],
    "emotional": [
        re.compile(p, re.I)
        for p in [
            r"\bscared\b",
            r"\bproud\b",
            r"\bhappy\b",
            r"\bsad\b",
            r"\bgrateful\b",
            r"\bworried\b",
            r"\bi feel\b",
            r"\bi miss\b",
            r"\bstressed\b",
            r"\bexcited\b",
        ]
    ],
}


def classify_memory_nature(text: str) -> str | None:
    if not text or len(text) < 15:
        return None
    best_type = None
    best_hits = 0
    for nature_type, patterns in _NATURE_MARKERS.items():
        hits = sum(1 for p in patterns if p.search(text))
        if hits > best_hits:
            best_hits = hits
            best_type = nature_type
    return best_type if best_hits >= 2 else None


_FILLER_WORDS = re.compile(
    r"\b(um|uh|uhh|umm|you know|basically|literally|honestly|i mean|sort of|kind of|i guess|right so)\b",
    re.IGNORECASE,
)
_COMMON_ABBREVS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bi think that\b", re.I), "imo"),
    (re.compile(r"\bin my opinion\b", re.I), "imo"),
    (re.compile(r"\bto be honest\b", re.I), "tbh"),
    (re.compile(r"\bas far as i know\b", re.I), "afaik"),
    (re.compile(r"\bin real life\b", re.I), "irl"),
    (re.compile(r"\bby the way\b", re.I), "btw"),
]
_REPEATED_PUNCT = re.compile(r"([!?.]){2,}")
_EMOTE_NOISE = re.compile(
    r"\b(KEKW|LULW|OMEGALUL|PogChamp|Kappa|monkaS|COPIUM|HOPIUM|PepeHands|FeelsGoodMan|FeelsBadMan|catJAM|EZ|Clap|GIGACHAD|Sadge|widepeepoHappy|peepoClap)\b"
)


def compress_memory_text(text: str) -> str:
    if not text:
        return text
    result = _EMOTE_NOISE.sub("", text)
    result = _FILLER_WORDS.sub("", result)
    for pattern, replacement in _COMMON_ABBREVS:
        result = pattern.sub(replacement, result)
    result = _REPEATED_PUNCT.sub(r"\1", result)
    result = re.sub(r"\s+", " ", result).strip()
    return result
