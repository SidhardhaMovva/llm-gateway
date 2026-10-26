"""
Prompt injection detection: simple pattern-based scan for common injection attempts.
"""
import re
from typing import Optional

# Patterns that often indicate prompt injection attempts (simplified)
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions",
    r"disregard\s+(your\s+|the\s+)?(instructions|system\s+prompt)",
    r"you\s+are\s+now\s+",
    r"pretend\s+you\s+are\s+",
    r"act\s+as\s+if\s+you\s+are\s+",
    r"\[system\]",
    r"<\|im_start\|>",
    r"human:\s*$",
    r"assistant:\s*$",
    r"###\s*system\s*:",
]

# MULTILINE so line-anchored patterns (``human:\s*$``, ``assistant:\s*$``)
# also match on lines inside a multi-line message, not only at the very
# end of the whole string.
COMPILED = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in INJECTION_PATTERNS]


def scan_prompt(text: Optional[str]) -> tuple[bool, Optional[str]]:
    """
    Returns (is_safe, matched_pattern_or_none).
    If any pattern matches, consider it potentially unsafe.
    """
    if not text or not text.strip():
        return True, None
    for pat in COMPILED:
        if pat.search(text):
            return False, pat.pattern
    return True, None


def scan_messages(messages: list[dict]) -> tuple[bool, Optional[str]]:
    """Scan all message contents in a conversation."""
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            safe, match = scan_prompt(content)
            if not safe:
                return False, match
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    safe, match = scan_prompt(part.get("text"))
                    if not safe:
                        return False, match
    return True, None


def sanitize_for_log(text: Optional[str], max_len: int = 500) -> str:
    """Truncate and strip for safe audit logging."""
    if not text:
        return ""
    s = (text or "").strip().replace("\n", " ")[:max_len]
    return s
