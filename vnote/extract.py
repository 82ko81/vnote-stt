"""Extract likely proper nouns and technical terms from Korean-heavy text."""

from __future__ import annotations

import re
from collections import Counter

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9-]*|[가-힣]+")
_PARTICLES = (
    "으로부터", "에서부터", "으로는", "에게서", "으로", "까지", "부터", "께서", "처럼",
    "보다", "에게", "에서", "로", "만큼", "이랑", "랑", "은", "는", "이", "가", "을",
    "를", "과", "와", "도", "의", "에", "만",
)
_COMMON_KOREAN = {
    "그리고", "그러나", "대한", "대해", "때문", "내용", "문서", "보고서", "사항", "자료",
    "회의", "결과", "목적", "방법", "부분", "설명", "검토", "관련", "추진", "진행", "사업",
    "기관", "정도", "경우", "이후", "이전", "현재", "예정", "가능", "필요", "사용", "위해",
    "통해", "대상", "방안", "계획", "확인", "제공", "운영", "경영",
}
_COMMON_ABBREVIATIONS = {
    "ASCII", "CSV", "DOCX", "HWP", "HTML", "JSON", "MD", "PDF", "PPTX", "RTF", "TXT", "URL", "XML",
}


def extract_terms(text: str, top_n: int = 50) -> list[str]:
    """Return frequent candidate terms, ranked for terminology-prompt use."""
    if top_n <= 0 or not text:
        return []

    tokens = [match.group(0) for match in _TOKEN.finditer(text)]
    counts: Counter[str] = Counter()
    for index, token in enumerate(tokens):
        candidate = _normalize_korean(token)
        if _is_korean_candidate(candidate):
            counts[candidate] += 1
        elif _is_abbreviation(token):
            counts[token] += 1

        if (
            _is_abbreviation(token)
            and index + 1 < len(tokens)
            and _is_mixed_korean_part(_normalize_korean(tokens[index + 1]))
        ):
            counts[f"{token} {_normalize_korean(tokens[index + 1])}"] += 1

    ranked = sorted(
        counts.items(),
        key=lambda item: (-item[1], -len(item[0].replace(" ", "")), item[0]),
    )
    return [term for term, _ in ranked[:top_n]]


def merge_terms(existing: dict, extracted: list[str]) -> dict:
    """Merge extracted candidates into an existing terminology dictionary."""
    old_terms = existing.get("terms", {}) if isinstance(existing, dict) else {}
    merged = (
        {str(term): variants for term, variants in old_terms.items()}
        if isinstance(old_terms, dict)
        else {}
    )
    for term in extracted:
        merged.setdefault(term, [])
    return {"terms": merged}


def _normalize_korean(token: str) -> str:
    if not token or not all("가" <= char <= "힣" for char in token):
        return token
    for particle in _PARTICLES:
        if token.endswith(particle) and len(token) - len(particle) >= 2:
            return token[: -len(particle)]
    return token


def _is_korean_candidate(token: str) -> bool:
    return (
        len(token) >= 2
        and _is_korean_word(token)
        and token not in _COMMON_KOREAN
        and not token.endswith("다")
    )


def _is_mixed_korean_part(token: str) -> bool:
    return len(token) >= 2 and _is_korean_word(token) and not token.endswith("다")


def _is_korean_word(token: str) -> bool:
    return all("가" <= char <= "힣" for char in token)


def _is_abbreviation(token: str) -> bool:
    return (
        len(token) >= 2
        and token.upper() == token
        and any("A" <= char <= "Z" for char in token)
        and token not in _COMMON_ABBREVIATIONS
        and not token.isdigit()
    )
