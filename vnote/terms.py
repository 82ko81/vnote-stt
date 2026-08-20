"""Terminology dictionary for Whisper prompt biasing and post-correction."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

PROMPT_CHAR_BUDGET = 400


@dataclass
class Term:
    canonical: str
    variants: list[str] = field(default_factory=list)


class TermDictionary:
    def __init__(self, terms: list[Term]):
        self._terms = terms
        self._rules = _build_rules(terms)
        self._combined_rule = (
            re.compile(
                "|".join(
                    f"(?P<rule_{index}>{pattern.pattern})"
                    for index, (pattern, _) in enumerate(self._rules)
                ),
                re.IGNORECASE,
            )
            if self._rules
            else None
        )

    @classmethod
    def load(cls, path: str | Path) -> "TermDictionary":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_obj(data)

    @classmethod
    def from_obj(cls, data: dict) -> "TermDictionary":
        terms = [
            Term(canonical=canonical, variants=list(variants or []))
            for canonical, variants in data.get("terms", {}).items()
        ]
        return cls(terms)

    @property
    def terms(self) -> list[Term]:
        return list(self._terms)

    def initial_prompt(self, budget: int = PROMPT_CHAR_BUDGET) -> str:
        """Return canonical spellings, truncated to a conservative prompt budget."""
        out: list[str] = []
        used = 0
        for term in self._terms:
            cost = len(term.canonical) + (2 if out else 0)
            if used + cost > budget:
                break
            out.append(term.canonical)
            used += cost
        return ", ".join(out)

    def correct(self, text: str) -> str:
        """Replace known ASR variants with canonical spellings."""
        if self._combined_rule is None:
            return text

        def replace(match: re.Match[str]) -> str:
            for index, (_, canonical) in enumerate(self._rules):
                if match.group(f"rule_{index}") is not None:
                    return _preserve_padding(match.group(0), canonical)
            raise RuntimeError("term rule matched without a replacement")

        return self._combined_rule.sub(replace, text)

    def correct_tokens(self, tokens: list[str]) -> list[tuple[int, int, str]]:
        """Correct tokenized text while preserving the token span each replacement consumed."""
        if self._combined_rule is None or not tokens:
            return [(index, index, token) for index, token in enumerate(tokens)]

        bounds: list[tuple[int, int]] = []
        cursor = 0
        for token in tokens:
            bounds.append((cursor, cursor + len(token)))
            cursor += len(token)
        joined = "".join(tokens)

        reach = list(range(len(tokens)))
        for match in self._combined_rule.finditer(joined):
            first = next(i for i, (_, end) in enumerate(bounds) if end > match.start())
            last = max(i for i, (start, _) in enumerate(bounds) if start < match.end())
            reach[first] = max(reach[first], last)

        out: list[tuple[int, int, str]] = []
        index = 0
        while index < len(tokens):
            last = reach[index]
            probe = index
            while probe <= last:
                last = max(last, reach[probe])
                probe += 1
            out.append((index, last, self.correct(joined[bounds[index][0] : bounds[last][1]])))
            index = last + 1
        return out


def _build_rules(terms: list[Term]) -> list[tuple[re.Pattern[str], str]]:
    rules: list[tuple[str, str]] = []
    for term in terms:
        for variant in term.variants:
            if variant.strip() and variant != term.canonical:
                rules.append((variant, term.canonical))
    rules.sort(key=lambda pair: len(pair[0]), reverse=True)
    return [(_compile_variant(variant), canonical) for variant, canonical in rules]


def _compile_variant(variant: str) -> re.Pattern[str]:
    """Match a variant while tolerating arbitrary whitespace between characters."""
    skeleton = [ch for ch in variant if not ch.isspace()]
    body = r"\s*".join(re.escape(ch) for ch in skeleton)
    if skeleton and _needs_boundary(skeleton[0]):
        body = r"\b" + body
    if skeleton and _needs_boundary(skeleton[-1]):
        body += r"\b"
    return re.compile(body, re.IGNORECASE)


def _is_latin(ch: str) -> bool:
    return "LATIN" in unicodedata.name(ch, "")


def _needs_boundary(ch: str) -> bool:
    return _is_latin(ch) or ch.isdigit()


def _preserve_padding(matched: str, canonical: str) -> str:
    lead = matched[: len(matched) - len(matched.lstrip())]
    trail = matched[len(matched.rstrip()) :]
    return f"{lead}{canonical}{trail}"
