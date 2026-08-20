"""Output formatters. Transcript segments in, text out."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list[Word] | None = None


def to_txt(segments: list[Segment]) -> str:
    return "\n".join(seg.text.strip() for seg in segments if seg.text.strip()) + "\n"


def to_markdown(segments: list[Segment], title: str, source: str) -> str:
    lines = [f"# {title}", "", f"- source: `{source}`", ""]
    last_speaker = object()
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        if seg.speaker != last_speaker:
            lines.append("")
            header = f"**{seg.speaker}**" if seg.speaker else "**Unknown speaker**"
            lines.append(f"{header} · `{_clock(seg.start)}`")
            lines.append("")
            last_speaker = seg.speaker
        lines.append(text)
    return "\n".join(lines).strip() + "\n"


def to_srt(segments: list[Segment]) -> str:
    blocks = []
    index = 0
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        index += 1
        if seg.speaker:
            text = f"[{seg.speaker}] {text}"
        blocks.append(f"{index}\n{_srt_time(seg.start)} --> {_srt_time(seg.end)}\n{text}\n")
    return "\n".join(blocks)


def _clock(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{total // 60 % 60:02d}:{total % 60:02d}"


def _srt_time(seconds: float) -> str:
    millis = round(seconds * 1000)
    ms = millis % 1000
    total = millis // 1000
    return f"{total // 3600:02d}:{total // 60 % 60:02d}:{total % 60:02d},{ms:03d}"
