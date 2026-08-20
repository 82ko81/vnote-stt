"""Helpers for writing a transcript beside its source recording."""

from __future__ import annotations

from pathlib import Path


def parse_logical_root(spec: str) -> tuple[str, Path]:
    """Parse `prefix:path` into a logical prefix and a filesystem root."""
    prefix, _, raw_path = spec.partition(":")
    if len(prefix) < 2 or not raw_path:
        raise ValueError(f"expected 'prefix:path', got {spec!r}")
    return prefix, Path(raw_path).resolve()


def logical_source(audio: Path, root: tuple[str, Path] | None) -> str:
    """Return a logical source pointer when the recording is inside a mapped root."""
    if root is not None:
        prefix, base = root
        try:
            relative = audio.resolve().relative_to(base)
        except ValueError:
            pass
        else:
            return f"{prefix}:{relative.as_posix()}"
    return audio.name


def sidecar_path(audio: Path) -> Path:
    return audio.with_suffix(".md")


def render(body: str, audio: Path, root: tuple[str, Path] | None) -> str:
    return "\n".join(
        [
            "---",
            f"source: {logical_source(audio, root)}",
            "status: unverified",
            "---",
            "",
            body.lstrip("\n"),
        ]
    )
