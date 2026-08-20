"""Command-line interface for local transcription."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from vnote.diarize import DIARIZE_OVERHEAD, assign_speakers, diarize, models_available
from vnote.sidecar import parse_logical_root, render
from vnote.terms import TermDictionary
from vnote.transcribe import (
    DEFAULT_MODEL,
    FAST_BATCH_SIZE,
    FAST_REALTIME_FACTOR,
    REALTIME_FACTOR,
    find_recordings,
    load_model,
    probe_duration,
    transcribe,
)
from vnote.writers import to_markdown, to_srt, to_txt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vnote-stt",
        description="Local speech-to-text for Korean meetings and interviews.",
    )
    parser.add_argument("target", type=Path, help="a recording or a folder of recordings")
    parser.add_argument("--out-dir", type=Path, default=None, help="output directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="faster-whisper model name")
    parser.add_argument("--terms", type=Path, default=None, help="JSON terminology dictionary")
    parser.add_argument("--language", default="ko", help="Whisper language code (default: ko)")
    parser.add_argument(
        "--sidecar",
        action="store_true",
        help="write <stem>.md beside each recording instead of Markdown/TXT/SRT into --out-dir",
    )
    parser.add_argument(
        "--logical-root",
        default=None,
        metavar="PREFIX:PATH",
        help="render the sidecar source pointer as PREFIX:<path relative to PATH>",
    )
    parser.add_argument(
        "--speakers",
        action="store_true",
        help="label passages by speaker; install the speakers extra and run vnote-models first",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="use batched decoding; faster, but it may omit or merge speech",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=-1,
        help="expected speaker count; -1 lets clustering decide",
    )
    args = parser.parse_args(argv)

    if not args.target.exists():
        print(f"No such path: {args.target}", file=sys.stderr)
        return 1

    if args.speakers and not models_available():
        print(
            "Speaker models are missing. Run: pip install -e '.[speakers]' && vnote-models",
            file=sys.stderr,
        )
        return 1

    root = parse_logical_root(args.logical_root) if args.logical_root else None
    recordings = find_recordings(args.target)
    if not recordings:
        print(f"No supported recordings found in {args.target}", file=sys.stderr)
        return 1

    pending = [p for p in recordings if not _output_exists(p, args, recordings)]
    if not pending:
        print(f"All {len(recordings)} recording(s) are already transcribed.")
        return 0

    total_audio = sum(probe_duration(p) for p in pending)
    estimate = total_audio / (FAST_REALTIME_FACTOR if args.fast else REALTIME_FACTOR)
    if args.speakers:
        estimate *= DIARIZE_OVERHEAD

    if args.fast:
        print("Fast mode enabled: batched decoding may omit or merge speech.")
    print(f"{len(pending)} recording(s), {_hms(total_audio)} of audio")
    print(f"Estimated runtime on the reference CPU: {_hms(estimate)}+", flush=True)
    if len(recordings) != len(pending):
        print(f"Skipping {len(recordings) - len(pending)} existing transcript(s)")

    terms = TermDictionary.load(args.terms) if args.terms else None
    print(f"Loading {args.model} ...", flush=True)
    model = load_model(args.model)

    failed = 0
    started = time.perf_counter()
    for index, audio in enumerate(pending, start=1):
        print(f"[{index}/{len(pending)}] {audio.name}", flush=True)
        try:
            _transcribe_one(audio, model, terms, root, args, recordings)
        except Exception as exc:
            failed += 1
            print(f"  Failed: {exc}", file=sys.stderr, flush=True)

    print(
        f"Done: {len(pending) - failed}/{len(pending)} in {_hms(time.perf_counter() - started)}"
        + (f", {failed} failed" if failed else "")
    )
    return 1 if failed else 0


def _output_stem(audio: Path, recordings: list[Path]) -> str:
    rivals = [
        candidate
        for candidate in recordings
        if candidate != audio and candidate.stem.casefold() == audio.stem.casefold()
    ]
    if not rivals:
        return audio.stem

    ordered = sorted([audio, *rivals], key=lambda p: (p.stat().st_mtime, p.name))
    if ordered[0] == audio:
        return audio.stem

    base = f"{audio.stem}-{audio.suffix.lstrip('.').lower()}"
    position = ordered.index(audio)
    same_extension_before = sum(
        p.suffix.casefold() == audio.suffix.casefold()
        for p in ordered[1:position]
    )
    counter = same_extension_before + 1
    candidate = base if counter == 1 else f"{base}-{counter}"
    used_stems = {p.stem.casefold() for p in recordings}
    while candidate.casefold() in used_stems:
        counter += 1
        candidate = f"{base}-{counter}"
    return candidate


def _output_exists(
    audio: Path,
    args: argparse.Namespace,
    recordings: list[Path] | None = None,
) -> bool:
    recordings = recordings or find_recordings(audio.parent)
    stem = _output_stem(audio, recordings)
    if args.sidecar:
        return (audio.parent / f"{stem}.md").exists()
    out_dir = args.out_dir or audio.parent
    return all((out_dir / f"{stem}{suffix}").exists() for suffix in (".md", ".txt", ".srt"))


def _transcribe_one(
    audio: Path,
    model,
    terms,
    root,
    args: argparse.Namespace,
    recordings: list[Path] | None = None,
) -> None:
    result = transcribe(
        audio,
        model,
        terms=terms,
        language=args.language,
        batch_size=FAST_BATCH_SIZE if args.fast else 0,
    )
    segments = result.segments

    if args.speakers:
        segments = assign_speakers(
            segments,
            diarize(
                audio,
                num_speakers=args.num_speakers,
                transcript=segments,
                samples=result.samples,
            ),
        )

    recordings = recordings or find_recordings(audio.parent)
    stem = _output_stem(audio, recordings)
    markdown = to_markdown(segments, title=stem, source=audio.name)

    if args.sidecar:
        target = audio.parent / f"{stem}.md"
        if not target.exists():
            target.write_text(render(markdown, audio, root), encoding="utf-8")
        return

    out_dir = args.out_dir or audio.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix, body in {
        ".md": markdown,
        ".txt": to_txt(segments),
        ".srt": to_srt(segments),
    }.items():
        target = out_dir / f"{stem}{suffix}"
        if not target.exists():
            target.write_text(body, encoding="utf-8")


def _hms(seconds: float) -> str:
    total = int(seconds)
    if total >= 3600:
        return f"{total // 3600}h {total // 60 % 60}m"
    if total >= 60:
        return f"{total // 60}m {total % 60}s"
    return f"{total}s"


if __name__ == "__main__":
    raise SystemExit(main())
