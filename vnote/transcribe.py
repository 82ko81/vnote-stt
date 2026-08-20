"""Speech-to-text via faster-whisper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vnote.terms import TermDictionary
from vnote.writers import Segment, Word

DEFAULT_MODEL = "large-v3-turbo"
SAMPLE_RATE = 16000
AUDIO_SUFFIXES = {".m4a", ".mp3", ".wav", ".mp4", ".aac", ".flac", ".ogg", ".m4v", ".mov"}

# Measured on a real five-minute meeting sample. These values are estimates for
# progress messaging only and are intentionally conservative.
REALTIME_FACTOR = 1.55
FAST_BATCH_SIZE = 4
FAST_REALTIME_FACTOR = 2.72


def find_recordings(target: Path) -> list[Path]:
    """Return one recording or every supported recording directly inside a folder."""
    if target.is_file():
        return [target]
    return sorted(p for p in target.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES)


def probe_duration(path: Path) -> float:
    """Return container duration in seconds without decoding the full stream."""
    import av

    try:
        with av.open(str(path)) as container:
            if container.duration is None:
                return 0.0
            return container.duration / av.time_base
    except Exception:
        return 0.0


def read_mono_16k(audio: str | Path, target_rate: int = SAMPLE_RATE):
    """Decode audio to the mono float samples used by transcription and diarization."""
    from faster_whisper.audio import decode_audio

    return decode_audio(str(audio), sampling_rate=target_rate)


@dataclass
class TranscriptionResult:
    segments: list[Segment]
    duration: float
    samples: object | None = None


def load_model(model_size: str = DEFAULT_MODEL, compute_type: str = "int8"):
    """Load a CTranslate2 Whisper model on CPU."""
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device="cpu", compute_type=compute_type)


def transcribe(
    audio: str | Path,
    model,
    terms: TermDictionary | None = None,
    language: str = "ko",
    batch_size: int = 0,
) -> TranscriptionResult:
    """Transcribe a recording, optionally using faster batched decoding."""
    samples = read_mono_16k(audio)
    engine = model
    options = {}
    if batch_size:
        from faster_whisper import BatchedInferencePipeline

        engine = BatchedInferencePipeline(model=model)
        options["batch_size"] = batch_size

    segments, info = engine.transcribe(
        samples,
        language=language,
        vad_filter=True,
        initial_prompt=terms.initial_prompt() if terms else None,
        beam_size=5,
        word_timestamps=True,
        **options,
    )

    out: list[Segment] = []
    for seg in segments:
        text = seg.text.strip()
        if terms:
            text = terms.correct(text)
        words = [
            Word(start=word.start, end=word.end, text=word.word)
            for word in (seg.words or [])
        ]
        if terms:
            words = _correct_words(words, terms)
        out.append(Segment(start=seg.start, end=seg.end, text=text, words=words))

    return TranscriptionResult(segments=out, duration=info.duration, samples=samples)


def _correct_words(words: list[Word], terms: TermDictionary) -> list[Word]:
    """Apply terminology correction to word timestamps without losing merged spans."""
    if not words:
        return []
    corrected = terms.correct_tokens([word.text for word in words])
    return [
        Word(start=words[first].start, end=words[last].end, text=text)
        for first, last, text in corrected
    ]
