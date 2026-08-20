"""Optional speaker diarization via sherpa-onnx."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from vnote.transcribe import SAMPLE_RATE, read_mono_16k
from vnote.writers import Segment, Word

MODEL_DIR = Path.home() / ".cache" / "sherpa-onnx"
SEGMENTATION_MODEL = MODEL_DIR / "sherpa-onnx-pyannote-segmentation-3-0" / "model.int8.onnx"
EMBEDDING_MODEL = MODEL_DIR / "campplus.onnx"
EMBED_FLOOR_SECONDS = 3.0
DEFAULT_THREADS = min(8, os.cpu_count() or 4)
DIARIZE_OVERHEAD = 1.05
DEFAULT_CLUSTER_THRESHOLD = 0.9


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str


def models_available() -> bool:
    return SEGMENTATION_MODEL.exists() and EMBEDDING_MODEL.exists()


def diarize(
    audio: Path,
    num_speakers: int = -1,
    threads: int = DEFAULT_THREADS,
    threshold: float | None = None,
    transcript: list[Segment] | None = None,
    samples=None,
) -> list[SpeakerTurn]:
    """Return speaker turns for a recording or an existing ASR transcript."""
    import sherpa_onnx

    if threshold is None:
        threshold = float(os.environ.get("VNOTE_DIARIZE_THRESHOLD", DEFAULT_CLUSTER_THRESHOLD))
    if not 0 < threshold < 1:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")

    if transcript:
        return _diarize_transcript(
            audio, transcript, num_speakers, threshold, threads, samples
        )

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(SEGMENTATION_MODEL)
            ),
            num_threads=threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(EMBEDDING_MODEL), num_threads=threads
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers, threshold=threshold
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise RuntimeError("sherpa-onnx rejected the diarization configuration")

    engine = sherpa_onnx.OfflineSpeakerDiarization(config)
    if samples is None or engine.sample_rate != SAMPLE_RATE:
        samples = read_mono_16k(audio, engine.sample_rate)
    result = engine.process(samples).sort_by_start_time()

    return [
        SpeakerTurn(start=r.start, end=r.end, speaker=f"Speaker {r.speaker + 1}")
        for r in result
    ]


def _widen(start: float, end: float, total: float, floor: float = EMBED_FLOOR_SECONDS) -> tuple[float, float]:
    """Expand a short ASR segment around its midpoint for a more stable embedding."""
    if end - start >= floor or total <= floor:
        return start, end
    middle = (start + end) / 2
    low = max(0.0, middle - floor / 2)
    high = min(total, low + floor)
    return max(0.0, high - floor), high


def _diarize_transcript(
    audio: Path,
    segments: list[Segment],
    num_speakers: int,
    threshold: float,
    threads: int,
    samples=None,
) -> list[SpeakerTurn]:
    """Cluster speaker embeddings on ASR-aligned speech segments."""
    import numpy as np
    import sherpa_onnx

    if samples is None:
        samples = read_mono_16k(audio, SAMPLE_RATE)
    total = len(samples) / SAMPLE_RATE

    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(EMBEDDING_MODEL), num_threads=threads
        )
    )

    valid: list[tuple[Segment, np.ndarray]] = []
    for segment in segments:
        window_start, window_end = _widen(segment.start, segment.end, total)
        start = max(0, int(window_start * SAMPLE_RATE))
        end = min(len(samples), int(window_end * SAMPLE_RATE))
        if end <= start:
            continue

        stream = extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples[start:end])
        stream.input_finished()
        if not extractor.is_ready(stream):
            continue

        embedding = np.asarray(extractor.compute(stream), dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            valid.append((segment, embedding / norm))

    if not valid:
        return []

    embeddings = np.stack([embedding for _, embedding in valid])
    clustering = sherpa_onnx.FastClustering(
        sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers, threshold=threshold
        )
    )
    labels = clustering(embeddings)
    return [
        SpeakerTurn(
            start=segment.start,
            end=segment.end,
            speaker=f"Speaker {label + 1}",
        )
        for (segment, _), label in zip(valid, labels)
    ]


def assign_speakers(segments: list[Segment], turns: list[SpeakerTurn]) -> list[Segment]:
    """Attach speaker labels to Whisper segments, splitting at word level when possible."""
    if not turns:
        return segments

    labelled: list[Segment] = []
    for seg in segments:
        if seg.words:
            labelled.extend(_split_words(seg, turns))
            continue

        overlaps: dict[str, float] = {}
        for turn in turns:
            overlap = min(seg.end, turn.end) - max(seg.start, turn.start)
            if overlap > 0:
                overlaps[turn.speaker] = overlaps.get(turn.speaker, 0.0) + overlap

        best = max(overlaps, key=overlaps.__getitem__) if overlaps else _nearest_bounded_speaker(seg, turns)
        labelled.append(Segment(start=seg.start, end=seg.end, text=seg.text, speaker=best))
    return labelled


def _split_words(segment: Segment, turns: list[SpeakerTurn]) -> list[Segment]:
    labelled_words: list[tuple[Word, str | None]] = []
    for word in segment.words or []:
        overlaps: dict[str, float] = {}
        for turn in turns:
            overlap = min(word.end, turn.end) - max(word.start, turn.start)
            if overlap > 0:
                overlaps[turn.speaker] = overlaps.get(turn.speaker, 0.0) + overlap
        speaker = (
            max(overlaps, key=overlaps.__getitem__)
            if overlaps
            else _nearest_bounded_speaker(Segment(word.start, word.end, word.text), turns)
        )
        labelled_words.append((word, speaker))

    groups: list[Segment] = []
    for word, speaker in labelled_words:
        if groups and groups[-1].speaker == speaker:
            groups[-1].end = word.end
            groups[-1].text += word.text
        else:
            groups.append(
                Segment(start=word.start, end=word.end, text=word.text, speaker=speaker)
            )
    for group in groups:
        group.text = group.text.strip()
    return groups


def _nearest_bounded_speaker(segment: Segment, turns: list[SpeakerTurn]) -> str | None:
    """Fill only an internal diarization gap; leave edge gaps unlabeled."""
    before = any(turn.end <= segment.start for turn in turns)
    after = any(turn.start >= segment.end for turn in turns)
    if not (before and after):
        return None

    midpoint = (segment.start + segment.end) / 2
    nearest = min(turns, key=lambda turn: abs((turn.start + turn.end) / 2 - midpoint))
    return nearest.speaker
