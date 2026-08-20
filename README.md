# vnote-stt

Local Korean speech-to-text with terminology correction, speaker diarization, and Markdown/TXT/SRT output.

`vnote-stt` is a CPU-friendly transcription pipeline built around `faster-whisper`. It is designed for Korean meetings and interviews where names, organizations, technical terms, and speaker changes matter more than producing a pretty but incomplete transcript.

## Features

- Local transcription with `faster-whisper` / CTranslate2
- Korean-first defaults with configurable language
- Terminology dictionary that biases Whisper before decoding and repairs known ASR variants afterward
- Optional speaker diarization with `sherpa-onnx`
- Markdown, plain-text, and SRT output
- Word-level term correction preserved through speaker diarization
- Folder mode with collision-safe output names and skip-existing behavior

## Install

```bash
pip install -e .
```

For speaker diarization:

```bash
pip install -e '.[speakers]'
python -m vnote.models
```

The speaker-model command downloads model files from the upstream `sherpa-onnx` release assets into `~/.cache/sherpa-onnx`. Model weights are **not** distributed by this repository and may have licensing terms separate from this code. Review the upstream model terms before redistribution or commercial packaging.

## Usage

```bash
vnote-stt meeting.m4a
```

This writes `meeting.md`, `meeting.txt`, and `meeting.srt` beside the recording.

```bash
vnote-stt recordings/ --out-dir transcripts/
```

With speaker labels:

```bash
vnote-stt meeting.m4a --speakers
```

With a terminology dictionary:

```bash
vnote-stt meeting.m4a --terms terms.json
```

Example `terms.json`:

```json
{
  "terms": {
    "Antigravity": ["anti gravity", "antigravity"],
    "KOSIS": ["Kosis", "K O S I S"]
  }
}
```

## Accuracy vs. speed

The default decoder favors transcript completeness. `--fast` enables batched decoding and can be substantially faster, but it may omit or merge speech. For meeting records where missing a sentence matters, keep the default mode.

## License

MIT. Third-party libraries and downloaded model weights retain their own licenses; see `THIRD_PARTY_NOTICES.md`.
