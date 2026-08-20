# Third-Party Notices

This repository contains original integration code and does not vendor third-party model weights.

## Runtime libraries

- **faster-whisper** — MIT License
- **sherpa-onnx** — Apache License 2.0
- **NumPy** — BSD-3-Clause License

These projects remain subject to their own license terms.

## Downloaded model weights

Optional speaker diarization downloads model artifacts from the upstream `k2-fsa/sherpa-onnx` GitHub release assets. Those files are cached locally under `~/.cache/sherpa-onnx` and are not included in this repository.

Pretrained model artifacts can have terms that differ from the source-code license of the framework that distributes them. Users are responsible for reviewing the applicable upstream model terms before redistributing model files or packaging them for commercial use.

The default Whisper model is downloaded through `faster-whisper` / Hugging Face infrastructure on first use and is likewise not included in this repository.
