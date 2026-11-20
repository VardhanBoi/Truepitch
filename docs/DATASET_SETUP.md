# Dataset setup

## Verified so far (Nov 2026, from the papers and the official MUStARD README)
| Item | MUStARD | MUStARD++ |
|---|---|---|
| Source | https://github.com/soujanyaporia/MUStARD | https://github.com/apoorva-nunna/MUStARD_Plus_Plus (URL from the LREC 2022 paper; README NOT yet inspected) |
| Size | 690 clips, balanced (345/345), avg ~5 s | 1202 clips, balanced (601/601) per secondary sources |
| Annotations | sarcasm label, utterance text, speaker, context | + emotion, valence, arousal, sarcasm type |
| Non-sarcastic source | subset of MELD (Friends) | not yet verified |
| License | MIT (repo LICENSE) | NOT VERIFIED: check repo before use |
| Audio | not shipped separately; extract from video | not yet verified |

## Open verification items (do before Phase 3)
1. Read the MUStARD++ README: video hosting, access conditions, license, label scales (valence/arousal range).
2. Confirm whether a `show` field exists in sarcasm_data.json.
3. After download: run `python scripts/make_splits.py` and record the printed validation report here.

## MUStARD download
- Annotations: `data/sarcasm_data.json` from the GitHub repo -> `data/mustard/sarcasm_data.json`
- Raw clips: the README links a Hugging Face zip (`MichiganNLP/MUStARD`, `mmsd_raw_data.zip`). Unzip under `data/mustard/`.
- Update `paths.mustard_video_dir` in `configs/default.yaml` to match the real folder layout.

## Audio extraction (16 kHz mono wav)
Install ffmpeg (Windows: `winget install Gyan.FFmpeg`; macOS: `brew install ffmpeg`; Ubuntu: `sudo apt install ffmpeg`), then:

    python scripts/extract_audio.py --video-root data/mustard

The script prints how many annotated ids were matched to a video and lists any missing.
