# Forensic Chat Screenshot Extractor

A Python toolkit for extracting chat conversations from screenshot images or screenshot collages and converting them into normalized CSV transcripts.

The pipeline supports:

- Facebook Messenger screenshots and collages
- Viber screenshots and collages
- single-image extraction
- batch extraction over folders of images
- automatic platform routing
- chronological merging of extracted CSV files

The final transcript format is:

```csv
"Time","Sender","Receiver","Message"
"DD/MM/YYYY HH:MM","Sender Name","Receiver Name","Message text"
```

## Overview

This project contains three main components:

1. **Shared extraction utilities**  
   Common OCR, image splitting, report parsing, CSV handling, timestamp normalization, and text-cleaning helpers.

2. **Platform-specific extractors**  
   Separate extraction pipelines for Facebook Messenger and Viber screenshots.

3. **Batch orchestration**  
   A batch script that classifies screenshots, runs the correct extractor, and merges all extracted CSV files into one transcript.

## Prerequisites

- Python 3.10 or newer
- Ollama installed and available on `PATH`
- An Ollama vision-capable model, for example `gemma3:12b`
- GPU recommended for faster OCR and vision-model inference

Install or pull the default model:

```bash
ollama pull gemma3:12b
```

Check that Ollama is available:

```bash
ollama list
```

If the Ollama server is not running:

```bash
ollama serve
```

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```text
forensic-chat-screenshot-extractor/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── mass_chat_extract.py          # Batch/orchestration script
├── facebook_extract.py           # Facebook Messenger extraction pipeline
├── viber_extract.py              # Viber extraction pipeline
├── extractor_utils.py            # Shared helper functions
└── .github/
    └── workflows/
        └── python-check.yml
```

Keep these files in the same directory:

```text
facebook_extract.py
viber_extract.py
extractor_utils.py
```

The platform extractors import shared helper functions from `extractor_utils.py`.

## Input Data

The scripts expect:

- a PDF or TXT case report
- one or more screenshot images or screenshot collages
- screenshots from Facebook Messenger or Viber

Supported image extensions:

```text
.png, .jpg, .jpeg, .webp, .bmp, .tif, .tiff
```

The case report is used to infer chat participants and map left/right chat sides to real sender/receiver names.

## Output Format

All final CSV transcripts use this header:

```csv
"Time","Sender","Receiver","Message"
```

Timestamp format:

```text
DD/MM/YYYY HH:MM
```

Example:

```csv
"Time","Sender","Receiver","Message"
"12/03/2026 10:15","Alice Example","Bob Example","Hello Bob."
"12/03/2026 10:17","Bob Example","Alice Example","Hi Alice."
```

## Quick Start

### Facebook Messenger single-image extraction

```bash
python3 facebook_extract.py \
  images/facebook/example_messenger_chat.png \
  case_reports/case_report.pdf \
  --model gemma3:12b \
  --langs en \
  --emoji-mode omit \
  --output results/per_image/example_messenger_chat_extracted.csv \
  --debug-dir results/per_image/example_messenger_chat_debug
```

### Viber single-image extraction

```bash
python3 viber_extract.py \
  images/viber/example_viber_chat.png \
  case_reports/case_report.pdf \
  --model gemma3:12b \
  --langs en \
  --emoji-mode omit \
  --output results/per_image/example_viber_chat_extracted.csv \
  --debug-dir results/per_image/example_viber_chat_debug
```

### Batch extraction

Use `mass_chat_extract.py` to process many screenshots and automatically route each image to the correct extractor.

```bash
python3 mass_chat_extract.py \
  case_reports/case_report.pdf \
  images/ \
  --results-dir results \
  --model gemma3:12b \
  --langs en \
  --classify-mode auto \
  --emoji-mode omit
```

Expected output layout:

```text
results/
├── per_image/
│   ├── <image_stem>_extracted.csv
│   └── <image_stem>_debug/
└── merged/
    ├── <case_report_stem>_merged_chats.csv
    └── <case_report_stem>_merged_chats.csv.manifest.json
```

## Platform Classification

`mass_chat_extract.py` supports three classification modes:

| Mode       | Behavior                                                                       |
| ---------- | ------------------------------------------------------------------------------ |
| `auto`     | Uses filename/path hints first, then falls back to the vision model if needed. |
| `filename` | Uses only filename/path keywords such as `facebook`, `messenger`, or `viber`.  |
| `vision`   | Uses the vision model first, then falls back to filename/path hints.           |

Example using filename-only classification:

```bash
python3 mass_chat_extract.py \
  case_reports/case_report.pdf \
  images/ \
  --classify-mode filename
```

You can also force all images to one extractor.

Force Facebook Messenger extraction:

```bash
python3 mass_chat_extract.py \
  case_reports/case_report.pdf \
  images/facebook/ \
  --force-platform facebook
```

Force Viber extraction:

```bash
python3 mass_chat_extract.py \
  case_reports/case_report.pdf \
  images/viber/ \
  --force-platform viber
```

## Useful Flags

| Flag                  | Purpose                                                                                |
| --------------------- | -------------------------------------------------------------------------------------- |
| `--output`            | Output CSV path for a single extractor, or merged CSV path for `mass_chat_extract.py`. |
| `--results-dir`       | Root output folder for batch extraction.                                               |
| `--debug-dir`         | Debug folder for a single extractor.                                                   |
| `--dump-ocr`          | Save OCR text and block positions.                                                     |
| `--dump-draft`        | Save intermediate model outputs.                                                       |
| `--dump-side-map`     | Save inferred left/right speaker mapping.                                              |
| `--cpu`               | Force CPU mode for EasyOCR.                                                            |
| `--no-vision`         | Use OCR text only instead of image+OCR vision prompting.                               |
| `--emoji-mode omit`   | Recommended mode. Removes emojis from the final output.                                |
| `--emoji-mode vision` | Attempts to keep only clearly visible emojis according to the vision model.            |
| `--grid`              | Manually split a collage into a fixed grid, for example `2x1`.                         |
| `--layout`            | Manually split uneven collage rows, for example `2,3`.                                 |
| `--classify-mode`     | Select platform classification strategy for batch extraction.                          |
| `--force-platform`    | Force all images to use the Facebook or Viber extractor.                               |

> **Note:** `--emoji-mode omit` is recommended for most runs. Emoji recognition is still experimental, and the extractor may miss, misread, or hallucinate emojis when using `--emoji-mode vision`.

Manual collage example:

```bash
python3 facebook_extract.py \
  images/facebook/collage.png \
  case_reports/case_report.pdf \
  --layout 2,3 \
  --output results/per_image/collage_extracted.csv \
  --debug-dir results/per_image/collage_debug
```

## Running on Shared or HPC Systems

For shared systems, do not run OCR, model inference, or batch extraction on a login node.

Use the login node only for editing, file transfer, environment setup, and job submission.

Submit extraction jobs through the scheduler used by your system, for example Slurm.

## Observed Results

In the final private test run, the pipeline produced the following approximate F1 scores:

| Dataset part                  | F1 score |
| ----------------------------- | -------: |
| Facebook Messenger set        |     0.32 |
| Viber set 1                   |     0.88 |
| Viber set 2                   |     0.71 |
| Overall strict score          |     0.54 |
| Overall timestamp-aware score |     0.90 |

The Facebook Messenger score is lower under strict timestamp matching because some Messenger screenshots show only screen-level or separator timestamps, not a visible timestamp for every individual message.

These numbers are dataset-specific and should not be treated as universal benchmark results.

## Troubleshooting

### No image files found

Check that the path exists and contains supported image files:

```bash
find images -type f
```

### Unknown platform

Rename files or folders so they include one of these words:

```text
facebook
messenger
viber
```

or use vision-based classification:

```bash
--classify-mode vision
```

You can also force a platform:

```bash
--force-platform facebook
```

or:

```bash
--force-platform viber
```

### Missing `extractor_utils.py`

Keep `extractor_utils.py` in the same folder as the two platform extractors:

```text
facebook_extract.py
viber_extract.py
extractor_utils.py
```

### Ollama connection error

Start the Ollama server:

```bash
ollama serve
```

Then check that models are available:

```bash
ollama list
```

### Wrong collage splitting

Use manual layout flags.

Fixed grid:

```bash
--grid 2x1
```

Uneven rows:

```bash
--layout 2,3
```

## Privacy Note

This tool is intended for local processing of screenshots and case reports.

Do not commit real screenshots, case reports, generated transcripts, debug folders, model logs, private phone numbers, usernames, IP addresses, or other sensitive data to a public repository.

Only publish synthetic, anonymized, or explicitly shareable examples.

## Notes

- This is a research/prototype pipeline.
- OCR and vision-model outputs may contain mistakes.
- Final CSV transcripts should be manually reviewed before being treated as final evidence.
- Runtime depends on image size, number of screenshots, OCR speed, GPU availability, and selected Ollama model.
- For faster iteration, process one screenshot first before running full batch extraction.
- For public repositories, keep input data and generated outputs outside Git tracking.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
