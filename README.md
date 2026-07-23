# Forensic Chat Screenshot Extractor

A Python toolkit for extracting chat conversations from screenshots or screenshot collages and converting them into normalized CSV transcripts.

The pipeline supports:

- Facebook Messenger screenshots and screenshot collages
- Viber screenshots and screenshot collages
- ZIP package input for integration workflows
- folder input for local development
- single-image extraction
- automatic platform routing
- recursive image discovery
- automatic case-report discovery inside a package
- skipping non-chat or unknown images
- chronological merging into one CSV transcript per package/folder

The final transcript format is:

```csv
"Time","Sender","Receiver","Message"
"DD/MM/YYYY HH:MM","Sender Name","Receiver Name","Message text"
```

## Overview

This project contains three main components:

1. **Shared extraction utilities**  
   Common OCR, image splitting, report parsing, timestamp normalization, CSV handling, conservative text cleaning, and final merged-output post-processing.

2. **Platform-specific extractors**  
   Separate extraction pipelines for Facebook Messenger and Viber screenshots.

3. **Batch orchestration**  
   A package/folder processing script that discovers inputs, classifies screenshots, runs the correct extractor, skips irrelevant images, and writes one merged CSV transcript.

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

Check that the expected packages are available from the active environment:

```bash
python3 -c "import cv2, easyocr, ollama, PyPDF2, numpy; print('ok')"
```

## Project Structure

```text
forensic-chat-screenshot-extractor/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── mass_chat_extract.py          # Batch/package orchestration script
├── facebook_extract.py           # Facebook Messenger extraction pipeline
├── viber_extract.py              # Viber extraction pipeline
└── extractor_utils.py            # Shared helper functions
```

Keep these files in the same directory:

```text
facebook_extract.py
viber_extract.py
extractor_utils.py
mass_chat_extract.py
```

The platform extractors import shared helper functions from `extractor_utils.py`.

## Input Data

The primary integration input is a **ZIP package** containing:

- a PDF or TXT case report / case overview
- one or more screenshot images or screenshot collages
- Facebook Messenger and/or Viber screenshots

Local development can also use an extracted folder.

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

### Recommended package mode

Process one ZIP package and create one merged CSV transcript:

```bash
python3 mass_chat_extract.py case_name.zip
```

The script will recursively search the package, find a case report when available, detect relevant chat screenshots, skip non-chat images, and create a merged output.

Expected default output layout:

```text
results/
├── _extracted_zips/
└── merged/
    ├── <package_stem>_extracted_chat.csv
    └── <package_stem>_extracted_chat.csv.manifest.json
```

By default, per-image CSV/debug folders are created in a temporary workspace and removed after the merged CSV is produced.

### Package mode with explicit case report

Use this when the package contains multiple possible reports or you want to force a specific report:

```bash
python3 mass_chat_extract.py \
  case_name.zip \
  --case-report case_reports/case_report.pdf
```

### Folder mode

Process an extracted folder instead of a ZIP:

```bash
python3 mass_chat_extract.py case_name/
```

### Legacy two-argument mode

Process a known case report and a known image folder:

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

## Single-Image Extraction

### Facebook Messenger

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

### Viber

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

## Platform Classification

`mass_chat_extract.py` supports automatic routing to the correct extractor.

| Mode       | Behavior                                                                       |
| ---------- | ------------------------------------------------------------------------------ |
| `auto`     | Uses filename/path hints first, then falls back to the vision model if needed. |
| `filename` | Uses only filename/path keywords such as `facebook`, `messenger`, or `viber`.  |
| `vision`   | Uses the vision model first, then falls back to filename/path hints.           |

Example using filename-only classification:

```bash
python3 mass_chat_extract.py \
  case_name.zip \
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
| `--case-report`       | Override automatic report discovery in package/folder mode.                            |
| `--output`            | Output CSV path for a single extractor, or merged CSV path for `mass_chat_extract.py`. |
| `--results-dir`       | Root output folder for batch extraction.                                               |
| `--debug-dir`         | Debug folder for a single extractor.                                                   |
| `--debug`             | Keep debug artifacts during batch extraction.                                          |
| `--keep-per-image`    | Persist per-image CSV/debug folders under `results/per_image/`.                        |
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

> **Note:** `--emoji-mode omit` is recommended for most runs. Emoji recognition is experimental, and the extractor may miss, misread, or hallucinate emojis when using `--emoji-mode vision`.

Manual collage example:

```bash
python3 facebook_extract.py \
  images/facebook/collage.png \
  case_reports/case_report.pdf \
  --layout 2,3 \
  --output results/per_image/collage_extracted.csv \
  --debug-dir results/per_image/collage_debug
```

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

Keep `extractor_utils.py` in the same folder as the two platform extractors and the batch script:

```text
facebook_extract.py
viber_extract.py
extractor_utils.py
mass_chat_extract.py
```

The child extractor commands printed by `mass_chat_extract.py` should use the same Python executable as the active environment.

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
- For faster iteration, process one screenshot first before running full package extraction.
- For public repositories, keep input data and generated outputs outside Git tracking.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
