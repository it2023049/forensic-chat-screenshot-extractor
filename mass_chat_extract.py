"""Batch orchestrator for classifying chat screenshots, running platform extractors, and merging CSV outputs."""

import os

# Do not override Slurm's GPU assignment.
# Outside Slurm, keep the old local/default GPU selection if nothing is set.
if "SLURM_JOB_ID" not in os.environ:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

import argparse
import csv
import glob
import io
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set


IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"
}

# Output layout created by this script:
# results/
#   per_image/  -> one CSV and one debug folder per source image
#   merged/     -> final merged CSV and manifest JSON
PER_IMAGE_DIR_NAME = "per_image"
MERGED_DIR_NAME = "merged"
SHARED_UTILS_FILENAME = "extractor_utils.py"


# ============================================================
# PATH / INPUT HELPERS
# ============================================================

def expand_inputs(inputs: List[str]) -> List[Path]:
    """Expands file, folder, and glob inputs into a stable list of image paths."""
    images: List[Path] = []

    for raw in inputs:
        expanded = glob.glob(raw)

        if expanded:
            for item in expanded:
                path = Path(item)
                if path.is_dir():
                    images.extend(list_images_in_folder(path))
                elif is_image_file(path):
                    images.append(path)
            continue

        path = Path(raw)

        if path.is_dir():
            images.extend(list_images_in_folder(path))
        elif is_image_file(path):
            images.append(path)
        else:
            print(f"[WARN] Skipping non-image input: {raw}")

    # Stable de-duplication while preserving sorted path order.
    unique: Dict[str, Path] = {}
    for path in sorted(images, key=lambda p: str(p).lower()):
        unique[str(path.resolve())] = path

    return list(unique.values())


def list_images_in_folder(folder: Path) -> List[Path]:
    """Returns all supported image files found recursively under a folder."""
    return [
        p for p in sorted(folder.rglob("*"), key=lambda x: str(x).lower())
        if p.is_file() and is_image_file(p)
    ]


def is_image_file(path: Path) -> bool:
    """Checks whether a path is a supported image file."""
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def default_script_path(script_name: str) -> Path:
    """Returns the default path for an extractor script beside this file."""
    return Path(__file__).resolve().parent / script_name


def resolve_output_path(path_value: Optional[str], base_dir: Path, default_name: str) -> Path:
    """Resolves an output path under a base directory unless it is absolute."""
    if path_value is None:
        return base_dir / default_name

    path = Path(path_value)

    if path.is_absolute():
        return path

    return base_dir / path


def safe_output_stem(path: Path) -> str:
    """Creates a filesystem-safe filename stem for generated outputs."""
    stem = path.stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem or "image"


def unique_output_stem(path: Path, used_stems: Set[str]) -> str:
    """Creates a collision-free output stem for one input image."""
    base = safe_output_stem(path)
    stem = base
    counter = 2

    while stem.lower() in used_stems:
        stem = f"{base}_{counter:03d}"
        counter += 1

    used_stems.add(stem.lower())
    return stem


def ensure_exists(path: Path, label: str) -> None:
    """Raises a clear error when a required file or folder is missing."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def ensure_extractor_utils_available(script_path: Path) -> None:
    """Checks that extractor_utils.py is beside a refactored extractor script."""
    utils_path = script_path.resolve().parent / SHARED_UTILS_FILENAME

    if not utils_path.exists():
        raise FileNotFoundError(
            f"{SHARED_UTILS_FILENAME} not found next to {script_path}. "
            "Keep extractor_utils.py in the same folder as facebook_extract.py and viber_extract.py."
        )


# ============================================================
# PLATFORM CLASSIFICATION
# ============================================================

def normalize_platform(value: str) -> str:
    """Normalizes platform aliases to facebook, viber, or unknown."""
    text = str(value or "").strip().lower()

    if text in {
        "facebook",
        "messenger",
        "facebook_messenger",
        "facebook messenger",
        "fb",
    }:
        return "facebook"

    if text == "viber":
        return "viber"

    return "unknown"


def classify_by_filename(image_path: Path) -> str:
    """Classifies the platform using deterministic filename and folder hints."""
    text = str(image_path).lower().replace("\\", "/")

    facebook_keywords = [
        "facebook",
        "messenger",
        "fb_",
        "/fb/",
        "/facebook/",
        "/messenger/",
    ]

    viber_keywords = [
        "viber",
        "/viber/",
    ]

    if any(keyword in text for keyword in facebook_keywords):
        return "facebook"

    if any(keyword in text for keyword in viber_keywords):
        return "viber"

    return "unknown"


def classify_platform_with_vlm(
    image_path: Path,
    model: str,
    timeout_seconds: int = 120,
) -> str:
    """Uses a vision-capable Ollama model to classify the platform."""
    try:
        import ollama
    except Exception as exc:
        print(f"[WARN] Could not import ollama for VLM classification: {exc}")
        return "unknown"

    prompt = """
You are classifying a chat screenshot or collage.

Look only at the image UI and decide which app it is:
- Viber
- Facebook Messenger

Return only JSON with this exact schema:
{"platform":"viber"}
or
{"platform":"facebook"}
or
{"platform":"unknown"}

Use "facebook" for Facebook Messenger / Messenger.
Use "viber" for Viber.
If unsure, return "unknown".
Do not explain.
"""

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [str(image_path)],
                }
            ],
            options={
                "temperature": 0,
            },
        )

        text = response["message"]["content"].strip()
        data = extract_json_object(text)
        platform = normalize_platform(data.get("platform", ""))

        if platform in {"viber", "facebook"}:
            return platform

        # Small fallback if model returned text instead of JSON.
        low = text.lower()
        if "viber" in low:
            return "viber"
        if "facebook" in low or "messenger" in low:
            return "facebook"

        return "unknown"

    except Exception as exc:
        print(f"[WARN] VLM classification failed for {image_path.name}: {exc}")
        return "unknown"


def extract_json_object(text: str) -> Dict[str, str]:
    """Extracts and parses a JSON object from model text output."""
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        text = text[start:end + 1]

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    return {}


def classify_platform(
    image_path: Path,
    model: str,
    mode: str,
    force_platform: str = "auto",
) -> str:
    """Applies the selected platform classification strategy."""
    forced = normalize_platform(force_platform)
    if forced in {"facebook", "viber"}:
        return forced

    if mode == "filename":
        return classify_by_filename(image_path)

    if mode == "vision":
        platform = classify_platform_with_vlm(image_path, model)
        if platform == "unknown":
            platform = classify_by_filename(image_path)
        return platform

    # Default: auto.
    # First use deterministic filename/path rules.
    # If they fail, ask the vision model.
    platform = classify_by_filename(image_path)
    if platform != "unknown":
        return platform

    return classify_platform_with_vlm(image_path, model)



# ============================================================
# EXTRACTOR RUNNING
# ============================================================

def build_extractor_command(
    script_path: Path,
    image_path: Path,
    report_path: Path,
    model: str,
    langs: str,
    use_cpu: bool,
    no_vision: bool,
    emoji_mode: str,
    dump_ocr: bool,
    dump_draft: bool,
    dump_side_map: bool,
    output_csv_path: Path,
    debug_dir_path: Path,
    extra_args: List[str],
) -> List[str]:
    """Builds the subprocess command for one platform extractor run."""
    cmd = [
        sys.executable,
        str(script_path),
        str(image_path),
        str(report_path),
        "--model",
        model,
        "--langs",
        langs,
        "--emoji-mode",
        emoji_mode,
        "--output",
        str(output_csv_path),
        "--debug-dir",
        str(debug_dir_path),
    ]

    if use_cpu:
        cmd.append("--cpu")

    if no_vision:
        cmd.append("--no-vision")

    if dump_ocr:
        cmd.append("--dump-ocr")

    if dump_draft:
        cmd.append("--dump-draft")

    if dump_side_map:
        cmd.append("--dump-side-map")

    cmd.extend(extra_args)
    return cmd


def run_extractor(
    platform: str,
    image_path: Path,
    report_path: Path,
    viber_script: Path,
    facebook_script: Path,
    model: str,
    langs: str,
    use_cpu: bool,
    no_vision: bool,
    emoji_mode: str,
    dump_ocr: bool,
    dump_draft: bool,
    dump_side_map: bool,
    output_csv_path: Path,
    debug_dir_path: Path,
    extra_args: List[str],
) -> Optional[Path]:
    """Runs the selected extractor and returns the produced CSV path."""
    if platform == "viber":
        script_path = viber_script
    elif platform == "facebook":
        script_path = facebook_script
    else:
        print(f"[WARN] Unknown platform for {image_path.name}; skipping.")
        return None

    ensure_exists(script_path, f"{platform} extractor script")

    cmd = build_extractor_command(
        script_path=script_path,
        image_path=image_path,
        report_path=report_path,
        model=model,
        langs=langs,
        use_cpu=use_cpu,
        no_vision=no_vision,
        emoji_mode=emoji_mode,
        dump_ocr=dump_ocr,
        dump_draft=dump_draft,
        dump_side_map=dump_side_map,
        output_csv_path=output_csv_path,
        debug_dir_path=debug_dir_path,
        extra_args=extra_args,
    )

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    debug_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[RUN] {platform.upper()} extractor for: {image_path}")
    print("[CMD]", " ".join(quote_for_log(x) for x in cmd))

    completed = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    print(completed.stdout)

    if completed.returncode != 0:
        print(f"[ERROR] Extractor failed for {image_path.name} with exit code {completed.returncode}")
        return None

    if output_csv_path.exists():
        return output_csv_path

    csv_path = parse_success_csv_path(completed.stdout)

    if csv_path and csv_path.exists():
        return csv_path

    print(f"[WARN] Could not locate CSV output for {image_path.name}")
    print(f"[WARN] Expected CSV at: {output_csv_path}")
    return None


def quote_for_log(value: str) -> str:
    """Quotes command arguments only when logging needs it for readability."""
    if re.search(r"\s", value):
        return f'"{value}"'
    return value


def parse_success_csv_path(output: str) -> Optional[Path]:
    """Finds a CSV output path mentioned in an extractor success log."""
    patterns = [
        r"\[SUCCESS\]\s*CSV saved to:\s*(.+)",
        r"CSV saved to:\s*(.+)",
        r"Saved to:\s*(.+\.csv)",
    ]

    for pattern in patterns:
        m = re.search(pattern, output)
        if not m:
            continue

        raw = m.group(1).strip().strip('"').strip("'")
        path = Path(raw)

        if path.suffix.lower() == ".csv":
            return path

    return None


# ============================================================
# CSV MERGING / SORTING
# ============================================================

def read_chat_csv(csv_path: Path, source_image: Optional[Path] = None) -> List[Dict[str, str]]:
    """Reads a final extractor CSV into normalized row dictionaries."""
    rows: List[Dict[str, str]] = []

    text = read_text_flexible(csv_path)
    reader = csv.reader(io.StringIO(text))

    try:
        header = next(reader)
    except StopIteration:
        print(f"[WARN] Empty CSV, skipping: {csv_path}")
        return rows

    normalized_header = [
        str(col or "").lstrip("\ufeff").strip()
        for col in header
    ]

    required_columns = ["Time", "Sender", "Receiver", "Message"]
    missing = [col for col in required_columns if col not in normalized_header]

    if missing:
        print(f"[WARN] CSV does not have expected columns, skipping: {csv_path}")
        print(f"[WARN] Header found: {normalized_header}")
        return rows

    indexes = {col: normalized_header.index(col) for col in required_columns}

    for index, row in enumerate(reader):
        if not row:
            continue

        def get_cell(col: str) -> str:
            """Returns a safe stripped CSV cell value by column name."""
            i = indexes[col]
            if i >= len(row):
                return ""
            return str(row[i]).strip()

        item = {
            "Time": get_cell("Time"),
            "Sender": get_cell("Sender"),
            "Receiver": get_cell("Receiver"),
            "Message": get_cell("Message"),
            "_source_csv": str(csv_path),
            "_source_image": str(source_image or ""),
            "_source_row": str(index),
        }

        if item["Time"] and item["Sender"] and item["Receiver"] and item["Message"]:
            rows.append(item)

    return rows


def read_text_flexible(path: Path) -> str:
    # Try utf-8-sig first so a BOM in the first header field becomes "Time", not "\ufeffTime".
    """Reads text using common encodings, including UTF-8 with BOM."""
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue

    return path.read_text(errors="ignore")


def parse_chat_datetime(time_value: str) -> Tuple[int, datetime]:
    """Parses transcript timestamps for chronological sorting."""
    text = str(time_value or "").strip()

    for fmt in ("%d/%m/%Y %H:%M", "%d/%m/%Y, %H:%M", "%d/%m/%Y,%H:%M"):
        try:
            return 0, datetime.strptime(text, fmt)
        except ValueError:
            pass

    return 1, datetime.max


def write_merged_csv(
    rows: List[Dict[str, str]],
    output_path: Path,
    dedupe: bool = True,
) -> None:
    """Writes one chronologically sorted and optionally deduplicated CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    indexed_rows = list(enumerate(rows))

    indexed_rows.sort(
        key=lambda pair: (
            parse_chat_datetime(pair[1]["Time"])[0],
            parse_chat_datetime(pair[1]["Time"])[1],
            pair[0],
        )
    )

    out_rows: List[Dict[str, str]] = []
    seen = set()

    for _, row in indexed_rows:
        key = (
            row["Time"],
            row["Sender"],
            row["Receiver"],
            normalize_message_for_dedupe(row["Message"]),
        )

        if dedupe and key in seen:
            continue

        seen.add(key)
        out_rows.append(row)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(["Time", "Sender", "Receiver", "Message"])

        for row in out_rows:
            writer.writerow([
                row["Time"],
                row["Sender"],
                row["Receiver"],
                row["Message"],
            ])


def normalize_message_for_dedupe(message: str) -> str:
    """Normalizes message text only for duplicate-row detection."""
    text = str(message or "").lower()
    text = text.replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def write_run_manifest(
    manifest_path: Path,
    records: List[Dict[str, str]],
    output_csv: Path,
) -> None:
    """Writes a JSON manifest describing per-image extraction results."""
    manifest = {
        "output_csv": str(output_csv),
        "items": records,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    """Parses CLI arguments and runs the batch extraction pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Classify chat screenshots/collages as Viber or Facebook Messenger, "
            "run the matching extractor, and merge all extracted CSV files "
            "chronologically into one CSV."
        )
    )

    parser.add_argument(
        "case_report",
        help="Path to case report PDF/TXT.",
    )

    parser.add_argument(
        "inputs",
        nargs="+",
        help="Image files, folders, or globs. Example: ../images/*.png",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Output merged CSV path. "
            "Default: ./results/merged/<case_report_stem>_merged_chats.csv. "
            "Relative paths are placed under --results-dir/merged."
        ),
    )

    parser.add_argument(
        "--results-dir",
        default="./results",
        help=(
            "Root folder for generated files. Default: ./results. "
            "The script creates per_image/ and merged/ inside it."
        ),
    )

    parser.add_argument(
        "--viber-script",
        default=None,
        help="Path to viber_extract.py. Default: next to this script.",
    )

    parser.add_argument(
        "--facebook-script",
        default=None,
        help="Path to facebook_extract.py. Default: next to this script.",
    )

    parser.add_argument(
        "--classify-mode",
        choices=["auto", "vision", "filename"],
        default="auto",
        help=(
            "Platform classification mode. "
            "Default: auto. "
            "auto = filename/path first then VLM if unknown, "
            "vision = VLM first then filename/path fallback, "
            "filename = only filename/path heuristics."
        ),
    )

    parser.add_argument(
        "--force-platform",
        choices=["auto", "facebook", "viber"],
        default="auto",
        help=(
            "Force all input images to one extractor and skip classification. "
            "Useful for focused tests, e.g. --force-platform facebook. "
            "Default: auto."
        ),
    )

    parser.add_argument(
        "--model",
        default="gemma3:12b",
        help="Ollama model for classification and extractors. Default: gemma3:12b",
    )

    parser.add_argument(
        "--langs",
        default="en",
        help="EasyOCR languages passed to extractors. Default: en",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Pass --cpu to extractors.",
    )

    parser.add_argument(
        "--no-vision",
        action="store_true",
        help=(
            "Pass --no-vision to extractors. "
            "Classification may still use VLM unless --classify-mode=filename "
            "or --force-platform is used."
        ),
    )

    parser.add_argument(
        "--emoji-mode",
        choices=["omit", "vision"],
        default="omit",
        help="Passed to extractors. Default: omit.",
    )

    parser.add_argument(
        "--dump-ocr",
        action="store_true",
        help="Pass --dump-ocr to extractors.",
    )

    parser.add_argument(
        "--dump-draft",
        action="store_true",
        help="Pass --dump-draft to extractors.",
    )

    parser.add_argument(
        "--dump-side-map",
        action="store_true",
        help="Pass --dump-side-map to extractors.",
    )

    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Do not remove exact duplicate final rows while merging.",
    )

    parser.add_argument(
        "--manifest",
        default=None,
        help=(
            "Optional JSON manifest path. "
            "Default: <output>.manifest.json. Relative paths are placed under --results-dir/merged."
        ),
    )

    parser.add_argument(
        "--extra-extractor-arg",
        action="append",
        default=[],
        help=(
            "Extra argument passed to both extractors. "
            "Use multiple times if needed, e.g. --extra-extractor-arg --some-flag"
        ),
    )

    args = parser.parse_args()

    report_path = Path(args.case_report)
    ensure_exists(report_path, "case report")

    viber_script = Path(args.viber_script) if args.viber_script else default_script_path("viber_extract.py")
    facebook_script = Path(args.facebook_script) if args.facebook_script else default_script_path("facebook_extract.py")

    # The refactored extractors import shared helpers from extractor_utils.py.
    # Check this early so users get a clear error instead of a child-process ImportError.
    ensure_exists(viber_script, "Viber extractor script")
    ensure_exists(facebook_script, "Facebook extractor script")
    ensure_extractor_utils_available(viber_script)
    ensure_extractor_utils_available(facebook_script)

    results_dir = Path(args.results_dir)
    per_image_dir = results_dir / PER_IMAGE_DIR_NAME
    merged_dir = results_dir / MERGED_DIR_NAME

    results_dir.mkdir(parents=True, exist_ok=True)
    per_image_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    output_path = resolve_output_path(
        args.output,
        merged_dir,
        f"{report_path.stem}_merged_chats.csv",
    )

    manifest_path = resolve_output_path(
        args.manifest,
        merged_dir,
        output_path.name + ".manifest.json",
    )

    images = expand_inputs(args.inputs)

    if not images:
        print("[ERROR] No image files found.")
        return 2

    print("[START]")
    print(f"-> Case report: {report_path}")
    print(f"-> Images found: {len(images)}")
    print(f"-> Viber script: {viber_script}")
    print(f"-> Facebook script: {facebook_script}")
    print(f"-> Results root: {results_dir}")
    print(f"-> Per-image outputs: {per_image_dir}")
    print(f"-> Merged outputs: {merged_dir}")
    print(f"-> Output CSV: {output_path}")
    print(f"-> Classify mode: {args.classify_mode}")
    print(f"-> Force platform: {args.force_platform}")

    all_rows: List[Dict[str, str]] = []
    manifest_records: List[Dict[str, str]] = []
    used_output_stems: Set[str] = set()

    for image_path in images:
        print(f"\n[CLASSIFY] {image_path}")

        platform = classify_platform(
            image_path=image_path,
            model=args.model,
            mode=args.classify_mode,
            force_platform=args.force_platform,
        )

        print(f"-> Platform: {platform}")

        output_stem = unique_output_stem(image_path, used_output_stems)
        extractor_csv_path = per_image_dir / f"{output_stem}_extracted.csv"
        extractor_debug_dir = per_image_dir / f"{output_stem}_debug"

        record = {
            "image": str(image_path),
            "platform": platform,
            "csv": "",
            "debug_dir": str(extractor_debug_dir),
            "status": "skipped",
        }

        csv_path = run_extractor(
            platform=platform,
            image_path=image_path,
            report_path=report_path,
            viber_script=viber_script,
            facebook_script=facebook_script,
            model=args.model,
            langs=args.langs,
            use_cpu=args.cpu,
            no_vision=args.no_vision,
            emoji_mode=args.emoji_mode,
            dump_ocr=args.dump_ocr,
            dump_draft=args.dump_draft,
            dump_side_map=args.dump_side_map,
            output_csv_path=extractor_csv_path,
            debug_dir_path=extractor_debug_dir,
            extra_args=args.extra_extractor_arg,
        )

        if csv_path:
            record["csv"] = str(csv_path)
            record["status"] = "ok"

            rows = read_chat_csv(csv_path, source_image=image_path)
            print(f"-> Rows read: {len(rows)}")
            all_rows.extend(rows)
        else:
            record["status"] = "failed"

        manifest_records.append(record)

    if not all_rows:
        print("[ERROR] No rows extracted. Merged CSV was not created.")
        write_run_manifest(manifest_path, manifest_records, output_path)
        print(f"[INFO] Manifest saved to: {manifest_path}")
        return 3

    write_merged_csv(
        rows=all_rows,
        output_path=output_path,
        dedupe=not args.keep_duplicates,
    )

    write_run_manifest(manifest_path, manifest_records, output_path)

    print("\n[SUCCESS]")
    print(f"CSV saved to: {output_path}")
    print(f"Rows merged: {len(all_rows)}")
    print(f"Manifest saved to: {manifest_path}")
    print(f"Per-image files saved under: {per_image_dir}")
    print(f"Merged files saved under: {merged_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())