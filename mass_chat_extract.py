"""Batch orchestrator for use-case ZIP/folder inputs and chat screenshot extraction."""

import argparse
import csv
import glob
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"
}
REPORT_EXTENSIONS = {".pdf", ".txt"}
IGNORED_DIR_NAMES = {
    "__macosx", ".git", ".svn", ".hg", "node_modules", "__pycache__",
}

# Output layout created by this script:
# results/
#   _extracted_zips/ -> extracted ZIP contents used during this run
#   merged/          -> final merged CSV and manifest JSON
#   per_image/       -> optional, created only with --keep-per-image or debug/dump flags
PER_IMAGE_DIR_NAME = "per_image"
MERGED_DIR_NAME = "merged"
EXTRACTED_ZIPS_DIR_NAME = "_extracted_zips"
SHARED_UTILS_FILENAME = "extractor_utils.py"

@dataclass
class EvidenceSource:
    """Resolved evidence source after optional ZIP extraction."""
    original_path: Path
    root_path: Path
    source_type: str  # file, folder, zip
    extracted: bool = False

@dataclass
class InputPlan:
    """Resolved input plan for either package or explicit-report mode."""
    mode: str
    report_path: Path
    evidence_sources: List[EvidenceSource]
    images: List[Path]
    file_inventory: List[Dict[str, str]]
    run_stem: str

# ============================================================
# PATH / INPUT HELPERS
# ============================================================
def is_image_file(path: Path) -> bool:
    """Checks whether a path is a supported image file."""
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS

def is_report_file(path: Path) -> bool:
    """Checks whether a path can be used as a case report/overview."""
    return path.is_file() and path.suffix.lower() in REPORT_EXTENSIONS

def is_zip_file(path: Path) -> bool:
    """Checks whether a path is a ZIP archive."""
    return path.is_file() and path.suffix.lower() == ".zip"

def default_script_path(script_name: str) -> Path:
    """Returns the default path for an extractor script beside this file."""
    return Path(__file__).resolve().parent / script_name

def safe_output_stem(path: Path) -> str:
    """Creates a filesystem-safe filename stem for generated outputs."""
    stem = path.stem if path.suffix else path.name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem or "input"

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

def resolve_output_path(path_value: Optional[str], base_dir: Path, default_name: str) -> Path:
    """Resolves an output path under a base directory unless it is absolute."""
    if path_value is None:
        return base_dir / default_name

    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path

def list_files_recursively(root: Path) -> List[Path]:
    """Returns all files under a folder, skipping common metadata/cache folders."""
    files: List[Path] = []

    if root.is_file():
        return [root]

    for path in sorted(root.rglob("*"), key=lambda p: str(p).lower()):
        if not path.is_file():
            continue
        if any(part.lower() in IGNORED_DIR_NAMES for part in path.parts):
            continue
        files.append(path)

    return files

def short_path_for_manifest(path: Path, root: Optional[Path] = None) -> str:
    """Returns a readable relative path when possible."""
    if root:
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return str(path)

# ============================================================
# ZIP HANDLING
# ============================================================
def stable_zip_extract_dir(zip_path: Path, extract_root: Path) -> Path:
    """Builds a stable extraction folder for one ZIP input."""
    try:
        stat = zip_path.stat()
        fingerprint_source = f"{zip_path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"
    except OSError:
        fingerprint_source = str(zip_path.resolve())

    digest = hashlib.sha1(fingerprint_source.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return extract_root / f"{safe_output_stem(zip_path)}_{digest}"

def safe_extract_zip(zip_path: Path, extract_root: Path) -> Path:
    """Extracts a ZIP archive safely under extract_root and returns the folder."""
    ensure_exists(zip_path, "ZIP input")
    output_dir = stable_zip_extract_dir(zip_path, extract_root)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            member_name = member.filename
            if not member_name or member_name.endswith("/"):
                continue

            target_path = output_dir / member_name
            resolved_target = target_path.resolve()
            resolved_root = output_dir.resolve()

            # Prevent ZIP Slip path traversal.
            if resolved_root not in resolved_target.parents and resolved_target != resolved_root:
                print(f"[WARN] Skipping unsafe ZIP member: {member_name}")
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)

    return output_dir

def resolve_evidence_source(raw_path: Path, extract_root: Path) -> EvidenceSource:
    """Resolves a file/folder/ZIP evidence source."""
    raw_path = raw_path.expanduser()
    ensure_exists(raw_path, "input")

    if is_zip_file(raw_path):
        extracted_dir = safe_extract_zip(raw_path, extract_root)
        return EvidenceSource(
            original_path=raw_path,
            root_path=extracted_dir,
            source_type="zip",
            extracted=True,
        )

    if raw_path.is_dir():
        return EvidenceSource(
            original_path=raw_path,
            root_path=raw_path,
            source_type="folder",
            extracted=False,
        )

    return EvidenceSource(
        original_path=raw_path,
        root_path=raw_path,
        source_type="file",
        extracted=False,
    )

def resolve_evidence_sources(raw_inputs: Sequence[str], extract_root: Path) -> List[EvidenceSource]:
    """Expands globs and resolves all evidence inputs."""
    sources: List[EvidenceSource] = []

    for raw in raw_inputs:
        expanded = glob.glob(raw)
        values = expanded if expanded else [raw]

        for value in values:
            path = Path(value)
            try:
                sources.append(resolve_evidence_source(path, extract_root))
            except FileNotFoundError:
                print(f"[WARN] Skipping missing input: {value}")

    return sources

# ============================================================
# PACKAGE SCANNING / REPORT DISCOVERY
# ============================================================
def report_candidate_score(path: Path, source_root: Path) -> int:
    """Scores likely case overview/report files inside a package."""
    if not is_report_file(path):
        return -10_000

    name = path.name.lower()
    full = str(path.relative_to(source_root)).lower() if source_root in path.parents or path == source_root else str(path).lower()

    score = 0

    # Strongly prefer the parent-level Cases Overview / initial allegation file.
    if re.search(r"cases?[_\s-]*overview", name):
        score += 100
    if "overview" in name:
        score += 60
    if "allegation" in name or "initial" in name:
        score += 45
    if "case" in name:
        score += 35
    if "report" in name:
        score += 30
    if "complaint" in name or "complainant" in name or "victim" in name:
        score += 25
    if "statement" in name:
        score += 15

    # De-prioritize obvious generated/output/debug files.
    noisy_terms = [
        "log", "ground_truth", "groundtruth", "accuracy", "result", "output",
        "debug", "extracted", "merged", "transcript",
    ]
    if any(term in name for term in noisy_terms):
        score -= 40

    # Prefer files closer to the package root.
    try:
        depth = len(path.relative_to(source_root).parts)
    except ValueError:
        depth = len(path.parts)
    score -= max(0, depth - 1) * 3

    if "evidence" in full:
        score -= 8
    if "structured" in full:
        score -= 10
    if "unstructured" in full:
        score -= 3

    # Prefer PDFs over TXT when names are otherwise similar.
    if path.suffix.lower() == ".pdf":
        score += 5

    return score

def discover_case_report(files: List[Path], source_root: Path) -> Optional[Path]:
    """Finds the most likely case report/overview inside a use-case package."""
    candidates = [p for p in files if is_report_file(p)]
    if not candidates:
        return None

    scored = sorted(
        ((report_candidate_score(path, source_root), path) for path in candidates),
        key=lambda item: (item[0], -len(item[1].parts), str(item[1]).lower()),
        reverse=True,
    )

    best_score, best_path = scored[0]
    if best_score <= -1000:
        return None
    return best_path

def scan_source_files(source: EvidenceSource) -> List[Path]:
    """Lists files contained in one evidence source."""
    if source.root_path.is_file():
        return [source.root_path]
    return list_files_recursively(source.root_path)

def build_file_inventory(
    sources: List[EvidenceSource],
    report_path: Optional[Path],
    images: List[Path],
) -> List[Dict[str, str]]:
    """Builds manifest records for all files found in inputs."""
    image_set = {str(p.resolve()) for p in images if p.exists()}
    report_resolved = str(report_path.resolve()) if report_path and report_path.exists() else ""
    records: List[Dict[str, str]] = []

    for source in sources:
        for path in scan_source_files(source):
            resolved = str(path.resolve()) if path.exists() else str(path)
            if resolved == report_resolved:
                status = "case_report"
            elif resolved in image_set:
                status = "candidate_image"
            elif is_report_file(path):
                status = "ignored_report_candidate"
            elif is_image_file(path):
                status = "candidate_image"
            else:
                status = "ignored_non_image"

            records.append({
                "source": str(source.original_path),
                "path": short_path_for_manifest(path, source.root_path if source.root_path.is_dir() else None),
                "absolute_path": str(path),
                "extension": path.suffix.lower(),
                "status": status,
            })

    return records

def collect_images_from_sources(sources: List[EvidenceSource]) -> List[Path]:
    """Collects image files recursively from all resolved sources."""
    images: List[Path] = []

    for source in sources:
        for path in scan_source_files(source):
            if is_image_file(path):
                images.append(path)

    unique: Dict[str, Path] = {}
    for path in sorted(images, key=lambda p: str(p).lower()):
        unique[str(path.resolve())] = path
    return list(unique.values())

def resolve_input_plan(args: argparse.Namespace, extract_root: Path) -> InputPlan:
    """Resolves CLI input into report path, image list, and manifest inventory."""
    first_path = Path(args.package_or_report).expanduser()

    # Mode 1: explicit --case-report, all positional paths are evidence inputs.
    if args.case_report:
        report_path = Path(args.case_report).expanduser()
        ensure_exists(report_path, "case report")
        raw_inputs = [args.package_or_report] + list(args.inputs)
        sources = resolve_evidence_sources(raw_inputs, extract_root)
        images = collect_images_from_sources(sources)
        run_stem = safe_output_stem(sources[0].original_path if sources else report_path)
        inventory = build_file_inventory(sources, report_path, images)
        return InputPlan("explicit_report", report_path, sources, images, inventory, run_stem)

    # Mode 2: legacy command: first positional is report, remaining are evidence inputs.
    if args.inputs:
        report_path = first_path
        ensure_exists(report_path, "case report")
        if not is_report_file(report_path):
            raise ValueError(
                "When multiple positional arguments are used, the first one must be a PDF/TXT case report. "
                "For package auto-discovery, pass only the ZIP/folder or use --case-report."
            )
        sources = resolve_evidence_sources(args.inputs, extract_root)
        images = collect_images_from_sources(sources)
        run_stem = safe_output_stem(sources[0].original_path if sources else report_path)
        inventory = build_file_inventory(sources, report_path, images)
        return InputPlan("explicit_report", report_path, sources, images, inventory, run_stem)

    # Mode 3: package auto-discovery: one ZIP/folder contains overview/report + evidence.
    package_source = resolve_evidence_source(first_path, extract_root)
    files = scan_source_files(package_source)
    report_path = discover_case_report(files, package_source.root_path if package_source.root_path.is_dir() else package_source.root_path.parent)

    if report_path is None:
        raise FileNotFoundError(
            "Could not auto-discover a case report/overview PDF or TXT in the input package. "
            "Use --case-report /path/to/report.pdf or the legacy form: "
            "python3 mass_chat_extract.py case_report.pdf evidence.zip"
        )

    images = [p for p in files if is_image_file(p)]
    unique: Dict[str, Path] = {}
    for path in sorted(images, key=lambda p: str(p).lower()):
        unique[str(path.resolve())] = path
    images = list(unique.values())

    inventory = build_file_inventory([package_source], report_path, images)
    return InputPlan(
        mode="package_auto_report",
        report_path=report_path,
        evidence_sources=[package_source],
        images=images,
        file_inventory=inventory,
        run_stem=safe_output_stem(package_source.original_path),
    )

# ============================================================
# PLATFORM CLASSIFICATION
# ============================================================
def normalize_platform(value: str) -> str:
    """Normalizes platform aliases to facebook, viber, non_chat, or unknown."""
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

    if text in {"facebook", "messenger", "facebook_messenger", "fb"}:
        return "facebook"
    if text == "viber":
        return "viber"
    if text in {"non_chat", "not_chat", "no_chat", "other", "irrelevant", "not_a_chat"}:
        return "non_chat"
    return "unknown"

def classify_by_filename(image_path: Path) -> str:
    """Classifies the platform using deterministic filename and folder hints."""
    text = str(image_path).lower().replace("\\", "/")

    facebook_keywords = [
        "facebook", "messenger", "fb_", "/fb/", "/facebook/", "/messenger/",
    ]
    viber_keywords = ["viber", "/viber/"]

    if any(keyword in text for keyword in facebook_keywords):
        return "facebook"
    if any(keyword in text for keyword in viber_keywords):
        return "viber"
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

def classify_platform_with_vlm(image_path: Path, model: str) -> str:
    """Uses a vision-capable Ollama model to classify the image or skip non-chat images."""
    try:
        import ollama
    except Exception as exc:
        print(f"[WARN] Could not import ollama for VLM classification: {exc}")
        return "unknown"

    prompt = """
You are classifying an evidence image.

Look only at the image UI/content and return one of:
- facebook: a Facebook Messenger chat screenshot or Messenger chat collage
- viber: a Viber chat screenshot or Viber chat collage
- non_chat: not a human chat screenshot/collage, or not enough chat UI evidence
- unknown: could be a chat, but the app cannot be determined

Return only JSON with this exact schema:
{"platform":"facebook"}
or
{"platform":"viber"}
or
{"platform":"non_chat"}
or
{"platform":"unknown"}

Rules:
1. Use facebook only for Facebook Messenger / Messenger chat UI.
2. Use viber only for Viber chat UI.
3. Use non_chat for documents, tables, photos, scans, maps, logos, screenshots of non-chat apps, or unrelated images.
4. If unsure whether it is chat evidence, return unknown.
5. Do not explain.
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
            options={"temperature": 0},
        )

        text = response["message"]["content"].strip()
        data = extract_json_object(text)
        platform = normalize_platform(data.get("platform", ""))

        if platform in {"viber", "facebook", "non_chat"}:
            return platform

        # Small fallback if model returned text instead of JSON.
        low = text.lower()
        if "non_chat" in low or "not chat" in low or "not a chat" in low:
            return "non_chat"
        if "viber" in low:
            return "viber"
        if "facebook" in low or "messenger" in low:
            return "facebook"
        return "unknown"

    except Exception as exc:
        print(f"[WARN] VLM classification failed for {image_path.name}: {exc}")
        return "unknown"

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

    # Default: auto. Filename/path first, then VLM if unknown.
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
        "--model", model,
        "--langs", langs,
        "--emoji-mode", emoji_mode,
        "--output", str(output_csv_path),
        "--debug-dir", str(debug_dir_path),
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
    keep_debug: bool,
) -> Optional[Path]:
    """Runs the selected extractor and returns the produced CSV path."""
    if platform == "viber":
        script_path = viber_script
    elif platform == "facebook":
        script_path = facebook_script
    else:
        print(f"[WARN] Unknown/non-chat platform for {image_path.name}; skipping.")
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

    if not keep_debug and debug_dir_path.exists():
        shutil.rmtree(debug_dir_path, ignore_errors=True)

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

    normalized_header = [str(col or "").lstrip("\ufeff").strip() for col in header]
    required_columns = ["Time", "Sender", "Receiver", "Message"]
    missing = [col for col in required_columns if col not in normalized_header]

    if missing:
        print(f"[WARN] CSV does not have expected columns, skipping: {csv_path}")
        print(f"[WARN] Header found: {normalized_header}")
        return rows

    indexes = {col: normalized_header.index(col) for col in required_columns}

    for index, row in enumerate(reader, start=2):
        if not row:
            continue

        def get_cell(col: str) -> str:
            i = indexes[col]
            return str(row[i]).strip() if i < len(row) else ""

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

def normalize_message_for_dedupe(message: str) -> str:
    """Normalizes message text only for duplicate-row detection."""
    text = str(message or "").lower()
    text = text.replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def conservative_final_message_cleanup(message: str) -> str:
    """Final low-risk OCR cleanup for merged transcript rows."""
    msg = str(message or "").strip()
    msg = msg.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    msg = re.sub(r"\s+", " ", msg).strip()
    msg = re.sub(r"\s+([,.;:!?])", r"\1", msg)
    msg = re.sub(r"([,.;:!?])(?=[A-Za-z])", r"\1 ", msg)
    return msg

def split_final_chat_row(row: Dict[str, str]) -> List[Dict[str, str]]:
    """Apply final message cleanup without dataset-specific row splitting."""
    fixed = dict(row)
    fixed["Message"] = conservative_final_message_cleanup(fixed.get("Message", ""))
    return [fixed] if fixed["Message"] else []

def merge_final_adjacent_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Return final rows without dataset-specific adjacent-row merges."""
    return rows

def postprocess_final_chat_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Apply final merged-transcript cleanup, guarded bubble splits, and tiny row merges."""
    out: List[Dict[str, str]] = []
    for row in rows:
        out.extend(split_final_chat_row(row))
    return merge_final_adjacent_rows(out)

def write_merged_csv(rows: List[Dict[str, str]], output_path: Path, dedupe: bool = True) -> int:
    """Writes one chronologically sorted and optionally deduplicated CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = postprocess_final_chat_rows(rows)
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
            writer.writerow([row["Time"], row["Sender"], row["Receiver"], row["Message"]])

    return len(out_rows)

# ============================================================
# MANIFEST
# ============================================================
def utc_now_iso() -> str:
    """Returns current UTC timestamp for manifest metadata."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def write_run_manifest(
    manifest_path: Path,
    input_plan: InputPlan,
    item_records: List[Dict[str, str]],
    output_csv: Path,
    rows_merged: int,
    args: argparse.Namespace,
) -> None:
    """Writes a JSON manifest describing extraction inputs/results."""
    manifest = {
        "created_at": utc_now_iso(),
        "mode": input_plan.mode,
        "case_report": str(input_plan.report_path),
        "output_csv": str(output_csv),
        "rows_merged": rows_merged,
        "settings": {
            "model": args.model,
            "langs": args.langs,
            "emoji_mode": args.emoji_mode,
            "classify_mode": args.classify_mode,
            "force_platform": args.force_platform,
            "debug": bool(args.debug),
            "no_vision": bool(args.no_vision),
            "cpu": bool(args.cpu),
        },
        "evidence_sources": [
            {
                "source": str(source.original_path),
                "resolved_root": str(source.root_path),
                "type": source.source_type,
                "extracted": source.extracted,
            }
            for source in input_plan.evidence_sources
        ],
        "counts": {
            "files_seen": len(input_plan.file_inventory),
            "candidate_images": len(input_plan.images),
            "processed_images": sum(1 for item in item_records if item.get("status") == "ok"),
            "skipped_images": sum(1 for item in item_records if item.get("status", "").startswith("skipped")),
            "failed_images": sum(1 for item in item_records if item.get("status") == "failed"),
        },
        "items": item_records,
        "files": input_plan.file_inventory,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

# ============================================================
# MAIN
# ============================================================
def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract chat transcripts from a use-case ZIP/folder or from an explicit "
            "case report plus evidence images/folders/ZIPs."
        )
    )

    parser.add_argument(
        "package_or_report",
        help=(
            "Either: (1) a use-case ZIP/folder containing the case overview/report and evidence, "
            "or (2) a PDF/TXT case report when additional input paths are supplied."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Legacy mode only: image files, folders, ZIPs, or globs to process with the explicit case report.",
    )
    parser.add_argument(
        "--case-report",
        default=None,
        help=(
            "Override auto-discovery and use this PDF/TXT report. With this option, "
            "all positional paths are treated as evidence inputs."
        ),
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help=(
            "Output merged CSV path. Default: ./results/merged/<input_stem>_extracted_chat.csv. "
            "Relative paths are placed under --results-dir/merged."
        ),
    )
    parser.add_argument(
        "--results-dir",
        default="./results",
        help="Root folder for generated files. Default: ./results.",
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
        help="Default: auto. auto = filename/path first, then VLM if unknown.",
    )
    parser.add_argument(
        "--force-platform",
        choices=["auto", "facebook", "viber"],
        default="auto",
        help="Force all candidate images to one extractor. Default: auto.",
    )
    parser.add_argument(
        "--model",
        default="gemma3:12b",
        help="Ollama model for classification and extractors. Default: gemma3:12b.",
    )
    parser.add_argument(
        "--langs",
        default="en",
        help="EasyOCR languages passed to extractors. Default: en.",
    )
    parser.add_argument("--cpu", action="store_true", help="Pass --cpu to extractors.")
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help=(
            "Pass --no-vision to extractors. Classification still uses VLM unless "
            "--classify-mode=filename or --force-platform is used."
        ),
    )
    parser.add_argument(
        "--emoji-mode",
        choices=["omit", "vision"],
        default="omit",
        help="Passed to extractors. Default: omit.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Keep per-image debug folders under results/per_image. "
            "Default: off; temporary per-image files are removed."
        ),
    )
    parser.add_argument(
        "--keep-per-image",
        action="store_true",
        help=(
            "Keep one intermediate CSV per processed image under results/per_image. "
            "Default: off; only the merged CSV and manifest are kept."
        ),
    )
    parser.add_argument("--dump-ocr", action="store_true", help="Pass --dump-ocr to extractors and keep debug output.")
    parser.add_argument("--dump-draft", action="store_true", help="Pass --dump-draft to extractors and keep debug output.")
    parser.add_argument("--dump-side-map", action="store_true", help="Pass --dump-side-map to extractors and keep debug output.")
    parser.add_argument("--keep-duplicates", action="store_true", help="Do not remove exact duplicate final rows while merging.")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional JSON manifest path. Default: <output>.manifest.json.",
    )
    parser.add_argument(
        "--extra-extractor-arg",
        action="append",
        default=[],
        help="Extra argument passed to both extractors. Use multiple times if needed.",
    )
    return parser

def main() -> int:
    """Parses CLI arguments and runs the batch extraction pipeline."""
    parser = build_parser()
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser()
    merged_dir = results_dir / MERGED_DIR_NAME
    extract_root = results_dir / EXTRACTED_ZIPS_DIR_NAME

    results_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)
    extract_root.mkdir(parents=True, exist_ok=True)

    try:
        input_plan = resolve_input_plan(args, extract_root)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    viber_script = Path(args.viber_script).expanduser() if args.viber_script else default_script_path("viber_extract.py")
    facebook_script = Path(args.facebook_script).expanduser() if args.facebook_script else default_script_path("facebook_extract.py")

    try:
        ensure_exists(viber_script, "Viber extractor script")
        ensure_exists(facebook_script, "Facebook extractor script")
        ensure_extractor_utils_available(viber_script)
        ensure_extractor_utils_available(facebook_script)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    output_path = resolve_output_path(
        args.output,
        merged_dir,
        f"{input_plan.run_stem}_extracted_chat.csv",
    )
    manifest_path = resolve_output_path(
        args.manifest,
        merged_dir,
        output_path.name + ".manifest.json",
    )

    if not input_plan.images:
        print("[ERROR] No candidate image files found in the input package/source.", file=sys.stderr)
        write_run_manifest(manifest_path, input_plan, [], output_path, 0, args)
        print(f"[INFO] Manifest saved to: {manifest_path}")
        return 3

    keep_debug = bool(args.debug or args.dump_ocr or args.dump_draft or args.dump_side_map)
    keep_per_image = bool(args.keep_per_image or keep_debug)
    temporary_per_image_workspace: Optional[tempfile.TemporaryDirectory[str]] = None

    if keep_per_image:
        per_image_dir = results_dir / PER_IMAGE_DIR_NAME
        per_image_dir.mkdir(parents=True, exist_ok=True)
        per_image_label = str(per_image_dir)
    else:
        temporary_per_image_workspace = tempfile.TemporaryDirectory(prefix="chat_extract_per_image_")
        per_image_dir = Path(temporary_per_image_workspace.name)
        per_image_dir.mkdir(parents=True, exist_ok=True)
        per_image_label = "temporary workspace, removed after merge"

    print("[START]")
    print(f"-> Mode: {input_plan.mode}")
    print(f"-> Case report: {input_plan.report_path}")
    print(f"-> Candidate images found: {len(input_plan.images)}")
    print(f"-> Viber script: {viber_script}")
    print(f"-> Facebook script: {facebook_script}")
    print(f"-> Results root: {results_dir}")
    print(f"-> Per-image outputs: {per_image_label}")
    print(f"-> Merged outputs: {merged_dir}")
    print(f"-> Output CSV: {output_path}")
    print(f"-> Manifest: {manifest_path}")
    print(f"-> Classify mode: {args.classify_mode}")
    print(f"-> Force platform: {args.force_platform}")
    print(f"-> Emoji mode: {args.emoji_mode}")
    print(f"-> Keep debug: {keep_debug}")
    print(f"-> Keep per-image outputs: {keep_per_image}")

    all_rows: List[Dict[str, str]] = []
    item_records: List[Dict[str, str]] = []
    used_output_stems: Set[str] = set()

    for image_path in input_plan.images:
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

        record: Dict[str, str] = {
            "image": str(image_path),
            "platform": platform,
            "csv": "",
            "debug_dir": str(extractor_debug_dir) if keep_debug else "",
            "status": "skipped_unknown_or_non_chat",
            "reason": "",
            "rows": "0",
        }

        if platform not in {"facebook", "viber"}:
            record["reason"] = "classifier returned non_chat/unknown"
            print("-> Skipping image because it is not a recognized Facebook/Viber chat screenshot.")
            item_records.append(record)
            continue

        csv_path = run_extractor(
            platform=platform,
            image_path=image_path,
            report_path=input_plan.report_path,
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
            keep_debug=keep_debug,
        )

        if csv_path:
            rows = read_chat_csv(csv_path, source_image=image_path)
            print(f"-> Rows read: {len(rows)}")
            all_rows.extend(rows)
            record["csv"] = str(csv_path) if keep_per_image else ""
            record["status"] = "ok"
            record["reason"] = "processed"
            record["rows"] = str(len(rows))
        else:
            record["status"] = "failed"
            record["reason"] = "extractor failed or CSV output missing"

        item_records.append(record)

    if not all_rows:
        print("[ERROR] No rows extracted. Merged CSV was not created.", file=sys.stderr)
        write_run_manifest(manifest_path, input_plan, item_records, output_path, 0, args)
        print(f"[INFO] Manifest saved to: {manifest_path}")
        if temporary_per_image_workspace is not None:
            temporary_per_image_workspace.cleanup()
        return 4

    rows_written = write_merged_csv(
        rows=all_rows,
        output_path=output_path,
        dedupe=not args.keep_duplicates,
    )

    write_run_manifest(manifest_path, input_plan, item_records, output_path, rows_written, args)

    print("\n[SUCCESS]")
    print(f"CSV saved to: {output_path}")
    print(f"Rows collected before final dedupe: {len(all_rows)}")
    print(f"Rows written: {rows_written}")
    print(f"Manifest saved to: {manifest_path}")
    if keep_per_image:
        print(f"Per-image files saved under: {per_image_dir}")
    else:
        print("Per-image files were temporary and have been removed.")
    print(f"Merged files saved under: {merged_dir}")

    if temporary_per_image_workspace is not None:
        temporary_per_image_workspace.cleanup()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

