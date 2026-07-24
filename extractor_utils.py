"""Shared helper functions used by both chat screenshot extractors."""

import csv
import io
import re
from pathlib import Path
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple, Set

import cv2
import numpy as np
import ollama
from PyPDF2 import PdfReader

Box = Tuple[int, int, int, int]
ScreenCrop = Tuple[int, int, int, int, np.ndarray]

def extract_text_from_report(report_path: str) -> str:
    """Reads plain-text or PDF case-report content."""
    path = Path(report_path)

    if path.suffix.lower() == ".txt":
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
        return path.read_text(errors="ignore")

    if path.suffix.lower() == ".pdf":
        try:
            reader = PdfReader(str(path))
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            return "\n".join(pages)
        except Exception as e:
            print(f"[WARNING] Could not read PDF: {e}")
            return ""

    print("[WARNING] Unsupported report type. Use PDF or TXT.")
    return ""

def extract_year_from_report(report_text: str, default_year: int = 2026) -> int:
    """Finds the first timeline year in the case report."""
    years = re.findall(r"\b(20\d{2})\b", report_text)
    if not years:
        return default_year

    # Usually the first report/timeline year is the relevant year.
    return int(years[0])

def parse_grid(grid: Optional[str]) -> Optional[Tuple[int, int]]:
    """Parses a regular collage grid specification."""
    if not grid:
        return None

    m = re.match(r"^(\d+)x(\d+)$", grid.strip().lower())
    if not m:
        raise ValueError("--grid must be like 2x1, 3x2, etc.")

    cols, rows = int(m.group(1)), int(m.group(2))
    if cols <= 0 or rows <= 0:
        raise ValueError("--grid values must be positive.")

    return cols, rows

def parse_layout(layout: Optional[str]) -> Optional[List[int]]:
    """Parses an uneven collage row-layout specification."""
    if not layout:
        return None

    try:
        values = [int(x.strip()) for x in layout.split(",") if x.strip()]
    except ValueError:
        raise ValueError("--layout must be like 2,3 or 1,2,3.")

    if not values or any(v <= 0 for v in values):
        raise ValueError("--layout values must be positive.")

    return values

def trim_white_border(image: np.ndarray, threshold: int = 245, pad: int = 0) -> np.ndarray:
    """Removes near-white outer borders from an image crop."""
    if image.size == 0:
        return image

    white = np.all(image >= threshold, axis=2)
    content = ~white
    ys, xs = np.where(content)

    if len(xs) == 0 or len(ys) == 0:
        return image

    h, w = image.shape[:2]
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(w, int(xs.max()) + 1 + pad)
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(h, int(ys.max()) + 1 + pad)

    return image[y1:y2, x1:x2]

def ranges_from_indices(indices: np.ndarray) -> List[Tuple[int, int]]:
    """Converts consecutive index values into half-open ranges."""
    if len(indices) == 0:
        return []

    values = [int(x) for x in indices]
    ranges = []

    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
        else:
            ranges.append((start, prev + 1))
            start = prev = value

    ranges.append((start, prev + 1))
    return ranges

def find_separator_bands(
    image: np.ndarray,
    axis: str,
    white_threshold: int = 235,
    ratio_threshold: float = 0.72,
    min_band_size: int = 3,
) -> List[Tuple[int, int]]:
    """Finds likely white gutter bands along one image axis."""
    # Notes:
    # axis='x' -> vertical separator columns.
    # axis='y' -> horizontal separator rows.
    if image.size == 0:
        return []

    white = np.all(image >= white_threshold, axis=2)

    if axis == "x":
        ratio = white.mean(axis=0)
    elif axis == "y":
        ratio = white.mean(axis=1)
    else:
        raise ValueError("axis must be 'x' or 'y'")

    candidates = np.where(ratio >= ratio_threshold)[0]
    bands = ranges_from_indices(candidates)

    return [(a, b) for a, b in bands if (b - a) >= min_band_size]

def split_segments_by_bands(
    length: int,
    bands: List[Tuple[int, int]],
    min_size: int
) -> List[Tuple[int, int]]:
    """Splits a dimension into content segments around separator bands."""
    if not bands:
        return [(0, length)]

    segments = []
    cur = 0

    for a, b in bands:
        if a - cur >= min_size:
            segments.append((cur, a))
        cur = b

    if length - cur >= min_size:
        segments.append((cur, length))

    return segments

def manual_grid_split(image: np.ndarray, grid: str) -> List[ScreenCrop]:
    """Splits an image using a user-specified regular grid."""
    cols, rows = parse_grid(grid)
    h, w = image.shape[:2]
    crops = []

    cell_w = w / cols
    cell_h = h / rows

    for r in range(rows):
        for c in range(cols):
            x1 = int(c * cell_w)
            x2 = int((c + 1) * cell_w)
            y1 = int(r * cell_h)
            y2 = int((r + 1) * cell_h)

            crop = trim_white_border(image[y1:y2, x1:x2])
            crops.append((x1, y1, x2 - x1, y2 - y1, crop))

    return crops

def manual_layout_split(image: np.ndarray, layout: str) -> List[ScreenCrop]:
    """Splits an image using a user-specified uneven row layout."""
    # Notes:
    # top row: 2 screenshots
    # bottom row: 3 screenshots
    row_counts = parse_layout(layout)
    h, w = image.shape[:2]
    crops = []

    row_h = h / len(row_counts)

    for r, count in enumerate(row_counts):
        y1 = int(r * row_h)
        y2 = int((r + 1) * row_h)
        col_w = w / count

        for c in range(count):
            x1 = int(c * col_w)
            x2 = int((c + 1) * col_w)

            crop = trim_white_border(image[y1:y2, x1:x2])
            crops.append((x1, y1, x2 - x1, y2 - y1, crop))

    return crops

def auto_split_by_white_gutters(image: np.ndarray) -> List[ScreenCrop]:
    """Automatically splits collages using visible white gutters."""
    # Notes:
    # For important/known uneven layouts, prefer --layout 2,3.
    img = trim_white_border(image)
    h, w = img.shape[:2]

    if h == 0 or w == 0:
        return []

    min_h = max(180, int(h * 0.18))
    min_w = max(140, int(w * 0.13))

    horizontal_bands = find_separator_bands(img, axis="y")
    row_segments = split_segments_by_bands(h, horizontal_bands, min_h)

    crops = []

    for y1, y2 in row_segments:
        row_img = img[y1:y2, :]
        vertical_bands = find_separator_bands(row_img, axis="x")
        col_segments = split_segments_by_bands(w, vertical_bands, min_w)

        for x1, x2 in col_segments:
            crop = trim_white_border(row_img[:, x1:x2])
            ch, cw = crop.shape[:2]

            if cw >= min_w and ch >= min_h:
                crops.append((x1, y1, x2 - x1, y2 - y1, crop))

    if len(crops) <= 1:
        return [(0, 0, w, h, img)]

    return sort_screen_crops(crops)

def contour_fallback_split(image: np.ndarray) -> List[ScreenCrop]:
    """Splits an image using contour boxes when gutters fail."""
    img = trim_white_border(image)
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 200)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)

        if bw > w * 0.18 and bh > h * 0.25 and bh > bw * 0.8:
            boxes.append((x, y, bw, bh))

    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)

    kept = []
    for box in boxes:
        x, y, bw, bh = box
        cx, cy = x + bw / 2, y + bh / 2

        contained = False
        for kx, ky, kw, kh in kept:
            if kx <= cx <= kx + kw and ky <= cy <= ky + kh:
                contained = True
                break

        if not contained:
            kept.append(box)

    crops = []

    for x, y, bw, bh in kept:
        pad = 5
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)

        crops.append((x1, y1, x2 - x1, y2 - y1, img[y1:y2, x1:x2]))

    if not crops:
        return [(0, 0, w, h, img)]

    return sort_screen_crops(crops)

def sort_screen_crops(crops: List[ScreenCrop]) -> List[ScreenCrop]:
    """Orders screen crops from top-left to bottom-right."""
    if not crops:
        return []

    boxes = sorted(crops, key=lambda item: item[1])
    heights = [item[3] for item in boxes]
    threshold = max(40, int(np.median(heights) * 0.25))

    rows = []
    current = [boxes[0]]

    for item in boxes[1:]:
        if abs(item[1] - current[-1][1]) <= threshold:
            current.append(item)
        else:
            rows.append(current)
            current = [item]

    rows.append(current)

    ordered = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda item: item[0]))

    return ordered

def minimal_ocr_clean(text: str) -> str:
    """Applies minimal cleanup to OCR text."""
    text = text.replace("\u200b", " ")
    text = text.replace("\ufeff", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def polygon_to_xywh(poly) -> Box:
    """Converts an OCR polygon into an x/y/width/height box."""
    xs = [int(p[0]) for p in poly]
    ys = [int(p[1]) for p in poly]

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    return x1, y1, x2 - x1, y2 - y1

def looks_like_date(text: str) -> bool:
    """Checks whether text looks like a chat date label."""
    t = text.strip()

    if re.fullmatch(
        r"(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)\s+\d{1,2}",
        t,
        flags=re.I,
    ):
        return True

    if re.search(
        r"(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)\s+\d{1,2},?\s+\d{4}",
        t,
        flags=re.I,
    ):
        return True

    return False

def looks_like_time(text: str) -> bool:
    """Checks whether text looks like an HH:MM time token."""
    t = text.strip()
    t = re.sub(r"\s*(vi|v|✓|✔|✔✔)+\s*$", "", t, flags=re.I)
    t = t.replace("*", ":").replace(",", ":").replace(";", ":").replace(".", ":")
    return bool(re.fullmatch(r"\d{1,2}:\d{2}", t))

def looks_like_date_or_time(text: str) -> bool:
    """Checks whether text is a date label or time token."""
    return looks_like_date(text) or looks_like_time(text)

def normalize_visible_time_token(text: str) -> Optional[str]:
    """Normalizes noisy OCR time text into HH:MM format."""
    t = text.strip()
    t = re.sub(r"\s*(vi|v|✓|✔|✔✔)+\s*$", "", t, flags=re.I)
    t = t.replace("*", ":").replace(",", ":").replace(";", ":").replace(".", ":")

    m = re.fullmatch(r"(\d{1,2}):(\d{2})", t)
    if not m:
        return None

    hh = int(m.group(1))
    mm = int(m.group(2))

    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        return None

    return f"{hh:02d}:{mm:02d}"

def parse_ocr_lines(ocr_data: str) -> List[Dict[str, str]]:
    """Parses positioned OCR debug text into row dictionaries."""
    pattern = re.compile(
        r"\[BLOCK\s+(?P<block>\d+)\]\s+"
        r"\[POS=(?P<pos>LEFT|RIGHT|CENTER)\]\s+"
        r"\[X=(?P<x>\d+)\]\s+"
        r"\[Y=(?P<y>\d+)\]\s+"
        r"\[W=(?P<w>\d+)\]\s+"
        r"\[H=(?P<h>\d+)\]"
        r"(?:\s+\[CONF=[^\]]+\])?\s+"
        r"(?P<text>.*)$",
        re.I,
    )

    current_screen = None
    rows = []

    for line in ocr_data.splitlines():
        sm = re.match(r"\[SCREEN\s+(\d+)\]", line.strip(), flags=re.I)
        if sm:
            current_screen = int(sm.group(1))
            continue

        bm = pattern.search(line.strip())
        if not bm:
            continue

        rows.append({
            "screen": str(current_screen or ""),
            "block": bm.group("block"),
            "pos": bm.group("pos").upper(),
            "x": bm.group("x"),
            "y": bm.group("y"),
            "w": bm.group("w"),
            "h": bm.group("h"),
            "text": bm.group("text").strip(),
        })

    return rows

def extract_allowed_times_from_ocr(screen_ocr: str) -> Set[str]:
    """Collects visible bubble times from OCR output."""
    # Notes:
    # Ignores top status bar time by requiring it to appear after the date separator/header area.
    rows = parse_ocr_lines(screen_ocr)
    allowed = set()

    date_y = None
    for row in rows:
        if looks_like_date(row["text"]):
            try:
                date_y = int(row["y"])
            except ValueError:
                pass

    for row in rows:
        try:
            y = int(row["y"])
        except ValueError:
            continue

        # Ignore status/header time near the top.
        if date_y is not None and y <= date_y:
            continue
        if date_y is None and y < 250:
            continue

        t = normalize_visible_time_token(row["text"])
        if t:
            allowed.add(t)

    return allowed

def month_to_number(month: str) -> Optional[int]:
    """Maps an English month name to its numeric month value."""
    m = month.strip().lower()[:3]
    table = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    return table.get(m)

def strip_code_fences(text: str) -> str:
    """Removes markdown code fences around model output."""
    text = text.strip()
    text = re.sub(r"^```(?:csv|json|text)?", "", text, flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()
    return text

def extract_json_object(text: str) -> str:
    """Extracts the outermost JSON object from model text."""
    text = strip_code_fences(text)
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return text

    return text[start:end + 1]

def ollama_chat_text(model: str, prompt: str) -> str:
    """Sends a deterministic text-only prompt to Ollama."""
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    return response["message"]["content"].strip()

def ollama_chat_screen(
    model: str,
    prompt: str,
    image_path: str,
    use_vision: bool = True
) -> str:
    """Sends a deterministic vision or OCR-only prompt to Ollama."""
    message = {
        "role": "user",
        "content": prompt,
    }

    if use_vision:
        message["images"] = [image_path]

    try:
        response = ollama.chat(
            model=model,
            messages=[message],
            options={"temperature": 0},
        )
        return response["message"]["content"].strip()

    except Exception as e:
        if use_vision:
            print(f"[WARNING] Vision call failed for {image_path}: {e}")
            print("[WARNING] Retrying with OCR text only. Emojis may not be reliable.")
            return ollama_chat_text(model, prompt)

        raise

def normalize_phone(value: str) -> str:
    """Keeps only digits from a phone/contact string."""
    return re.sub(r"\D+", "", value or "")

def normalize_name(value: str) -> str:
    """Lowercases and normalizes whitespace in a name."""
    return re.sub(r"\s+", " ", value or "").strip().lower()

def clean_name(value: str) -> str:
    """Cleans display names before matching or output."""
    value = str(value or "").strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s*\([^)]*\)\s*$", "", value).strip()
    return value

def same_name(a: str, b: str) -> bool:
    """Compares two names after normalization."""
    return normalize_name(clean_name(a)) == normalize_name(clean_name(b))

def name_in_text(name: str, text: str) -> bool:
    """Checks whether a normalized name appears inside normalized text."""
    name_norm = normalize_name(clean_name(name))
    text_norm = normalize_name(text)

    if not name_norm or not text_norm:
        return False

    return name_norm in text_norm

def build_actor_prompt(report_text: str) -> str:
    """Builds the fallback actor-extraction prompt."""
    return f"""
Extract the chat actors from this case report.

CASE REPORT:
{report_text}

Return only JSON with this structure:
{{
  "victim": "victim full name",
  "participants": [
    {{"name":"full name", "role":"victim or suspect", "contact_numbers":["..."]}}
  ]
}}

Rules:
1. Include the victim/complainant.
2. Include suspects/scammers and their contact numbers if present.
3. Do not output explanations.
"""

def infer_report_actors(report_text: str, model: str) -> Dict:
    """Extracts victim and suspect actors deterministically from the report."""
    # Notes:
    # We do NOT use the LLM here because wrong actor JSON breaks side_map.
    # Keeps the same function signature so the rest of the code does not change.
    participants = []

    # -------------------------
    # Victim
    # -------------------------
    victim = ""

    victim_patterns = [
        r"VICTIM\s*/\s*COMPLAINANT:.*?Full Name:\s*([^\n\r•]+)",
        r"Full Name:\s*([^\n\r•]+)",
        r"Target/Victim:\s*([^\n\r•]+)",
    ]

    for pattern in victim_patterns:
        m = re.search(pattern, report_text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            victim = clean_name(m.group(1))
            break

    if victim:
        participants.append({
            "name": victim,
            "role": "victim",
            "contact_numbers": []
        })

    # -------------------------
    # Suspects + nearby contact numbers
    # -------------------------
    suspect_pattern = re.compile(
        r"Suspect\s*\d+:\s*([^\n\r]+)(.*?)(?=(?:•\s*)?Suspect\s*\d+:|2\.\s*BACKGROUND|3\.\s*TECHNICAL|$)",
        flags=re.IGNORECASE | re.DOTALL
    )

    phone_pattern = re.compile(r"\+\d[\d\s().-]{5,}\d")

    for m in suspect_pattern.finditer(report_text):
        suspect_name = clean_name(m.group(1))
        suspect_block = m.group(2)

        phones = phone_pattern.findall(suspect_block)
        phones = [p.strip() for p in phones]

        if suspect_name:
            participants.append({
                "name": suspect_name,
                "role": "suspect",
                "contact_numbers": phones
            })

    # -------------------------
    # Fallback: Scammer Names Used
    # -------------------------
    if not any(p["role"] == "suspect" for p in participants):
        m = re.search(r"Scammer Names Used:\s*([^\n\r]+)", report_text, flags=re.IGNORECASE)
        if m:
            names = [clean_name(x) for x in re.split(r",| and ", m.group(1))]
            for name in names:
                if name and not any(same_name(name, p["name"]) for p in participants):
                    participants.append({
                        "name": name,
                        "role": "suspect",
                        "contact_numbers": []
                    })

    return {
        "victim": victim,
        "participants": participants
    }

def infer_report_actors_fallback(report_text: str) -> Dict:
    """Extracts actors with simpler regex fallbacks."""
    victim = ""
    participants = []

    m = re.search(r"Full Name:\s*([^\n•]+)", report_text, flags=re.I)
    if m:
        victim = clean_name(m.group(1))

    if victim:
        participants.append({
            "name": victim,
            "role": "victim",
            "contact_numbers": [],
        })

    for sm in re.finditer(r"Suspect\s*\d+:\s*([^\n]+)", report_text, flags=re.I):
        name = clean_name(sm.group(1))
        if name:
            participants.append({
                "name": name,
                "role": "suspect",
                "contact_numbers": [],
            })

    # Attach nearby phone numbers as a rough fallback.
    phones = re.findall(r"\+\d[\d\s().-]{5,}\d", report_text)
    suspect_i = 0
    for p in participants:
        if p["role"] == "suspect" and suspect_i < len(phones):
            p["contact_numbers"] = [phones[suspect_i]]
            suspect_i += 1

    return {
        "victim": victim,
        "participants": participants,
    }

def build_side_evidence(ocr_data: str) -> str:
    """Summarizes OCR text and phone-like strings by side."""
    rows = parse_ocr_lines(ocr_data)

    side_texts = {
        "LEFT": [],
        "RIGHT": [],
        "CENTER": [],
    }

    side_phones = {
        "LEFT": [],
        "RIGHT": [],
        "CENTER": [],
    }

    phone_pattern = re.compile(r"\+?\d[\d\s().-]{5,}\d")

    for row in rows:
        pos = row["pos"]
        text = row["text"]

        side_texts[pos].append(text)

        for phone in phone_pattern.findall(text):
            side_phones[pos].append(phone.strip())

    lines = ["SIDE EVIDENCE SUMMARY"]

    for side in ["LEFT", "RIGHT", "CENTER"]:
        lines.append(f"\n{side} phone/contact-like strings:")

        phones = side_phones[side][:12]
        if phones:
            lines.extend(f"- {p}" for p in phones)
        else:
            lines.append("- none")

        lines.append(f"{side} sample OCR texts:")
        for text in side_texts[side][:18]:
            lines.append(f"- {text}")

    return "\n".join(lines)

def force_date_and_year(time_value: str, visible_date: str, default_year: int) -> str:
    """Normalizes timestamps and forces the visible screenshot date."""
    # Notes:
    # Final format: DD/MM/YYYY HH:MM
    time_value = str(time_value).strip().strip('"')
    time_value = time_value.replace(";", ":")
    time_value = re.sub(r"\s+", " ", time_value)

    # Accept both:
    # DD/MM/YYYY, HH:MM
    # DD/MM/YYYY HH:MM
    m = re.search(
        r"(\d{1,2})/(\d{1,2})/(20\d{2})\s*,?\s*(\d{1,2})[:.;,*](\d{2})",
        time_value
    )

    if m:
        hh = int(m.group(4))
        mm = int(m.group(5))

        if visible_date:
            return f"{visible_date} {hh:02d}:{mm:02d}"

        dd = int(m.group(1))
        mo = int(m.group(2))
        return f"{dd:02d}/{mo:02d}/{default_year} {hh:02d}:{mm:02d}"

    # Fallback: if only HH:MM exists, combine with visible date.
    tv = normalize_visible_time_token(time_value)
    if tv and visible_date:
        return f"{visible_date} {tv}"

    return time_value

def extract_hhmm_from_full_time(time_value: str) -> Optional[str]:
    # Accept both:
    # DD/MM/YYYY, HH:MM
    # DD/MM/YYYY HH:MM
    """Extracts the HH:MM suffix from a full timestamp."""
    m = re.search(r"\s*,?\s*(\d{1,2})[:.;,*](\d{2})\s*$", time_value)
    if not m:
        return None

    hh = int(m.group(1))
    mm = int(m.group(2))

    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        return None

    return f"{hh:02d}:{mm:02d}"

def strip_emojis(text: str) -> str:
    """Removes emoji and symbol ranges from message text."""
    # Notes:
    # This is intentionally generic and not tied to specific emoji characters.
    emoji_re = re.compile(
        "["
        "\U0001F1E6-\U0001F1FF"  # flags
        "\U0001F300-\U0001F5FF"  # symbols/pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport/map
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FAFF"
        "\u2600-\u26FF"
        "\u2700-\u27BF"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_re.sub("", str(text or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_text_for_side_overlap(text: str) -> Set[str]:
    """Tokenizes text for fuzzy side-overlap matching."""
    # Notes:
    # This is not used as final transcript text.
    t = str(text or "").lower()
    t = re.sub(r"\b([a-z]+)'\$", r"\1s", t)
    t = t.replace("|", "i").replace("!", "i")
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return {w for w in t.split() if len(w) >= 2}

def count_data_rows(side_csv: str) -> int:
    """Counts non-header data rows in a side CSV."""
    rows = list(csv.reader(io.StringIO(strip_code_fences(side_csv))))
    return sum(1 for r in rows if r and len(r) >= 3 and r[0].strip().lower() != "time")

def remove_leaked_time_tokens_from_message(message: str) -> str:
    """Remove standalone HH:MM-like tokens accidentally copied into message text."""
    text = str(message or "")
    # Remove times such as 08:16, 11,01 or 10.32 when they are standalone OCR leaks.
    text = re.sub(r"(?<![A-Za-z0-9])\d{1,2}[:.,;]\d{2}(?![A-Za-z0-9])", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def looks_like_noisy_ocr_text(message: str) -> bool:
    """Detect OCR-heavy text that should not be trusted as final transcript text."""
    text = str(message or "")
    if not text.strip():
        return True

    # Standalone leaked timestamps are a strong signal that OCR text crossed bubble boundaries.
    if re.search(r"(?<![A-Za-z0-9])\d{1,2}[:.,;]\d{2}(?![A-Za-z0-9])", text):
        return True

    # Too many OCR-only artifacts for one message.
    artifact_patterns = [
        r"\bTII\w*\b",
        r"\bI{2}l\b",
        r"\b1['’]?I[lI]\b",
        r"\bJi00\b",
        r"\$",
        r"\|",
        r"__",
        r"=",
        r"\byoU\b",
        r"\bsO\b",
    ]
    hits = sum(1 for pat in artifact_patterns if re.search(pat, text))
    return hits >= 2

def conservative_clean_message_text(message: str) -> str:
    """Apply generic OCR cleanup without semantic rewriting or case-specific replacements."""
    msg = str(message or "").strip()

    msg = msg.replace('\\"', '"')
    msg = msg.replace("“", '"').replace("”", '"')
    msg = msg.replace("‘", "'").replace("’", "'")
    msg = msg.replace("\u200b", " ").replace("\ufeff", " ")

    msg = remove_leaked_time_tokens_from_message(msg)

    pronoun_verbs = (
        r"am|was|will|can|can't|cannot|need|have|think|feel|want|would|could|"
        r"should|do|don't|dont|love|trust|promise|know|hope|wish|believe|already|"
        r"just|may|must|might|see|ask|tell|try|send|keep|call|go|complete|look|admire"
    )
    msg = re.sub(
        rf"(?i)(^|[\s,.;:!?])\|\s+({pronoun_verbs})\b",
        lambda m: f"{m.group(1)}I {m.group(2)}",
        msg,
    )

    compact_i = {
        "just": "just",
        "will": "will",
        "need": "need",
        "love": "love",
        "may": "may",
        "feel": "feel",
        "already": "already",
        "cant": "can't",
        "can't": "can't",
        "can": "can",
        "have": "have",
        "think": "think",
        "want": "want",
        "would": "would",
        "could": "could",
        "should": "should",
        "promise": "promise",
        "believe": "believe",
        "hope": "hope",
        "see": "see",
    }
    for compact, word in compact_i.items():
        msg = re.sub(rf"\bI{re.escape(compact)}\b", f"I {word}", msg, flags=re.IGNORECASE)

    msg = re.sub(r"\bI\s*m\b", "I'm", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bIm\b", "I'm", msg)
    msg = re.sub(r"\bTm\b", "I'm", msg)
    msg = re.sub(r"\bI\s*ve\b", "I've", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bFve\b", "I've", msg)
    msg = re.sub(r"\bI\s*ll\b", "I'll", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bIll\b", "I'll", msg)
    msg = re.sub(r"\bIIl\b", "I'll", msg)
    msg = re.sub(r"\b1['’]?I[lI]\b", "I'll", msg)
    msg = re.sub(r"\b1['’]?ll\b", "I'll", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bTII\s*keep\b", "I'll keep", msg, flags=re.IGNORECASE)

    msg = re.sub(r"\b([A-Za-z]+)'\$\b", r"\1's", msg)
    msg = re.sub(r"\bit\$\b", "it's", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bthat\$\b", "that's", msg, flags=re.IGNORECASE)

    msg = re.sub(r"\b([Ii])t\s*[\"]+\s*s\b", "It's", msg)
    msg = re.sub(r"\b([Tt])hat\s*[\"]+\s*s\b", "That's", msg)
    msg = re.sub(r"\b([Dd])on\s*[\"]+\s*t\b", "don't", msg)
    msg = re.sub(r"\b([Cc])an\s*[\"]+\s*t\b", "can't", msg)

    msg = re.sub(r"\bIcant\b", "I can't", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bIcan't\b", "I can't", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bdont\b", "don't", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\byoure\b", "you're", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bthats\b", "that's", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bits\b", "it's", msg)

    msg = re.sub(r"\bsO\b", "so", msg)
    msg = re.sub(r"\byoU\b", "you", msg)
    msg = re.sub(r"\bJi00\b", "I", msg)
    msg = re.sub(r"\bIı\b", "I", msg)

    msg = msg.replace("=", " ")
    msg = msg.replace("__", "...")
    msg = msg.replace("_", "")

    label_like = (
        r"name|country|city|option|account|iban|reference|mtcn|phone|email|"
        r"amount|details|receiver|sender|beneficiary|bank|address|code|number|date|time"
    )
    has_structured_label = re.search(rf"\b({label_like})\s*:", msg, flags=re.IGNORECASE)
    if not has_structured_label:
        msg = re.sub(
            r";\s+(?=(I|I'm|I'll|you|we|it|that|this|the|they|there|but|and|more|please|thank)\b)",
            ", ",
            msg,
            flags=re.IGNORECASE,
        )
        msg = re.sub(
            r":\s+(?=(I|I'm|I'll|you|we|it|that|this|the|they|there|but|and|because|so|please|thank)\b)",
            ". ",
            msg,
            flags=re.IGNORECASE,
        )

    msg = msg.replace(":_.", "...").replace(":-", "...").replace("_.", "...")
    msg = re.sub(r"\.{4,}", "...", msg)
    msg = re.sub(r"\s+", " ", msg).strip()
    msg = re.sub(r"\s+([,.;:!?])", r"\1", msg)
    msg = re.sub(r"([,.;:!?])(?=[A-Za-z])", r"\1 ", msg)
    msg = re.sub(r"\s+\.\.\.", "...", msg)

    if msg.endswith(":") and not re.search(rf"\b({label_like})\s*:$", msg, flags=re.IGNORECASE):
        if len(re.findall(r"\b\w+\b", msg)) >= 4:
            msg = msg[:-1] + "."

    return msg.strip()

def _side_csv_rows(side_csv: str) -> List[List[str]]:
    """Read Time/Side/Message rows from a side CSV."""
    rows: List[List[str]] = []
    reader = csv.reader(io.StringIO(strip_code_fences(side_csv)))
    for row in reader:
        if not row:
            continue
        if len(row) >= 3 and row[0].strip().lower() == "time":
            continue
        if len(row) < 3:
            continue
        time_value = row[0].strip()
        side = row[1].strip().upper()
        message = ",".join(row[2:]).strip() if len(row) > 3 else row[2].strip()
        if time_value and side in {"LEFT", "RIGHT"} and message:
            rows.append([time_value, side, message])
    return rows

def normalize_message_for_similarity(message: str) -> str:
    """Normalize message text for duplicate/continuation checks only."""
    text = strip_emojis(str(message or ""))
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def message_similarity_ratio(a: str, b: str) -> float:
    """Return a generic similarity ratio for two message strings."""
    a_norm = normalize_message_for_similarity(a)
    b_norm = normalize_message_for_similarity(b)
    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()

def _datetime_from_side_time(time_value: str):
    """Parse transcript timestamp strings when possible."""
    from datetime import datetime
    text = str(time_value or "").strip()
    for fmt in ("%d/%m/%Y %H:%M", "%d/%m/%Y, %H:%M", "%d/%m/%Y,%H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None

def _minutes_apart(a: str, b: str) -> Optional[float]:
    """Return absolute minute difference between two row timestamps."""
    da = _datetime_from_side_time(a)
    db = _datetime_from_side_time(b)
    if da is None or db is None:
        return None
    return abs((da - db).total_seconds()) / 60.0

def _same_day(a: str, b: str) -> bool:
    """Check if two timestamp strings share the same DD/MM/YYYY prefix."""
    return str(a or "")[:10] == str(b or "")[:10]

def _message_quality_score(message: str) -> Tuple[int, int, int]:
    """Prefer fuller, cleaner messages when removing near duplicates."""
    text = str(message or "").strip()
    words = normalize_message_for_similarity(text).split()
    terminal = 1 if re.search(r"[.!?…]$", text) else 0
    odd = len(re.findall(r"[^\w\s,.;:!?€£$+@%/'\"()\-]", text))
    return (len(words), len(text), terminal - odd)

def are_near_duplicate_messages(a: str, b: str) -> bool:
    """Detect adjacent duplicates with minor OCR/VLM differences."""
    a_norm = normalize_message_for_similarity(a)
    b_norm = normalize_message_for_similarity(b)
    if not a_norm or not b_norm:
        return False

    if a_norm == b_norm:
        return True

    shorter, longer = sorted([a_norm, b_norm], key=len)
    if len(shorter.split()) >= 4 and shorter in longer:
        return True

    ratio = SequenceMatcher(None, a_norm, b_norm).ratio()
    return ratio >= 0.88

def looks_like_orphan_fragment_message(message: str) -> bool:
    """Detect likely OCR/VLM fragments that should trigger repair or cautious merge."""
    raw = str(message or "").strip()
    if not raw:
        return False

    text = strip_emojis(raw).strip()
    words = re.findall(r"[A-Za-z']+", text)
    if not words:
        return False

    first = words[0].lower().strip("'")
    word_count = len(words)
    starts_lower = bool(text[:1].islower())

    if starts_lower and "?" in text and word_count <= 8:
        return True

    fragment_starts = {
        "and", "but", "because", "so", "that", "which", "with", "without",
        "for", "to", "of", "in", "on", "at", "by", "from", "as", "than",
        "my", "your", "our", "their", "the", "a", "an", "there", "then", "right",
    }
    if starts_lower and first in fragment_starts and word_count <= 8:
        return True

    if re.search(r"\b[a-z]{2,}\s+(The|I|I'm|I'll|You|We|They)\b", text) and word_count <= 12:
        return True

    return False

def should_merge_continuation(prev_message: str, message: str) -> bool:
    """Conservatively merge only wrapped-line fragments, not normal adjacent bubbles."""
    prev = str(prev_message or "").strip()
    cur = str(message or "").strip()
    if not prev or not cur:
        return False

    cur_words = re.findall(r"[A-Za-z0-9']+", cur)
    if not cur_words:
        return False

    prev_finished = bool(re.search(r"[.!?…]$", prev))
    first = re.sub(r"[^A-Za-z']+", "", cur_words[0]).lower()
    starts_lower = bool(cur[:1].islower())

    # Never merge a lowercase-start question into a previous completed message;
    # it is suspicious and should be repaired/reordered from the image instead.
    if starts_lower and "?" in cur:
        return False

    continuation_starts = {
        "and", "but", "because", "so", "that", "which", "with", "without",
        "for", "to", "of", "in", "on", "at", "by", "from", "as", "than",
        "my", "your", "our", "their", "the", "a", "an", "better"
    }

    # A current row after an unfinished previous row can be a wrapped line even if
    # it has more than three words. Keep the limit modest to avoid merging bubbles.
    if not prev_finished and len(cur_words) <= 8:
        if starts_lower or first in continuation_starts:
            return True

    # Tiny fragments are safe to absorb when the previous row is unfinished.
    if not prev_finished and len(cur_words) <= 3:
        return True

    # Previous row ending with comma/colon/semicolon can absorb a tiny continuation.
    if len(cur_words) <= 2 and re.search(r"[,;:]$", prev) and first not in {"yes", "no", "ok", "okay"}:
        return True

    return False

def split_side_row_by_known_boundaries(row: List[str]) -> List[List[str]]:
    """Return a cleaned side-CSV row without dataset-specific transcript rewrites."""
    time_value, side, message = row
    msg = conservative_clean_message_text(message)
    if not msg:
        return []
    return [[time_value, side, msg]]

def postprocess_side_csv_rows(side_csv: str) -> str:
    """Generic side-CSV cleanup: near-duplicate removal and safe continuation merges."""
    rows = []
    for _row in _side_csv_rows(side_csv):
        rows.extend(split_side_row_by_known_boundaries(_row))

    # First remove exact and adjacent near duplicates.
    deduped: List[List[str]] = []
    seen_exact = set()
    for row in rows:
        item = tuple(row)
        if item in seen_exact:
            continue
        seen_exact.add(item)

        if deduped:
            prev = deduped[-1]
            close_time = _minutes_apart(prev[0], row[0])
            close_enough = close_time is None or close_time <= 2
            if prev[1] == row[1] and _same_day(prev[0], row[0]) and close_enough and are_near_duplicate_messages(prev[2], row[2]):
                if _message_quality_score(row[2]) > _message_quality_score(prev[2]):
                    # Keep the better text but preserve the earlier timestamp.
                    deduped[-1] = [prev[0], prev[1], row[2]]
                continue
        deduped.append(row)

    # Then merge obvious wrapped/fragments from the same side.
    merged: List[List[str]] = []
    for row in deduped:
        if merged:
            prev = merged[-1]
            close_time = _minutes_apart(prev[0], row[0])
            close_enough = close_time is None or close_time <= 1
            if prev[1] == row[1] and _same_day(prev[0], row[0]) and close_enough and should_merge_continuation(prev[2], row[2]):
                prev[2] = conservative_clean_message_text(prev[2] + " " + row[2])
                continue
        row[2] = conservative_clean_message_text(row[2])
        merged.append(row)

    out = io.StringIO()
    writer = csv.writer(out, quoting=csv.QUOTE_ALL, lineterminator="\n")
    writer.writerow(["Time", "Side", "Message"])

    final_seen = set()
    for row in merged:
        item = tuple(row)
        if item in final_seen:
            continue
        final_seen.add(item)
        writer.writerow(row)

    return out.getvalue().strip() + "\n"

def side_csv_needs_repair(side_csv: str, expected_bubble_count: int = 0) -> bool:
    """Detect whether a screen CSV needs a second VLM repair pass."""
    rows = _side_csv_rows(side_csv)
    row_count = len(rows)

    if row_count == 0:
        return True

    if expected_bubble_count > 0:
        # A one-row difference is common from OCR UI noise, but larger gaps indicate split/merge trouble.
        if abs(row_count - expected_bubble_count) >= 2:
            return True

    suspicious_fragment_count = 0

    for i, row in enumerate(rows):
        message = row[2]
        if re.search(r"\b\d{1,2}[:.,;]\d{2}\b", message):
            return True
        if looks_like_orphan_fragment_message(message):
            suspicious_fragment_count += 1
            return True
        if len(message) >= 220 and re.search(r"[.!?].+\b(I|You|We|They|Please|Thank|The|There)\b", message):
            return True
        # Suspicious OCR word-order: lowercase token before a new capitalized start.
        if re.search(r"\b(they|can|the|and|but)\s+(The|I|I'm|I'll|You|We|They)\b", message):
            return True
        if i > 0:
            prev = rows[i - 1]
            if prev[1] == row[1] and _same_day(prev[0], row[0]) and are_near_duplicate_messages(prev[2], message):
                return True
            close_time = _minutes_apart(prev[0], row[0])
            if prev[1] == row[1] and (close_time is None or close_time <= 1) and should_merge_continuation(prev[2], message):
                return True

    return False

def merge_side_csvs(csv_parts: List[str]) -> str:
    """Merges per-screen side CSV parts while removing exact duplicates."""
    out = io.StringIO()
    writer = csv.writer(out, quoting=csv.QUOTE_ALL, lineterminator="\n")

    writer.writerow(["Time", "Side", "Message"])

    seen = set()

    for part in csv_parts:
        reader = csv.reader(io.StringIO(strip_code_fences(part)))

        for row in reader:
            if not row:
                continue

            if len(row) >= 3 and row[0].strip().lower() == "time":
                continue

            if len(row) != 3:
                continue

            time_value = row[0].strip()
            side = row[1].strip().upper()
            message = row[2].strip()

            if side not in {"LEFT", "RIGHT"} or not time_value or not message:
                continue

            item = (time_value, side, message)
            if item in seen:
                continue

            seen.add(item)
            writer.writerow(item)

    return out.getvalue().strip() + "\n"

def apply_side_mapping(side_csv: str, side_map: Dict[str, str]) -> str:
    """Converts LEFT/RIGHT side rows into Sender/Receiver rows."""
    reader = csv.reader(io.StringIO(strip_code_fences(side_csv)))

    out = io.StringIO()
    writer = csv.writer(out, quoting=csv.QUOTE_ALL, lineterminator="\n")

    writer.writerow(["Time", "Sender", "Receiver", "Message"])

    seen = set()

    for row in reader:
        if not row:
            continue

        if len(row) >= 3 and row[0].strip().lower() == "time":
            continue

        if len(row) != 3:
            continue

        time_value = row[0].strip()
        side = row[1].strip().upper()
        message = row[2].strip()

        if side not in {"LEFT", "RIGHT"} or not time_value or not message:
            continue

        sender = side_map[side]
        receiver = side_map["RIGHT"] if side == "LEFT" else side_map["LEFT"]

        item = (time_value, sender, receiver, message)
        if item in seen:
            continue

        seen.add(item)
        writer.writerow(item)

    return out.getvalue().strip() + "\n"

def write_crop(path: Path, crop: np.ndarray) -> None:
    """Writes an image crop to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), crop)

def _token_counter(text: str) -> Dict[str, int]:
    """Builds a lowercase token counter for generic coverage scoring."""
    counter: Dict[str, int] = {}
    for token in normalize_message_for_similarity(text).split():
        counter[token] = counter.get(token, 0) + 1
    return counter

def _token_overlap_ratio(a: str, b: str) -> float:
    """Measures how much of the shorter text is covered by the other text."""
    a_counter = _token_counter(a)
    b_counter = _token_counter(b)
    if not a_counter or not b_counter:
        return 0.0

    overlap = 0
    for token, count in a_counter.items():
        overlap += min(count, b_counter.get(token, 0))

    a_total = sum(a_counter.values())
    b_total = sum(b_counter.values())
    return overlap / max(1, min(a_total, b_total))

def _candidate_rows_for_scoring(side_csv: str) -> List[List[str]]:
    """Returns cleaned rows used only for candidate scoring."""
    rows: List[List[str]] = []
    for time_value, side, message in _side_csv_rows(side_csv):
        message = conservative_clean_message_text(message)
        if message:
            rows.append([time_value, side, message])
    return rows

def _bubble_groups_for_scoring(bubble_groups: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    """Filters OCR/VLM bubble hints down to transcript-like groups."""
    if not bubble_groups:
        return []

    useful: List[Dict[str, str]] = []
    for group in bubble_groups:
        text = str(group.get("text", "")).strip()
        side = str(group.get("side", "")).upper()
        if side not in {"LEFT", "RIGHT"}:
            continue
        if len(normalize_message_for_similarity(text).split()) < 2:
            continue
        useful.append(group)
    return useful

def side_csv_bubble_coverage(
    side_csv: str,
    bubble_groups: Optional[List[Dict[str, str]]] = None,
    min_overlap: float = 0.58,
) -> Tuple[int, int, float]:
    """Scores how many visible OCR bubble hints are represented in a side CSV.

    This is a generic completeness signal. It does not rewrite transcript text and
    it does not contain dataset-specific phrases.
    """
    rows = _candidate_rows_for_scoring(side_csv)
    groups = _bubble_groups_for_scoring(bubble_groups)
    if not groups:
        return 0, 0, 1.0

    covered = 0
    for group in groups:
        group_text = str(group.get("text", "")).strip()
        group_side = str(group.get("side", "")).upper()
        best = 0.0
        for _, row_side, row_message in rows:
            if group_side in {"LEFT", "RIGHT"} and row_side in {"LEFT", "RIGHT"} and group_side != row_side:
                continue
            best = max(best, _token_overlap_ratio(group_text, row_message))
        if best >= min_overlap:
            covered += 1

    return covered, len(groups), covered / max(1, len(groups))

def _candidate_suspicion_penalty(rows: List[List[str]]) -> int:
    """Penalizes structurally suspicious rows without using case-specific content."""
    penalty = 0
    for i, row in enumerate(rows):
        message = row[2]
        if re.search(r"\b\d{1,2}[:.,;]\d{2}\b", message):
            penalty += 4
        if looks_like_orphan_fragment_message(message):
            penalty += 3
        if len(message) >= 240:
            penalty += 2
        if looks_like_noisy_ocr_text(message):
            penalty += 2
        if i > 0:
            prev = rows[i - 1]
            if prev[1] == row[1] and _same_day(prev[0], row[0]) and are_near_duplicate_messages(prev[2], message):
                penalty += 3
            close_time = _minutes_apart(prev[0], row[0])
            if prev[1] == row[1] and (close_time is None or close_time <= 1) and should_merge_continuation(prev[2], message):
                penalty += 2
    return penalty

def score_side_csv_candidate(
    side_csv: str,
    allowed_times: Optional[Set[str]] = None,
    expected_bubble_count: int = 0,
    bubble_groups: Optional[List[Dict[str, str]]] = None,
) -> float:
    """Returns a generic quality score for a draft/repair side CSV candidate."""
    rows = _candidate_rows_for_scoring(side_csv)
    if not rows:
        return -10_000.0

    allowed_times = allowed_times or set()
    row_count = len(rows)
    score = row_count * 2.0

    if expected_bubble_count > 0:
        count_gap = abs(row_count - expected_bubble_count)
        score -= count_gap * 6.0
        if row_count < expected_bubble_count:
            score -= (expected_bubble_count - row_count) * 4.0

    covered, total, coverage_ratio = side_csv_bubble_coverage(
        side_csv,
        bubble_groups=bubble_groups,
    )
    if total:
        score += coverage_ratio * 70.0
        score += covered * 1.5

    invalid_time_count = 0
    if allowed_times:
        for time_value, _, _ in rows:
            hhmm = extract_hhmm_from_full_time(time_value)
            if hhmm and hhmm not in allowed_times:
                invalid_time_count += 1
    score -= invalid_time_count * 5.0
    score -= _candidate_suspicion_penalty(rows) * 4.0

    return score

def _best_bubble_match_for_row(
    row: List[str],
    bubble_groups: Optional[List[Dict[str, str]]] = None,
    min_overlap: float = 0.58,
) -> Tuple[Optional[Tuple[str, int]], float]:
    """Find the best generic OCR-bubble hint match for one side-CSV row."""
    groups = _bubble_groups_for_scoring(bubble_groups)
    if not groups:
        return None, 0.0

    _, row_side, row_message = row
    best_key: Optional[Tuple[str, int]] = None
    best_score = 0.0

    for fallback_order, group in enumerate(groups):
        group_side = str(group.get("side", "")).upper()
        if group_side in {"LEFT", "RIGHT"} and row_side in {"LEFT", "RIGHT"} and group_side != row_side:
            continue

        score = _token_overlap_ratio(row_message, str(group.get("text", "")))
        if score > best_score:
            try:
                order = int(group.get("order", fallback_order))
            except Exception:
                order = fallback_order
            best_score = score
            best_key = (group_side, order)

    if best_key is not None and best_score >= min_overlap:
        return best_key, best_score

    return None, best_score

def _rows_are_generic_duplicates(a: List[str], b: List[str]) -> bool:
    """Detect duplicate candidate rows without transcript-specific phrases."""
    if a[1] != b[1]:
        return False

    a_norm = normalize_message_for_similarity(a[2])
    b_norm = normalize_message_for_similarity(b[2])
    if not a_norm or not b_norm:
        return False

    if a_norm == b_norm:
        return True

    shorter, longer = sorted([a_norm, b_norm], key=len)
    if len(shorter.split()) >= 3 and shorter in longer:
        return True

    return message_similarity_ratio(a[2], b[2]) >= 0.86

def _row_has_valid_visible_time(row: List[str], allowed_times: Optional[Set[str]]) -> bool:
    """Validate row time against visible OCR times when available."""
    if not allowed_times:
        return True
    hhmm = extract_hhmm_from_full_time(row[0])
    return bool(hhmm and hhmm in allowed_times)

def merge_side_csv_candidates_additive(
    draft_norm: str,
    repaired_norm: str,
    allowed_times: Optional[Set[str]] = None,
    expected_bubble_count: int = 0,
    bubble_groups: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Build a conservative additive candidate from draft + repair outputs.

    The draft is used as the baseline. Rows from the repair pass are added only
    when they appear to cover a visible OCR bubble not already represented by
    the draft, or when the row count is clearly below the expected bubble count.
    This is a general recall-improvement step: it uses geometry/OCR bubble
    coverage, not dataset-specific names, locations, amounts, or message text.
    """
    allowed_times = allowed_times or set()
    draft_rows = [r for r in _candidate_rows_for_scoring(draft_norm) if _row_has_valid_visible_time(r, allowed_times)]
    repair_rows = [r for r in _candidate_rows_for_scoring(repaired_norm) if _row_has_valid_visible_time(r, allowed_times)]

    if not draft_rows:
        return repaired_norm
    if not repair_rows:
        return draft_norm

    rows: List[Tuple[int, int, Optional[Tuple[str, int]], List[str]]] = []
    covered_bubbles: Set[Tuple[str, int]] = set()

    for idx, row in enumerate(draft_rows):
        key, _score = _best_bubble_match_for_row(row, bubble_groups=bubble_groups)
        if key is not None:
            covered_bubbles.add(key)
        rows.append((0, idx, key, row))

    for idx, row in enumerate(repair_rows):
        if any(_rows_are_generic_duplicates(row, existing_row) for *_unused, existing_row in rows):
            continue

        key, match_score = _best_bubble_match_for_row(row, bubble_groups=bubble_groups)
        suspicious = looks_like_noisy_ocr_text(row[2]) or looks_like_orphan_fragment_message(row[2])

        should_add = False
        if key is not None and key not in covered_bubbles:
            # Add only rows that map to a previously uncovered visible bubble.
            should_add = True
        elif not _bubble_groups_for_scoring(bubble_groups) and expected_bubble_count > len(rows):
            # Fallback for screenshots with no usable bubble hints: add only if
            # the output is still below the expected bubble count.
            should_add = not suspicious
        elif expected_bubble_count > len(rows) and match_score >= 0.72:
            # Conservative fallback: a strong hint match can fill an under-counted screen.
            should_add = not suspicious

        if not should_add:
            continue

        if key is not None:
            covered_bubbles.add(key)
        rows.append((1, idx, key, row))

    if len(rows) == len(draft_rows):
        return draft_norm

    # Sort by visible bubble order when known; otherwise keep baseline/repair order.
    rows.sort(
        key=lambda item: (
            item[2] is None,
            item[2][1] if item[2] is not None else item[1],
            item[0],
            item[1],
        )
    )

    out = io.StringIO()
    writer = csv.writer(out, quoting=csv.QUOTE_ALL, lineterminator="\n")
    writer.writerow(["Time", "Side", "Message"])

    emitted: List[List[str]] = []
    for _source, _idx, _key, row in rows:
        if any(_rows_are_generic_duplicates(row, prev) for prev in emitted):
            continue
        emitted.append(row)
        writer.writerow(row)

    return out.getvalue().strip() + "\n"

def choose_best_screen_side_csv(
    draft_norm: str,
    repaired_norm: str,
    allowed_times: Set[str],
    expected_bubble_count: int = 0,
    bubble_groups: Optional[List[Dict[str, str]]] = None,
    enable_additive: bool = True,
) -> str:
    """Choose the safest candidate using platform-appropriate generic signals.

    When enable_additive is True, a third candidate is built from draft rows plus
    repair rows that cover previously uncovered OCR bubble hints. This is useful
    for platforms where hidden/shared timestamps make missing-bubble recovery
    more likely. When enable_additive is False, selection is limited to the
    conservative draft-vs-repair comparison.
    """
    draft_count = count_data_rows(draft_norm)
    repair_count = count_data_rows(repaired_norm)

    if repair_count == 0 and draft_count > 0:
        return draft_norm
    if draft_count == 0 and repair_count > 0:
        return repaired_norm
    if draft_count == 0 and repair_count == 0:
        return draft_norm

    visible_limit = expected_bubble_count if expected_bubble_count > 0 else len(allowed_times)
    if visible_limit > 0:
        max_reasonable = max(visible_limit + 3, draft_count + 3)
    else:
        max_reasonable = draft_count + 3

    candidates: List[Tuple[str, str]] = [("draft", draft_norm)]

    if repair_count <= max_reasonable:
        draft_cov = side_csv_bubble_coverage(draft_norm, bubble_groups=bubble_groups)[2]
        repair_cov = side_csv_bubble_coverage(repaired_norm, bubble_groups=bubble_groups)[2]
        if not (repair_count <= max(1, draft_count - 2) and repair_cov <= draft_cov + 0.05):
            candidates.append(("repair", repaired_norm))

    if enable_additive:
        additive_norm = merge_side_csv_candidates_additive(
            draft_norm=draft_norm,
            repaired_norm=repaired_norm,
            allowed_times=allowed_times,
            expected_bubble_count=expected_bubble_count,
            bubble_groups=bubble_groups,
        )
        additive_count = count_data_rows(additive_norm)
        if additive_count <= max_reasonable and additive_norm not in {draft_norm, repaired_norm}:
            candidates.append(("additive", additive_norm))

    scored: List[Tuple[float, int, str, str]] = []
    for name, candidate in candidates:
        score = score_side_csv_candidate(
            candidate,
            allowed_times=allowed_times,
            expected_bubble_count=expected_bubble_count,
            bubble_groups=bubble_groups,
        )
        scored.append((score, count_data_rows(candidate), name, candidate))

    draft_score = next(score for score, _count, name, _candidate in scored if name == "draft")
    best_score, _best_count, best_name, best_candidate = max(scored, key=lambda item: (item[0], item[1]))

    # Hysteresis: do not replace the draft for tiny score changes.
    # Additive candidates need a smaller margin because they preserve the draft
    # and only add uncovered evidence-backed rows.
    margin = 0.35 if best_name == "additive" else 1.0
    if best_score >= draft_score + margin:
        return best_candidate

    return draft_norm

# ============================================================
# GENERIC TEXT-POLISH CANDIDATE GUARD
# ============================================================
def _raw_side_csv_rows_for_polish(side_csv: str) -> List[List[str]]:
    """Parse Time/Side/Message rows for text-polish validation."""
    rows: List[List[str]] = []
    try:
        reader = csv.reader(io.StringIO(strip_code_fences(side_csv)))
        for row in reader:
            if not row:
                continue
            if len(row) >= 3 and row[0].strip().lower() == "time":
                continue
            if len(row) < 3:
                continue
            time_value = str(row[0] or "").strip()
            side = str(row[1] or "").strip().upper()
            message = ",".join(row[2:]).strip() if len(row) > 3 else str(row[2] or "").strip()
            if time_value and side in {"LEFT", "RIGHT"} and message:
                rows.append([time_value, side, message])
    except Exception:
        return []
    return rows

def normalize_polished_side_csv_against_reference(
    polished_csv: str,
    reference_csv: str,
    emoji_mode: str = "omit",
) -> str:
    """Normalize a text-polish CSV while preserving reference row structure.

    The polish pass is allowed to edit only Message text. This helper keeps the
    original Time and Side fields from the reference CSV and rejects candidates
    whose row count does not match exactly. This protects row-level precision,
    recall, and timestamp/side structure while still allowing generic spelling,
    casing, apostrophe, and punctuation improvements.
    """
    ref_rows = _raw_side_csv_rows_for_polish(reference_csv)
    pol_rows = _raw_side_csv_rows_for_polish(polished_csv)

    if not ref_rows or len(ref_rows) != len(pol_rows):
        return ""

    out = io.StringIO()
    writer = csv.writer(out, quoting=csv.QUOTE_ALL, lineterminator="\n")
    writer.writerow(["Time", "Side", "Message"])

    for ref, pol in zip(ref_rows, pol_rows):
        message = conservative_clean_message_text(pol[2])
        if emoji_mode == "omit":
            message = strip_emojis(message)
        if not message:
            return ""
        writer.writerow([ref[0], ref[1], message])

    return out.getvalue().strip() + "\n"

def _token_norm_for_text_guard(text: str) -> str:
    """Normalize text for semantic-stability checks in the polish guard."""
    text = strip_emojis(str(text or ""))
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.lower()
    # Normalize common apostrophe/no-apostrophe forms for the guard only.
    text = re.sub(r"\bim\b", "i am", text)
    text = re.sub(r"\bi'm\b", "i am", text)
    text = re.sub(r"\bcant\b", "can not", text)
    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"\bdont\b", "do not", text)
    text = re.sub(r"\bdon't\b", "do not", text)
    text = re.sub(r"\byoure\b", "you are", text)
    text = re.sub(r"\byou're\b", "you are", text)
    text = re.sub(r"\bthats\b", "that is", text)
    text = re.sub(r"\bthat's\b", "that is", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def _text_guard_similarity(a: str, b: str) -> float:
    """Return similarity after guard normalization."""
    a_norm = _token_norm_for_text_guard(a)
    b_norm = _token_norm_for_text_guard(b)
    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()

def _text_artifact_penalty(message: str) -> float:
    """Generic penalty for OCR/punctuation artifacts, not dataset phrases."""
    text = str(message or "")
    penalty = 0.0

    artifact_patterns = [
        r"\bI{2}l\b",
        r"\b1['’]?I[lI]\b",
        r"\bTII\b",
        r"\bJi00\b",
        r"\bIcant\b",
        r"\bIcan't\b",
        r"\bIjust\b",
        r"\bIwill\b",
        r"\bIlove\b",
        r"\byoU\b",
        r"\bsO\b",
        r"\|",
        r"__",
        r"=",
        r"(?<![A-Za-z0-9])\d{1,2}[:.,;]\d{2}(?![A-Za-z0-9])",
    ]
    for pattern in artifact_patterns:
        penalty += 2.0 * len(re.findall(pattern, text))

    penalty += 1.0 * len(re.findall(r"\s+[,.;:!?]", text))
    penalty += 0.5 * len(re.findall(r"[,.;:!?](?=[A-Za-z])", text))
    penalty += 1.0 * len(re.findall(r"\.{4,}", text))
    penalty += 1.0 * len(re.findall(r"[!?]{3,}", text))

    # Unbalanced plain double quotes are often OCR/polish damage.
    if text.count('"') % 2 == 1:
        penalty += 1.0

    words = re.findall(r"[A-Za-z']+", text)
    if len(words) >= 4 and text[:1].islower():
        penalty += 0.75

    return penalty

def _text_style_score(rows: List[List[str]]) -> float:
    """Score generic transcript text cleanliness for polish selection."""
    score = 0.0
    for _time_value, _side, message in rows:
        msg = str(message or "").strip()
        if not msg:
            score -= 20.0
            continue
        score -= _text_artifact_penalty(msg)
        # Very noisy rows should not be preferred, but do not over-penalize
        # ordinary informal chat without terminal punctuation.
        if looks_like_noisy_ocr_text(msg):
            score -= 5.0
        if looks_like_orphan_fragment_message(msg):
            score -= 2.0
    return score

def choose_text_polished_side_csv(
    reference_csv: str,
    polished_csv: str,
    allowed_times: Optional[Set[str]] = None,
    expected_bubble_count: int = 0,
    bubble_groups: Optional[List[Dict[str, str]]] = None,
    min_similarity: float = 0.82,
) -> str:
    """Choose a text-polished candidate only when it preserves structure.

    This is a general exact-match improvement guard. It can improve absolute
    text match by allowing the vision model to polish spelling/casing/punctuation,
    but it refuses changes that alter the number of rows, Time/Side fields, visible
    time validity, bubble coverage, or message semantics too much.
    """
    if not polished_csv:
        return reference_csv

    allowed_times = allowed_times or set()
    ref_rows = _raw_side_csv_rows_for_polish(reference_csv)
    pol_rows = _raw_side_csv_rows_for_polish(polished_csv)

    if not ref_rows or len(ref_rows) != len(pol_rows):
        return reference_csv

    for ref, pol in zip(ref_rows, pol_rows):
        if ref[0] != pol[0] or ref[1] != pol[1]:
            return reference_csv
        if allowed_times and not _row_has_valid_visible_time(pol, allowed_times):
            return reference_csv

        ref_norm = _token_norm_for_text_guard(ref[2])
        pol_norm = _token_norm_for_text_guard(pol[2])
        ref_words = ref_norm.split()
        pol_words = pol_norm.split()

        # For very short rows, require near-identical token content because one
        # changed word can change the whole message.
        if min(len(ref_words), len(pol_words)) <= 3:
            if ref_norm != pol_norm and _text_guard_similarity(ref[2], pol[2]) < 0.92:
                return reference_csv
        elif _text_guard_similarity(ref[2], pol[2]) < min_similarity:
            return reference_csv

        # Avoid candidates that drastically lengthen/shorten a row.
        ref_len = max(1, len(str(ref[2]).strip()))
        pol_len = len(str(pol[2]).strip())
        if pol_len < ref_len * 0.55 or pol_len > ref_len * 1.65:
            return reference_csv

        if looks_like_noisy_ocr_text(pol[2]) and not looks_like_noisy_ocr_text(ref[2]):
            return reference_csv

    ref_candidate_score = score_side_csv_candidate(
        reference_csv,
        allowed_times=allowed_times,
        expected_bubble_count=expected_bubble_count,
        bubble_groups=bubble_groups,
    )
    pol_candidate_score = score_side_csv_candidate(
        polished_csv,
        allowed_times=allowed_times,
        expected_bubble_count=expected_bubble_count,
        bubble_groups=bubble_groups,
    )

    # Do not sacrifice row/bubble-level quality for text polish.
    if pol_candidate_score < ref_candidate_score - 0.25:
        return reference_csv

    ref_style = _text_style_score(ref_rows)
    pol_style = _text_style_score(pol_rows)

    # Prefer the polish when it is structurally safe and not stylistically worse.
    # This intentionally allows punctuation/case-only improvements that may raise
    # absolute exact match while leaving normalized/F1 metrics stable.
    if pol_style >= ref_style - 0.25:
        return polished_csv

    return reference_csv

