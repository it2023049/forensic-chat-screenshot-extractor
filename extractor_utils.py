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
        "my", "your", "our", "their", "the", "a", "an", "better", "communication"
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

def choose_best_screen_side_csv(
    draft_norm: str,
    repaired_norm: str,
    allowed_times: Set[str],
) -> str:
    """Chooses the safer draft or repair CSV for one screen."""
    # Notes:
    # The repair output must not explode in row count.
    draft_count = count_data_rows(draft_norm)
    repair_count = count_data_rows(repaired_norm)

    if repair_count == 0 and draft_count > 0:
        return draft_norm

    if draft_count == 0 and repair_count > 0:
        return repaired_norm

    # A screen cannot have wildly more rows than visible bubble times + some same-time messages.
    max_reasonable = max(len(allowed_times) + 4, draft_count + 3)

    if repair_count > max_reasonable:
        return draft_norm

    # Prefer repaired if reasonable.
    return repaired_norm
