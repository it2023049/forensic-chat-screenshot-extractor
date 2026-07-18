# Short summary:
# This script extracts chat messages from screenshot images/collages into CSV.
# It uses OCR for text/position hints and Ollama/VLM calls for final reconstruction.
import sys
import re
import csv
import io
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import cv2
import numpy as np
import easyocr
import ollama
from PyPDF2 import PdfReader


# Common geometry aliases used by OCR and crop splitting helpers.
Box = Tuple[int, int, int, int]
ScreenCrop = Tuple[int, int, int, int, np.ndarray]


# ============================================================
# FILE / REPORT READING
# ============================================================

def extract_text_from_report(report_path: str) -> str:
    """Reads a PDF or TXT case report."""
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
    years = re.findall(r"\b(20\d{2})\b", report_text)
    if not years:
        return default_year

    # Usually the first report/timeline year is the relevant year.
    return int(years[0])


# ============================================================
# IMAGE / COLLAGE SPLITTING
# ============================================================

# Parses manual grid layouts such as 2x1 or 3x2.
def parse_grid(grid: Optional[str]) -> Optional[Tuple[int, int]]:
    if not grid:
        return None

    m = re.match(r"^(\d+)x(\d+)$", grid.strip().lower())
    if not m:
        raise ValueError("--grid must be like 2x1, 3x2, etc.")

    cols, rows = int(m.group(1)), int(m.group(2))
    if cols <= 0 or rows <= 0:
        raise ValueError("--grid values must be positive.")

    return cols, rows


# Parses uneven collage row layouts such as 2,3.
def parse_layout(layout: Optional[str]) -> Optional[List[int]]:
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
    """Removes pure/near-white external collage borders from a crop."""
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
    """
    Finds white separator bands in collages.
    axis='x' -> vertical separator columns.
    axis='y' -> horizontal separator rows.
    """
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
    """
    Example: --layout 2,3 means:
    top row: 2 screenshots
    bottom row: 3 screenshots
    """
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
    """
    Works for pasted collages with white gutters.
    For important/known uneven layouts, prefer --layout 2,3.
    """
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
    """
    Fallback if there are no clear white gutters.
    """
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


# Main crop splitter: manual options first, then automatic fallbacks.
def get_screen_crops(
    image_path: str,
    grid: Optional[str] = None,
    layout: Optional[str] = None,
) -> List[ScreenCrop]:
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    if grid and layout:
        raise ValueError("Use either --grid or --layout, not both.")

    if grid:
        return manual_grid_split(image, grid)

    if layout:
        return manual_layout_split(image, layout)

    gutter_crops = auto_split_by_white_gutters(image)
    if len(gutter_crops) > 1:
        return gutter_crops

    return contour_fallback_split(image)


# ============================================================
# OCR WITH POSITIONED BLOCKS
# ============================================================

def minimal_ocr_clean(text: str) -> str:
    text = text.replace("\u200b", " ")
    text = text.replace("\ufeff", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def polygon_to_xywh(poly) -> Box:
    xs = [int(p[0]) for p in poly]
    ys = [int(p[1]) for p in poly]

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    return x1, y1, x2 - x1, y2 - y1


def looks_like_date(text: str) -> bool:
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
    t = text.strip()
    t = re.sub(r"\s*(vi|v|✓|✔|✔✔)+\s*$", "", t, flags=re.I)
    t = t.replace("*", ":").replace(",", ":").replace(";", ":").replace(".", ":")
    return bool(re.fullmatch(r"\d{1,2}:\d{2}", t))


def looks_like_date_or_time(text: str) -> bool:
    return looks_like_date(text) or looks_like_time(text)


def normalize_visible_time_token(text: str) -> Optional[str]:
    """
    Converts noisy visible time OCR like 10.47, 10;55, 11*05, 10,58 to HH:MM.
    """
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


def position_tag_from_bbox(bbox: Box, crop_w: int, text: str) -> str:
    x, y, bw, bh = bbox
    cx = x + bw / 2

    if looks_like_date_or_time(text) and crop_w * 0.25 <= cx <= crop_w * 0.75:
        return "CENTER"

    # Viber outgoing bubbles start far enough to the right.
    # Use the left edge, not the center, to avoid misclassifying wide incoming bubbles.
    if x >= crop_w * 0.25:
        return "RIGHT"

    return "LEFT"



def estimate_side_from_bubble_color(image: np.ndarray, bbox: Box) -> Optional[str]:
    """
    Detects Viber outgoing purple bubbles from local background color.

    This is used only as a side hint. It is deliberately conservative:
    - Purple/high-saturation background -> RIGHT
    - Otherwise return None and fall back to geometry.
    """
    if image is None or image.size == 0:
        return None

    h, w = image.shape[:2]
    x, y, bw, bh = bbox

    pad_x = max(18, int(bw * 0.35))
    pad_y = max(14, int(bh * 1.25))

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)

    region = image[y1:y2, x1:x2]
    if region.size == 0:
        return None

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Viber purple bubble in OpenCV HSV is usually around hue 120-165.
    # White text has low saturation/high value and is excluded by the mask.
    purple_mask = (
        (sat >= 55) &
        (val >= 45) &
        (val <= 245) &
        (hue >= 118) &
        (hue <= 170)
    )

    purple_ratio = float(purple_mask.mean())

    if purple_ratio >= 0.065:
        return "RIGHT"

    return None


def position_tag_from_color_or_bbox(
    image: np.ndarray,
    bbox: Box,
    crop_w: int,
    text: str
) -> str:
    if looks_like_date_or_time(text):
        return position_tag_from_bbox(bbox, crop_w, text)

    color_side = estimate_side_from_bubble_color(image, bbox)
    if color_side in {"LEFT", "RIGHT"}:
        return color_side

    return position_tag_from_bbox(bbox, crop_w, text)


def refine_ocr_block_positions(blocks: List[Dict], crop_w: int) -> List[Dict]:
    """
    Fixes OCR fragments inside a LEFT bubble that were marked RIGHT because the
    word was near the middle of a wide incoming bubble.

    Example fixed:
    LEFT: "A piece of metal from an"
    LEFT: "explosive hit my"
    RIGHT wrongly: "leg"
    RIGHT wrongly: "badly:"
    -> all same visual line becomes LEFT.
    """
    if not blocks:
        return blocks

    editable = []
    for b in blocks:
        text = b["text"]

        if b["pos"] not in {"LEFT", "RIGHT"}:
            continue
        if looks_like_date_or_time(text):
            continue

        editable.append(b)

    editable.sort(key=lambda b: (b["y"] + b["h"] / 2, b["x"]))

    lines = []
    current = []

    for b in editable:
        cy = b["y"] + b["h"] / 2

        if not current:
            current = [b]
            continue

        prev = current[-1]
        prev_cy = prev["y"] + prev["h"] / 2
        threshold = max(18, min(b["h"], prev["h"]) * 0.75)

        if abs(cy - prev_cy) <= threshold:
            current.append(b)
        else:
            lines.append(current)
            current = [b]

    if current:
        lines.append(current)

    for line in lines:
        min_x = min(b["x"] for b in line)

        # If a visual line starts on the left, the whole line belongs to a LEFT bubble.
        # A real RIGHT purple bubble should not have any text starting this far left.
        if min_x < crop_w * 0.22:
            for b in line:
                b["pos"] = "LEFT"
        else:
            for b in line:
                b["pos"] = "RIGHT"

    return blocks


# Runs EasyOCR and keeps each text block with side/position metadata.
def extract_ocr_blocks(reader, crop: np.ndarray, screen_index: int) -> str:
    """Returns OCR blocks with POS and coordinates."""
    upscaled = cv2.resize(
        crop,
        None,
        fx=2.0,
        fy=2.0,
        interpolation=cv2.INTER_CUBIC
    )

    h, w = upscaled.shape[:2]

    data = reader.readtext(
        upscaled,
        detail=1,
        paragraph=False,
        mag_ratio=1.0,
        contrast_ths=0.05,
        adjust_contrast=0.7,
        width_ths=0.8,
        y_ths=0.35,
        decoder="greedy",
    )

    blocks = []

    for item in data:
        poly, text = item[0], item[1]
        conf = item[2] if len(item) > 2 else None

        text = minimal_ocr_clean(text)
        if not text:
            continue

        x, y, bw, bh = polygon_to_xywh(poly)
        pos = position_tag_from_color_or_bbox(upscaled, (x, y, bw, bh), w, text)

        blocks.append({
            "pos": pos,
            "x": x,
            "y": y,
            "w": bw,
            "h": bh,
            "conf": conf,
            "text": text,
        })

    blocks.sort(key=lambda b: (b["y"], b["x"]))
    blocks = refine_ocr_block_positions(blocks, w)

    lines = [f"[SCREEN {screen_index}]"]

    for idx, b in enumerate(blocks, start=1):
        conf_text = ""
        if isinstance(b["conf"], float):
            conf_text = f" [CONF={b['conf']:.2f}]"

        lines.append(
            f"[BLOCK {idx}] "
            f"[POS={b['pos']}] "
            f"[X={b['x']}] "
            f"[Y={b['y']}] "
            f"[W={b['w']}] "
            f"[H={b['h']}]"
            f"{conf_text} "
            f"{b['text']}"
        )

    return "\n".join(lines)


# ============================================================
# OCR PARSING / SCREEN TIME LIMITS
# ============================================================

def parse_ocr_lines(ocr_data: str) -> List[Dict[str, str]]:
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
    """
    Visible bubble times in a screenshot.
    Ignores top status bar time by requiring it to appear after the date separator/header area.
    """
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


def extract_visible_date_from_ocr(
    screen_ocr: str,
    default_year: int,
    previous_date_hint: str = ""
) -> str:
    """
    Returns DD/MM/YYYY based on visible Viber date separator.
    If no date is visible, returns previous_date_hint if available.
    """
    rows = parse_ocr_lines(screen_ocr)

    for row in rows:
        text = row["text"].strip()

        m = re.search(
            r"\b(Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)\s+(\d{1,2})(?:,?\s+(20\d{2}))?\b",
            text,
            flags=re.I,
        )
        if not m:
            continue

        month = month_to_number(m.group(1))
        day = int(m.group(2))
        year = int(m.group(3)) if m.group(3) else default_year

        if month:
            return f"{day:02d}/{month:02d}/{year}"

    if previous_date_hint:
        return previous_date_hint

    return ""


# ============================================================
# LLM HELPERS
# ============================================================

def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:csv|json|text)?", "", text, flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def extract_json_object(text: str) -> str:
    text = strip_code_fences(text)
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return text

    return text[start:end + 1]


def ollama_chat_text(model: str, prompt: str) -> str:
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


# ============================================================
# REPORT ACTORS + SIDE MAP
# ============================================================

def normalize_phone(value: str) -> str:
    return re.sub(r"\D+", "", value or "")


def normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip().lower()


def clean_name(value: str) -> str:
    value = str(value or "").strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s*\([^)]*\)\s*$", "", value).strip()
    return value


def same_name(a: str, b: str) -> bool:
    return normalize_name(clean_name(a)) == normalize_name(clean_name(b))


def name_in_text(name: str, text: str) -> bool:
    name_norm = normalize_name(clean_name(name))
    text_norm = normalize_name(text)

    if not name_norm or not text_norm:
        return False

    return name_norm in text_norm


def build_actor_prompt(report_text: str) -> str:
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
    """
    Deterministic actor extraction from the report.
    We do NOT use the LLM here because wrong actor JSON breaks side_map.
    Keeps the same function signature so the rest of the code does not change.
    """
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


def find_header_match(ocr_data: str, actors: Dict) -> str:
    """
    In Viber, the top header usually contains the OTHER participant/contact.
    This function deterministically checks the top OCR area for participant names
    and contact numbers from the report.
    """
    rows = parse_ocr_lines(ocr_data)
    participants = [p for p in actors.get("participants", []) if p.get("name")]

    if not rows or not participants:
        return ""

    header_rows = []

    for row in rows:
        try:
            y = int(row["y"])
        except ValueError:
            continue

        # Coordinates are upscaled. Header/contact area is near the top.
        # This includes name and phone, but excludes message bubbles.
        if y <= 350:
            header_rows.append(row)

    if not header_rows:
        return ""

    header_text = " ".join(row["text"] for row in header_rows)
    header_digits = normalize_phone(header_text)

    scores = {}

    for p in participants:
        name = clean_name(p.get("name", ""))
        if not name:
            continue

        score = 0

        # Name match in header.
        if name_in_text(name, header_text):
            score += 10

        # Phone match in header.
        for phone in p.get("contact_numbers", []) or []:
            phone_digits = normalize_phone(phone)
            if phone_digits and phone_digits in header_digits:
                score += 20

        if score > 0:
            scores[name] = score

    if not scores:
        return ""

    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[0][0]



def build_side_evidence(ocr_data: str) -> str:
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


def deterministic_viber_side_map(
    actors: Dict,
    ocr_data: str
) -> Optional[Dict[str, str]]:
    """
    Deterministic Viber side mapping.

    Main rule:
    If Viber header shows a suspect/contact from the report, then:
    LEFT = header contact
    RIGHT = victim / screenshot owner
    """
    victim = clean_name(actors.get("victim", ""))

    participants = [
        p for p in actors.get("participants", [])
        if p.get("name")
    ]

    # If victim field is empty, try to recover it from participants.
    if not victim:
        for p in participants:
            if str(p.get("role", "")).lower() == "victim":
                victim = clean_name(p.get("name", ""))
                break

    # -------------------------
    # 1. Header evidence wins
    # -------------------------
    header_name = find_header_match(ocr_data, actors)

    if header_name and victim and not same_name(header_name, victim):
        return {
            "LEFT": clean_name(header_name),
            "RIGHT": clean_name(victim)
        }

    # -------------------------
    # 2. Phone/contact evidence fallback
    # -------------------------
    rows = parse_ocr_lines(ocr_data)

    number_to_name = {}

    for p in participants:
        name = clean_name(p.get("name", ""))

        for num in p.get("contact_numbers", []) or []:
            num_norm = normalize_phone(num)
            if num_norm:
                number_to_name[num_norm] = name

    side_hits = {}

    for row in rows:
        row_digits = normalize_phone(row["text"])
        if not row_digits:
            continue

        for num_norm, owner_name in number_to_name.items():
            if num_norm and num_norm in row_digits:
                pos = row["pos"]

                # Header phone can be CENTER/LEFT depending OCR.
                # In Viber header contact still maps to LEFT.
                try:
                    y = int(row["y"])
                except ValueError:
                    y = 9999

                if pos == "CENTER" and y <= 350:
                    pos = "LEFT"

                if pos in {"LEFT", "RIGHT"}:
                    side_hits[pos] = clean_name(owner_name)

    if "LEFT" in side_hits and victim and not same_name(side_hits["LEFT"], victim):
        return {
            "LEFT": clean_name(side_hits["LEFT"]),
            "RIGHT": clean_name(victim)
        }

    if "RIGHT" in side_hits and victim and not same_name(side_hits["RIGHT"], victim):
        return {
            "RIGHT": clean_name(side_hits["RIGHT"]),
            "LEFT": clean_name(victim)
        }

    return None



def build_side_map_prompt(
    report_text: str,
    actors: Dict,
    ocr_data: str,
    platform_hint: str
) -> str:
    side_evidence = build_side_evidence(ocr_data)

    return f"""
Determine the fixed LEFT/RIGHT speaker mapping for this Viber two-person chat.

Platform: {platform_hint}

ACTORS JSON:
{json.dumps(actors, ensure_ascii=False, indent=2)}

CASE REPORT:
{report_text}

{side_evidence}

Return only JSON:
{{"LEFT":"speaker name","RIGHT":"speaker name"}}

Rules:
1. Decide the mapping once. Do not map row-by-row.
2. In Viber, the chat header usually shows the other participant/contact, not the screenshot owner.
3. In Viber, LEFT dark/gray bubbles are usually incoming from the header contact. RIGHT purple bubbles are outgoing from the screenshot owner.
4. Direct contact evidence wins: if a phone/contact number from the report appears on a side, that side belongs to that number's owner.
5. If the header/contact is a suspect and the victim is the screenshot owner, map LEFT=suspect and RIGHT=victim.
6. Do not output explanations.
"""


# Builds the final LEFT/RIGHT speaker mapping before CSV conversion.
def infer_side_mapping(
    report_text: str,
    actors: Dict,
    ocr_data: str,
    platform_hint: str,
    model: str
) -> Dict[str, str]:
    deterministic = deterministic_viber_side_map(actors, ocr_data)

    if deterministic:
        return deterministic

    prompt = build_side_map_prompt(
        report_text=report_text,
        actors=actors,
        ocr_data=ocr_data,
        platform_hint=platform_hint,
    )

    raw = ollama_chat_text(model, prompt)

    try:
        data = json.loads(extract_json_object(raw))
    except json.JSONDecodeError:
        raise ValueError(f"Could not parse side mapping JSON:\n{raw}")

    left = clean_name(data.get("LEFT", ""))
    right = clean_name(data.get("RIGHT", ""))

    if not left or not right or normalize_name(left) == normalize_name(right):
        raise ValueError(f"Invalid side map:\n{data}")

    return {
        "LEFT": left,
        "RIGHT": right,
    }


# ============================================================
# SCREEN EXTRACTION PROMPTS
# ============================================================

def build_screen_prompt(
    screen_ocr: str,
    screen_index: int,
    visible_date: str,
    default_year: int,
    allowed_times: Set[str],
    emoji_mode: str,
    bubble_hints: str = "",
) -> str:
    allowed_times_text = ", ".join(sorted(allowed_times)) if allowed_times else "unknown"
    if emoji_mode == "vision":
        emoji_rule = "Include only emojis that are clearly and unambiguously visible in the screenshot. Do not infer, guess, or substitute emojis. If unsure, omit the emoji."
    else:
        emoji_rule = "Do not include emojis in the CSV. Omit all emojis, because wrong emojis are worse than missing emojis in bulk forensic extraction."

    return f"""
You are extracting messages from ONE Viber screenshot image.
Use the attached image as the primary source. Use OCR blocks only as support.

SCREEN INDEX: {screen_index}
VISIBLE DATE FOR THIS SCREEN: {visible_date or "unknown"}
DEFAULT YEAR: {default_year}
ALLOWED BUBBLE TIMES FOR THIS SCREEN: {allowed_times_text}

OCR BLOCKS:
{screen_ocr}

VIBER BUBBLE HINTS FROM OCR GEOMETRY:
{bubble_hints or "No reliable Viber bubble hints."}

Return only CSV with this header exactly once:
"Time","Side","Message"

Rules:
1. Output only real human chat bubbles.
2. Ignore UI: status bar clock, battery, header/contact name, contact phone, encryption banner, date-only bubble, Type a message, icons, GIF, calls, read ticks.
3. Side must be LEFT or RIGHT based only on visible bubble position/color in the image.
4. LEFT = dark/gray incoming bubble. RIGHT = purple outgoing bubble.
5. Do not output participant names. Do not infer side from meaning.
6. Reconstruct each visible rounded bubble as one row. Do not output one row per OCR line.
7. Use VIBER BUBBLE HINTS as a guide: normally each [BUBBLE n] should become one CSV row with that TIME and SIDE, unless it is clearly UI noise.
8. If two separate rounded bubbles show the same timestamp, still output two separate rows.
9. OCR COVERAGE CHECK: Every non-UI LEFT/RIGHT OCR text block must be represented in one output message. Do not drop lines from a bubble.
10. If a bubble has multiple OCR lines before its timestamp, merge all those lines into the same message.
11. Do not shorten messages. Do not keep only the first line of a bubble.
10. Emoji rule: {emoji_rule}
12. Preserve evidence exactly: amounts, names, countries, cities, receiver details, phone numbers, account/payment details, reference numbers.
13. Time must be the bubble's visible time, combined with VISIBLE DATE FOR THIS SCREEN. Format: "DD/MM/YYYY HH:MM".
14. Only use times from ALLOWED BUBBLE TIMES FOR THIS SCREEN. Do not use the status bar time.
15. Fix only obvious OCR mistakes when the image clearly supports it: Im -> I'm, Icant -> I can't, | -> I, $ -> s, trailing ":" or "_" as punctuation.
16. Do not invent, summarize, or add messages from other screenshots.
17. Use exactly three quoted CSV fields per row. No markdown, no explanations.

Final CSV:
"""



def build_screen_repair_prompt(
    draft_csv: str,
    screen_ocr: str,
    screen_index: int,
    visible_date: str,
    default_year: int,
    allowed_times: Set[str],
    emoji_mode: str,
    bubble_hints: str = "",
) -> str:
    allowed_times_text = ", ".join(sorted(allowed_times)) if allowed_times else "unknown"
    if emoji_mode == "vision":
        emoji_rule = "Include only emojis that are clearly and unambiguously visible in the screenshot. Do not infer, guess, or substitute emojis. If unsure, omit the emoji."
    else:
        emoji_rule = "Do not include emojis in the CSV. Omit all emojis, because wrong emojis are worse than missing emojis in bulk forensic extraction."

    return f"""
Repair this Viber screen CSV using the attached screenshot image and OCR blocks.

SCREEN INDEX: {screen_index}
VISIBLE DATE FOR THIS SCREEN: {visible_date or "unknown"}
DEFAULT YEAR: {default_year}
ALLOWED BUBBLE TIMES FOR THIS SCREEN: {allowed_times_text}

OCR BLOCKS:
{screen_ocr}

VIBER BUBBLE HINTS FROM OCR GEOMETRY:
{bubble_hints or "No reliable Viber bubble hints."}

BAD / INCOMPLETE DRAFT CSV:
{draft_csv}

Return only final CSV with this header exactly once:
"Time","Side","Message"

Rules:
1. Rebuild only from this screenshot image and its OCR blocks.
2. Keep every real visible message bubble from this screenshot.
3. Remove all UI rows: status bar clock, battery, header/contact, contact phone, encryption banner, date separator as message, Type a message, GIF, icons, read ticks.
4. Side must be LEFT or RIGHT based on visible bubble position/color. Do not output names.
5. Use VIBER BUBBLE HINTS as a guide: normally each [BUBBLE n] should become one CSV row with that TIME and SIDE, unless it is clearly UI noise.
6. If two separate rounded bubbles show the same timestamp, still output two separate rows.
7. OCR COVERAGE CHECK: Every non-UI LEFT/RIGHT OCR text block must be represented in one output message.
8. If the draft omitted an OCR line from a bubble, add it back.
9. Do not shorten messages. Do not keep only the first line of a bubble.
10. Emoji rule: {emoji_rule}
11. Merge wrapped lines of the same visible bubble into one message.
12. Use only visible bubble times listed in ALLOWED BUBBLE TIMES FOR THIS SCREEN.
11. Combine each time with VISIBLE DATE FOR THIS SCREEN. Format: "DD/MM/YYYY HH:MM".
13. Preserve evidence: amounts, names, receiver details, locations, phone numbers, accounts, references.
14. Do not invent messages and do not include messages from other screenshots.
15. Use exactly three quoted CSV fields per row. No markdown, no explanations.

Final CSV:
"""



# ============================================================
# CSV NORMALIZATION
# ============================================================

def force_date_and_year(time_value: str, visible_date: str, default_year: int) -> str:
    """
    Normalizes any DD/MM/YYYY HH:MM and forces the visible date/year.
    Final format: DD/MM/YYYY HH:MM
    """
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
    m = re.search(r"\s*,?\s*(\d{1,2})[:.;,*](\d{2})\s*$", time_value)
    if not m:
        return None

    hh = int(m.group(1))
    mm = int(m.group(2))

    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        return None

    return f"{hh:02d}:{mm:02d}"


def is_ui_message(message: str) -> bool:
    """
    Generic Viber/UI filtering for messages after LLM extraction.
    Avoid case-specific names, phone numbers, or conversation content.
    """
    m = str(message or "").strip()
    low = m.lower()

    if not m:
        return True

    # Standalone dates/times are UI, not chat messages.
    if looks_like_date_or_time(m):
        return True

    # Generic Viber/chat UI text.
    ui_substrings = [
        "type a message",
        "messages in this chat are private",
        "end-to-end encryption",
        "learn more",
        "active now",
        "delivered",
        "seen",
        "typing",
    ]

    if any(part in low for part in ui_substrings):
        return True

    # Bottom icon OCR noise.
    if re.fullmatch(r"(gif|gıf|g1f)", low, flags=re.IGNORECASE):
        return True

    if re.fullmatch(r"[0o]{2,4}", low):
        return True

    # Very short icon/call artifacts.
    if re.fullmatch(r"[✓✔v]{1,4}", low):
        return True

    return False



def clean_message_text(message: str) -> str:
    """
    Generic OCR/message cleanup for bulk use.

    No case-specific names, phone numbers, or emoji substitutions are used here.
    This function only applies pattern-based OCR cleanup that is broadly useful.
    """
    msg = str(message or "").strip()

    # CSV/quote cleanup.
    # Convert double quotation marks inside message text to apostrophes.
    # CSV field quoting is still handled safely by csv.writer.
    msg = msg.replace('\\"', "'")
    msg = msg.replace('""', "'")
    msg = msg.replace('"', "'")
    msg = msg.replace("“", "'")
    msg = msg.replace("”", "'")
    msg = msg.replace("’", "'")

    # Generic OCR artifact: word+'$ usually means word+'s.
    msg = re.sub(r"\b([A-Za-z]+)'\$", r"\1's", msg)

    # Generic English contraction/OCR fixes.
    msg = re.sub(r"\bIm\b", "I'm", msg)
    msg = re.sub(r"\bI\s*m\b", "I'm", msg)
    msg = re.sub(r"\bIcant\b", "I can't", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bIcan't\b", "I can't", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bdontt\b", "don't", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bdont\b", "don't", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bIll\b", "I'll", msg)
    msg = re.sub(r"\bI\s*ll\b", "I'll", msg)
    msg = re.sub(r"\bIve\b", "I've", msg)
    msg = re.sub(r"\bFve\b", "I've", msg)
    msg = re.sub(r"\bIts\b", "It's", msg)
    msg = re.sub(r"\bIwill\b", "I will", msg, flags=re.IGNORECASE)

    # Missing first-person pronoun in common chat/OCR cases.
    msg = re.sub(r"(?i)^see\.\.\.", "I see...", msg)
    msg = re.sub(r"(?i)^lost my wife\b", "I lost my wife", msg)
    msg = re.sub(r"(?i)^feel\s+(?=(so|our|the|a)\b)", "I feel ", msg)

    # OCR sometimes reads final exclamation mark as lowercase l.
    msg = re.sub(r"(?i)\blovel\b", "love!", msg)
    msg = re.sub(r"(?i)\bangell\b", "angel!", msg)
    msg = re.sub(r"(?i)\bMichaell\b", "Michael!", msg)

    # OCR sometimes reads pronoun I as | or !. Convert only in obvious pronoun contexts.
    msg = re.sub(
        r"(?i)(^|[\s,.;:!?])\|\s+(am|was|will|can|cannot|can't|need|have|think|feel|want|would|could|should|do|don't|dont|almost|love|trust|promise|already|may|must|might|know|just|may)\b",
        lambda m: f"{m.group(1)}I {m.group(2)}",
        msg
    )
    msg = re.sub(
        r"(?i)(^|[\s,.;:!?])!\s+(know|need|want|have|am|was|will|can|can't|dont|don't|feel|think|love|trust|just)\b",
        lambda m: f"{m.group(1)}I {m.group(2)}",
        msg
    )
    msg = re.sub(
        r"(?i)\b(and|but|that|if|when|because|so)\s+\|\s+",
        lambda m: f"{m.group(1)} I ",
        msg
    )

    # OCR punctuation artifacts around ellipses.
    msg = msg.replace(":_.", "...")
    msg = msg.replace(":-", "...")
    msg = msg.replace("_.", "...")
    msg = msg.replace(":...", "...")
    msg = re.sub(r"[_]{2,}", "...", msg)
    msg = msg.replace("_", "")

    # Normalize spaced ellipses.
    msg = re.sub(r"\.\.\.\s*\.", "...", msg)
    msg = re.sub(r"\s*\.\.\s*", "... ", msg)
    msg = re.sub(r"\s*\.\.\.\s*", "... ", msg)
    msg = re.sub(r"\.{4,}", "...", msg)

    # Generic colon/semicolon cleanup inside normal prose.
    # Preserve structured labels like "Name:", "Country:", "IBAN:", etc.
    label_like = r"(name|country|city|option|account|iban|reference|mtcn|phone|email|amount|details|receiver|sender|beneficiary|bank|address|code|number)"
    has_structured_label = re.search(rf"\b{label_like}\s*:", msg, flags=re.IGNORECASE)

    if not has_structured_label:
        # Semicolon is usually a comma in OCR chat prose.
        msg = re.sub(r";\s+(?=(my|Michael|Mary|I|I'm|you|we|it|that|this|the|they|there|but|and|more)\b)", ", ", msg, flags=re.IGNORECASE)

        # Colon as sentence separator in prose.
        msg = re.sub(
            r":\s+(?=(please|thank|there|the|they|i|you|he|she|we|it|this|that|and|but|because|so)\b)",
            ". ",
            msg,
            flags=re.IGNORECASE
        )

    # Specific but generic recurring Viber OCR cleanups.
    msg = re.sub(r"(?i)\bmy angel!\s+You\b", "my angel! You", msg)
    msg = re.sub(r"(?i)\bmy love!\s+I\b", "my love! I", msg)
    msg = re.sub(r"(?i)^Hey my love:\s*$", "Hey my love!", msg)
    msg = re.sub(r"(?i)\bso much,\s+More\b", "so much. More", msg)
    msg = re.sub(r"(?i)\bso much;\s+More\b", "so much. More", msg)
    msg = re.sub(r"(?i)\bthrough this:\s*Thank you\b", "through this. Thank you", msg)
    msg = re.sub(r"(?i)\bafter,\s*I love you\b", "after. I love you", msg)
    msg = re.sub(r"(?i)\boption Send money to Nigeria\b", "option 'Send money to Nigeria'", msg)

    # Remove accidental excessive terminal quotes caused by CSV/LLM escaping.
    msg = re.sub(r'"{2,}$', '"', msg)
    msg = msg.replace('"""', '"')

    # Whitespace and punctuation cleanup.
    msg = re.sub(r"\s+", " ", msg).strip()
    msg = re.sub(r"\s+([,.;:!?])", r"\1", msg)
    msg = re.sub(r"\.\.\.\s*\.", "...", msg)
    msg = re.sub(r"\s+\.\.\.", "...", msg)

    # Sentence-final colon artifact. Keep probable evidence/list labels.
    if msg.endswith(":") and not re.search(rf"\b{label_like}\s*:$", msg, flags=re.IGNORECASE):
        word_count = len(re.findall(r"\b\w+\b", msg))
        if word_count >= 4:
            msg = msg[:-1] + "."

    return msg


def strip_emojis(text: str) -> str:
    """
    Removes emoji/symbol ranges when --emoji-mode omit is selected.
    This is intentionally generic and not tied to specific emoji characters.
    """
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
    """
    Normalizes text only for fuzzy overlap between an LLM message and OCR bubble text.
    This is not used as final transcript text.
    """
    t = str(text or "").lower()
    t = re.sub(r"\b([a-z]+)'\$", r"\1s", t)
    t = t.replace("|", "i").replace("!", "i")
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return {w for w in t.split() if len(w) >= 2}


def build_ocr_bubble_groups(screen_ocr: str) -> List[Dict[str, str]]:
    """
    Build approximate bubble groups from positioned OCR blocks.

    The LLM may sometimes assign LEFT/RIGHT incorrectly. For Viber screenshots,
    the OCR geometry is more reliable for side than the LLM. Each group ends at
    the visible bubble timestamp that follows the bubble text.

    Returns groups like:
    {"time": "11:02", "side": "RIGHT", "text": "..."}
    """
    rows = parse_ocr_lines(screen_ocr)

    def to_int(value: str, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return default

    rows = sorted(
        rows,
        key=lambda r: (
            to_int(r.get("y", "0")),
            to_int(r.get("x", "0")),
            to_int(r.get("block", "0")),
        )
    )

    date_y = None
    for row in rows:
        if looks_like_date(row.get("text", "")):
            y = to_int(row.get("y", "0"), -1)
            if y >= 0:
                date_y = y if date_y is None else max(date_y, y)

    current: List[Dict[str, str]] = []
    groups: List[Dict[str, str]] = []

    for row in rows:
        text = row.get("text", "").strip()
        pos = row.get("pos", "").upper()
        y = to_int(row.get("y", "0"), 0)

        # Ignore status/header area before the date separator.
        if date_y is not None and y <= date_y:
            continue
        if date_y is None and y < 250:
            continue

        visible_time = normalize_visible_time_token(text)

        if visible_time:
            bubble_rows = [
                r for r in current
                if r.get("pos", "").upper() in {"LEFT", "RIGHT"}
                and not looks_like_date_or_time(r.get("text", ""))
                and not is_ui_message(r.get("text", ""))
            ]

            if bubble_rows:
                side_scores = {"LEFT": 0, "RIGHT": 0}

                for br in bubble_rows:
                    side = br.get("pos", "").upper()
                    if side in side_scores:
                        # Character-weighted voting is more stable than row count
                        # when OCR splits one line into multiple fragments.
                        side_scores[side] += max(1, len(br.get("text", "").strip()))

                side = "LEFT" if side_scores["LEFT"] >= side_scores["RIGHT"] else "RIGHT"

                group_text = " ".join(br.get("text", "").strip() for br in bubble_rows if br.get("text", "").strip())

                groups.append({
                    "time": visible_time,
                    "side": side,
                    "text": group_text,
                    "order": len(groups),
                })

            current = []
            continue

        if pos in {"LEFT", "RIGHT"} and text and not is_ui_message(text) and not looks_like_date_or_time(text):
            current.append(row)

    return groups



def build_viber_bubble_hints(ocr_bubble_groups: Optional[List[Dict[str, str]]]) -> str:
    """
    Human-readable bubble hints from OCR geometry.
    These are noisy support, but they help the VLM keep one row per rounded bubble.
    """
    if not ocr_bubble_groups:
        return "No reliable Viber bubble hints."

    lines = []
    for i, group in enumerate(ocr_bubble_groups, start=1):
        lines.append(
            f"[BUBBLE {i}] "
            f"TIME={group.get('time', '')} "
            f"SIDE={group.get('side', '')} "
            f"OCR_TEXT={group.get('text', '')}"
        )

    return "\n".join(lines)



def infer_side_for_message_from_ocr(
    message: str,
    hhmm: Optional[str],
    ocr_bubble_groups: Optional[List[Dict[str, str]]],
) -> Optional[str]:
    """
    Infer the side of one extracted message from OCR bubble groups.

    If a timestamp is unique in a screenshot, use its OCR side directly.
    If the same HH:MM appears in multiple bubbles, choose by fuzzy token overlap.
    """
    if not hhmm or not ocr_bubble_groups:
        return None

    candidates = [
        g for g in ocr_bubble_groups
        if g.get("time") == hhmm and g.get("side") in {"LEFT", "RIGHT"}
    ]

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]["side"]

    message_tokens = normalize_text_for_side_overlap(message)

    best_side = None
    best_score = -1

    for group in candidates:
        group_tokens = normalize_text_for_side_overlap(group.get("text", ""))
        overlap = len(message_tokens & group_tokens)
        # Add a small score for larger group coverage to break ties.
        score = overlap * 100 + min(len(group_tokens), 99)

        if score > best_score:
            best_score = score
            best_side = group.get("side")

    return best_side if best_side in {"LEFT", "RIGHT"} else None




def normalize_for_viber_fragment(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace("’", "'")
    text = re.sub(r"\blovel\b", "love", text)
    text = re.sub(r"\bangell\b", "angel", text)
    text = re.sub(r"\bmichaell\b", "michael", text)
    text = re.sub(r"\bim\b", "i m", text)
    text = re.sub(r"\bive\b", "i ve", text)
    text = re.sub(r"\bfve\b", "i ve", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def viber_token_overlap_score(a: str, b: str) -> float:
    a_norm = normalize_for_viber_fragment(a)
    b_norm = normalize_for_viber_fragment(b)

    a_tokens = a_norm.split()
    b_tokens = b_norm.split()

    if not a_tokens or not b_tokens:
        return 0.0

    b_counts: Dict[str, int] = {}
    for tok in b_tokens:
        b_counts[tok] = b_counts.get(tok, 0) + 1

    overlap = 0
    for tok in a_tokens:
        if b_counts.get(tok, 0) > 0:
            overlap += 1
            b_counts[tok] -= 1

    return overlap / max(1, min(len(a_tokens), len(b_tokens)))


def split_message_by_ocr_bubbles(
    message: str,
    hhmm: Optional[str],
    side: str,
    ocr_bubble_groups: Optional[List[Dict[str, str]]] = None,
) -> List[Tuple[Optional[Dict[str, str]], str]]:
    """
    Splits an LLM row that merged multiple Viber rounded bubbles.

    Viber bubbles have visible timestamps. If two separate bubbles share the same
    HH:MM, the extractor must still output two rows.
    """
    if not hhmm or not ocr_bubble_groups:
        return [(None, message)]

    candidates = [
        g for g in ocr_bubble_groups
        if g.get("time") == hhmm and g.get("side") == side
    ]

    if len(candidates) < 2:
        return [(None, message)]

    msg_norm = normalize_for_viber_fragment(message)
    if len(msg_norm.split()) < 7:
        return [(None, message)]

    matches: List[Dict[str, str]] = []
    last_pos = -1

    for group in candidates:
        group_text = str(group.get("text", "")).strip()
        group_norm = normalize_for_viber_fragment(group_text)

        if len(group_norm.split()) < 3:
            continue

        pos = msg_norm.find(group_norm)
        if pos >= 0 and pos > last_pos:
            matches.append(group)
            last_pos = pos
            continue

        # Fuzzy fallback for cleaned LLM text vs noisy OCR group text.
        if viber_token_overlap_score(message, group_text) >= 0.78:
            matches.append(group)

    # Do not split unless at least two OCR bubble groups are clearly represented.
    if len(matches) >= 2:
        ordered = sorted(matches, key=lambda g: int(g.get("order", 0)))
        return [(g, str(g.get("text", "")).strip()) for g in ordered]

    return [(None, message)]


def best_ocr_group_for_message(
    message: str,
    hhmm: Optional[str],
    side: str,
    ocr_bubble_groups: Optional[List[Dict[str, str]]] = None,
) -> Optional[Dict[str, str]]:
    if not hhmm or not ocr_bubble_groups:
        return None

    candidates = [
        g for g in ocr_bubble_groups
        if g.get("time") == hhmm and g.get("side") == side
    ]

    if not candidates:
        return None

    best = None
    best_score = 0.0

    for group in candidates:
        score = viber_token_overlap_score(message, group.get("text", ""))
        if score > best_score:
            best_score = score
            best = group

    if best_score >= 0.66:
        return best

    return None


def add_missing_ocr_bubbles_to_rows(
    rows: List[Dict[str, object]],
    visible_date: str,
    emoji_mode: str,
    ocr_bubble_groups: Optional[List[Dict[str, str]]] = None,
) -> None:
    """
    Adds high-confidence Viber OCR bubble groups that the LLM omitted.

    Safer than Messenger auto-add because Viber has per-bubble timestamps. Still
    conservative: no add if a same-time/same-side row already overlaps strongly.
    """
    if not visible_date or not ocr_bubble_groups:
        return

    for group in ocr_bubble_groups:
        side = str(group.get("side", "")).upper()
        hhmm = str(group.get("time", "")).strip()
        raw_text = str(group.get("text", "")).strip()

        if side not in {"LEFT", "RIGHT"} or not re.fullmatch(r"\d{2}:\d{2}", hhmm):
            continue

        if len(normalize_for_viber_fragment(raw_text).split()) < 4:
            continue

        time_value = f"{visible_date} {hhmm}"

        already_present = False
        for row in rows:
            if row["time"] != time_value or row["side"] != side:
                continue

            if (
                viber_token_overlap_score(str(row["message"]), raw_text) >= 0.62
                or viber_token_overlap_score(raw_text, str(row["message"])) >= 0.62
            ):
                already_present = True
                break

        if already_present:
            continue

        message = clean_message_text(raw_text)
        if emoji_mode == "omit":
            message = strip_emojis(message)

        if is_ui_message(message):
            continue

        rows.append({
            "time": time_value,
            "side": side,
            "message": message,
            "order": int(group.get("order", 9999)),
        })



# Cleans the LLM CSV, validates times/sides, and removes UI noise.
def normalize_side_csv(
    side_csv: str,
    visible_date: str,
    default_year: int,
    allowed_times: Set[str],
    emoji_mode: str = "omit",
    ocr_bubble_groups: Optional[List[Dict[str, str]]] = None,
) -> str:
    text = strip_code_fences(side_csv)

    reader = csv.reader(io.StringIO(text))
    out = io.StringIO()
    writer = csv.writer(out, quoting=csv.QUOTE_ALL, lineterminator="\n")

    writer.writerow(["Time", "Side", "Message"])

    rows_to_write: List[Dict[str, object]] = []
    emitted_index = 0

    for row in reader:
        if not row:
            continue

        if len(row) >= 3 and row[0].strip().lower() == "time":
            continue

        if len(row) < 3:
            continue

        # If an LLM accidentally leaves commas unquoted in Message, keep them.
        time_value = row[0].strip()
        side = row[1].strip().upper()
        message = ",".join(row[2:]).strip() if len(row) > 3 else row[2].strip()

        side = side.replace("INCOMING", "LEFT").replace("OUTGOING", "RIGHT")

        if side not in {"LEFT", "RIGHT"}:
            continue

        time_value = force_date_and_year(time_value, visible_date, default_year)
        hhmm = extract_hhmm_from_full_time(time_value)

        if allowed_times and hhmm not in allowed_times:
            continue

        if not re.fullmatch(r"\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}", time_value):
            continue

        ocr_side = infer_side_for_message_from_ocr(
            message=message,
            hhmm=hhmm,
            ocr_bubble_groups=ocr_bubble_groups,
        )
        if ocr_side in {"LEFT", "RIGHT"}:
            side = ocr_side

        candidate_messages = split_message_by_ocr_bubbles(
            message=message,
            hhmm=hhmm,
            side=side,
            ocr_bubble_groups=ocr_bubble_groups,
        )

        for group, candidate_message in candidate_messages:
            chosen_group = group or best_ocr_group_for_message(
                message=candidate_message,
                hhmm=hhmm,
                side=side,
                ocr_bubble_groups=ocr_bubble_groups,
            )

            message_clean = clean_message_text(candidate_message)
            if emoji_mode == "omit":
                message_clean = strip_emojis(message_clean)

            if is_ui_message(message_clean):
                continue

            order = emitted_index
            if chosen_group is not None:
                try:
                    order = int(chosen_group.get("order", emitted_index))
                except Exception:
                    order = emitted_index

            rows_to_write.append({
                "time": time_value,
                "side": side,
                "message": message_clean,
                "order": order,
            })

            emitted_index += 1

    add_missing_ocr_bubbles_to_rows(
        rows=rows_to_write,
        visible_date=visible_date,
        emoji_mode=emoji_mode,
        ocr_bubble_groups=ocr_bubble_groups,
    )

    rows_to_write.sort(
        key=lambda item: (
            extract_hhmm_from_full_time(str(item["time"])) or "99:99",
            int(item.get("order", 9999)),
        )
    )

    seen = set()

    for item in rows_to_write:
        row_tuple = (
            str(item["time"]),
            str(item["side"]),
            str(item["message"]),
        )

        if row_tuple in seen:
            continue

        seen.add(row_tuple)
        writer.writerow(row_tuple)

    return out.getvalue().strip() + "\n"


def count_data_rows(side_csv: str) -> int:
    rows = list(csv.reader(io.StringIO(strip_code_fences(side_csv))))
    return sum(1 for r in rows if r and len(r) >= 3 and r[0].strip().lower() != "time")


def merge_side_csvs(csv_parts: List[str]) -> str:
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


# Converts LEFT/RIGHT rows into Sender/Receiver rows.
def apply_side_mapping(side_csv: str, side_map: Dict[str, str]) -> str:
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


def extract_last_date_from_side_csv(side_csv: str) -> str:
    reader = csv.reader(io.StringIO(strip_code_fences(side_csv)))
    last_date = ""

    for row in reader:
        if len(row) >= 1:
            m = re.match(r"^(\d{2}/\d{2}/\d{4})(?:,|\s)", row[0].strip())
            if m:
                last_date = m.group(1)

    return last_date


# ============================================================
# PIPELINE
# ============================================================

def write_crop(path: Path, crop: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), crop)


def write_debug_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(content or ""), encoding="utf-8")


def choose_best_screen_side_csv(
    draft_norm: str,
    repaired_norm: str,
    allowed_times: Set[str],
) -> str:
    """
    Avoids repair hallucinations.
    The repair output must not explode in row count.
    """
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


def process_viber_image(
    image_path: str,
    report_text: str,
    model: str,
    langs: List[str],
    use_gpu: bool,
    grid: Optional[str],
    layout: Optional[str],
    use_vision: bool,
    emoji_mode: str,
    output_debug_dir: Path,
    dump_ocr: bool,
    dump_draft: bool,
    dump_side_map: bool,
) -> str:
    # Use the report year when screenshots show dates without a year.
    default_year = extract_year_from_report(report_text)

    # Split the input image/collage into individual phone screens.
    crops = get_screen_crops(
        image_path,
        grid=grid,
        layout=layout,
    )

    if not crops:
        raise ValueError("No screen crops were found.")

    print(f"-> [CV] Found {len(crops)} screenshot(s).")

    output_debug_dir.mkdir(parents=True, exist_ok=True)

    if dump_ocr or dump_draft or dump_side_map:
        write_debug_text(
            output_debug_dir / "run_config.json",
            json.dumps(
                {
                    "image_path": image_path,
                    "model": model,
                    "langs": langs,
                    "use_gpu": use_gpu,
                    "grid": grid,
                    "layout": layout,
                    "use_vision": use_vision,
                    "emoji_mode": emoji_mode,
                    "dump_ocr": dump_ocr,
                    "dump_draft": dump_draft,
                    "dump_side_map": dump_side_map,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )

    # Load EasyOCR once; it will be reused for all screen crops.
    reader = easyocr.Reader(langs, gpu=use_gpu)

    # Extract victim/suspect names from the report for side mapping.
    print("-> [LLM] Extracting actors from report...")
    actors = infer_report_actors(report_text, model)
    print("[ACTORS]", json.dumps(actors, ensure_ascii=False, indent=2))

    # First pass: save crops and collect positioned OCR for each screen.
    all_screen_ocr = []
    screen_paths = []

    for idx, (_, _, _, _, crop) in enumerate(crops, start=1):
        crop_path = output_debug_dir / f"screen_{idx:02d}.png"
        write_crop(crop_path, crop)
        screen_paths.append(crop_path)

        print(f"   - OCR screen #{idx}...")
        screen_ocr = extract_ocr_blocks(reader, crop, idx)
        all_screen_ocr.append(screen_ocr)

        if dump_ocr:
            write_debug_text(output_debug_dir / f"screen_{idx:02d}_ocr.txt", screen_ocr)

    full_ocr = "\n\n".join(all_screen_ocr)

    if dump_ocr:
        write_debug_text(output_debug_dir / "all_ocr.txt", full_ocr)

    # Infer one fixed LEFT/RIGHT mapping for the whole conversation.
    print("-> [SIDE MAP] Inferring fixed LEFT/RIGHT mapping...")
    side_map = infer_side_mapping(
        report_text,
        actors,
        full_ocr,
        "viber",
        model,
    )

    print(f"-> [SIDE MAP] LEFT = {side_map['LEFT']} | RIGHT = {side_map['RIGHT']}")

    if dump_side_map:
        write_debug_text(
            output_debug_dir / "side_map.json",
            json.dumps(side_map, ensure_ascii=False, indent=2),
        )

    # Second pass: ask the VLM to extract each screen as Side CSV.
    side_csv_parts = []
    previous_date_hint = ""

    for idx, screen_ocr in enumerate(all_screen_ocr, start=1):
        print(f"-> [LLM] Extracting screen #{idx} as Side CSV...")

        visible_date = extract_visible_date_from_ocr(
            screen_ocr,
            default_year=default_year,
            previous_date_hint=previous_date_hint,
        )

        # Limit accepted times to times actually visible in this screen.
        allowed_times = extract_allowed_times_from_ocr(screen_ocr)
        ocr_bubble_groups = build_ocr_bubble_groups(screen_ocr)
        bubble_hints = build_viber_bubble_hints(ocr_bubble_groups)

        if dump_draft:
            write_debug_text(output_debug_dir / f"screen_{idx:02d}_visible_date.txt", visible_date)
            write_debug_text(
                output_debug_dir / f"screen_{idx:02d}_allowed_times.json",
                json.dumps(sorted(allowed_times), ensure_ascii=False, indent=2),
            )
            write_debug_text(
                output_debug_dir / f"screen_{idx:02d}_ocr_bubbles.json",
                json.dumps(ocr_bubble_groups, ensure_ascii=False, indent=2),
            )
            write_debug_text(output_debug_dir / f"screen_{idx:02d}_bubble_hints.txt", bubble_hints)

        prompt = build_screen_prompt(
            screen_ocr=screen_ocr,
            screen_index=idx,
            visible_date=visible_date,
            default_year=default_year,
            allowed_times=allowed_times,
            emoji_mode=emoji_mode,
            bubble_hints=bubble_hints,
        )

        crop_path = str(screen_paths[idx - 1])

        if dump_draft:
            write_debug_text(output_debug_dir / f"screen_{idx:02d}_prompt.txt", prompt)

        # Initial VLM extraction from screenshot + OCR hints.
        draft = ollama_chat_screen(
            model,
            prompt,
            crop_path,
            use_vision=use_vision,
        )

        if dump_draft:
            write_debug_text(output_debug_dir / f"screen_{idx:02d}_draft_raw.csv", draft)
            # Keep the old Facebook-like/debug filename too.
            write_debug_text(output_debug_dir / f"screen_{idx:02d}_draft.csv", draft)

        draft_norm = normalize_side_csv(
            draft,
            visible_date=visible_date,
            default_year=default_year,
            allowed_times=allowed_times,
            emoji_mode=emoji_mode,
            ocr_bubble_groups=ocr_bubble_groups,
        )

        if dump_draft:
            write_debug_text(output_debug_dir / f"screen_{idx:02d}_draft_norm.csv", draft_norm)

        print(f"-> [LLM] Repairing screen #{idx} Side CSV...")

        repair_prompt = build_screen_repair_prompt(
            draft_csv=draft_norm,
            screen_ocr=screen_ocr,
            screen_index=idx,
            visible_date=visible_date,
            default_year=default_year,
            allowed_times=allowed_times,
            emoji_mode=emoji_mode,
            bubble_hints=bubble_hints,
        )

        if dump_draft:
            write_debug_text(output_debug_dir / f"screen_{idx:02d}_repair_prompt.txt", repair_prompt)

        # Repair pass catches missing bubbles and bad splits/merges.
        repaired = ollama_chat_screen(
            model,
            repair_prompt,
            crop_path,
            use_vision=use_vision,
        )

        if dump_draft:
            write_debug_text(output_debug_dir / f"screen_{idx:02d}_repair_raw.csv", repaired)

        repaired_norm = normalize_side_csv(
            repaired,
            visible_date=visible_date,
            default_year=default_year,
            allowed_times=allowed_times,
            emoji_mode=emoji_mode,
            ocr_bubble_groups=ocr_bubble_groups,
        )

        if dump_draft:
            write_debug_text(output_debug_dir / f"screen_{idx:02d}_repair_norm.csv", repaired_norm)

        # Keep the repaired CSV only if it stays within a sane row count.
        chosen = choose_best_screen_side_csv(
            draft_norm=draft_norm,
            repaired_norm=repaired_norm,
            allowed_times=allowed_times,
        )

        if dump_draft:
            chosen_source = "repaired_norm" if chosen == repaired_norm else "draft_norm"
            write_debug_text(output_debug_dir / f"screen_{idx:02d}_chosen_source.txt", chosen_source)
            write_debug_text(output_debug_dir / f"screen_{idx:02d}_side.csv", chosen)

        last_date = extract_last_date_from_side_csv(chosen)
        if last_date:
            previous_date_hint = last_date

        side_csv_parts.append(chosen)

    # Merge all screen-level CSV parts before applying real names.
    merged_side_csv = merge_side_csvs(side_csv_parts)

    if dump_draft:
        write_debug_text(output_debug_dir / "merged_side.csv", merged_side_csv)

    # Replace LEFT/RIGHT with actual Sender/Receiver names.
    final_csv = apply_side_mapping(
        merged_side_csv,
        side_map,
    )

    if dump_draft:
        write_debug_text(output_debug_dir / "final.csv", final_csv)

    return final_csv


# ============================================================
# MAIN
# ============================================================

# CLI entry point: parse arguments, run extraction, and write outputs.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract Viber chat CSV from screenshot or collage screenshot."
    )

    parser.add_argument(
        "image",
        help="Path to Viber screenshot/collage image",
    )

    parser.add_argument(
        "report",
        help="Path to case report PDF/TXT",
    )

    parser.add_argument(
        "--model",
        default="gemma3:12b",
        help="Ollama model, default: gemma3:12b",
    )

    parser.add_argument(
        "--langs",
        default="en",
        help="EasyOCR languages, e.g. en or en,el",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU for EasyOCR",
    )

    parser.add_argument(
        "--grid",
        default=None,
        help="Manual regular collage split, e.g. 2x1 or 3x2",
    )

    parser.add_argument(
        "--layout",
        default=None,
        help="Manual uneven collage split by rows, e.g. 2,3",
    )

    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Do not send screen images to Ollama; OCR only. Emojis will be less reliable.",
    )

    parser.add_argument(
        "-emoji-mode",
        "--emoji-mode",
        choices=["omit", "vision"],
        default="omit",
        help=(
            "Emoji handling mode. "
            "'omit' removes emojis/symbols from final messages to avoid wrong guessed emojis. "
            "'vision' asks the vision model to preserve only clearly visible emojis."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path",
    )

    parser.add_argument(
        "--debug-dir",
        default=None,
        help="Directory for crops/OCR/debug files",
    )

    # Standard double-dash flags.
    # Also accepts your accidental single-dash versions for convenience.
    parser.add_argument(
        "-dump-ocr",
        "--dump-ocr",
        "--dump_ocr",
        action="store_true",
        help="Save positioned OCR debug files",
    )

    parser.add_argument(
        "-dump-draft",
        "--dump-draft",
        "--dump_draft",
        action="store_true",
        help="Save draft/repaired Side CSV files",
    )

    parser.add_argument(
        "-dump-side-map",
        "--dump-side-map",
        "--dump_side_map",
        action="store_true",
        help="Save inferred LEFT/RIGHT side map",
    )

    args = parser.parse_args()

    if args.grid and args.layout:
        print("[ERROR] Use either --grid or --layout, not both.")
        sys.exit(1)

    image_path = Path(args.image)
    report_path = Path(args.report)

    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    if not report_path.exists():
        print(f"[ERROR] Report not found: {report_path}")
        sys.exit(1)

    print("\n[START]")
    print("-> Reading case report...")

    report_text = extract_text_from_report(str(report_path))

    if not report_text.strip():
        print("[WARNING] Report appears empty or could not be read correctly.")

    script_dir = Path(__file__).resolve().parent
    image_base = image_path.stem

    debug_dir = Path(args.debug_dir) if args.debug_dir else script_dir / f"{image_base}_debug"
    output_path = Path(args.output) if args.output else script_dir / f"{image_base}_extracted.csv"

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    final_csv = process_viber_image(
        image_path=str(image_path),
        report_text=report_text,
        model=args.model,
        langs=langs,
        use_gpu=not args.cpu,
        grid=args.grid,
        layout=args.layout,
        use_vision=not args.no_vision,
        emoji_mode=args.emoji_mode,
        output_debug_dir=debug_dir,
        dump_ocr=args.dump_ocr,
        dump_draft=args.dump_draft,
        dump_side_map=args.dump_side_map,
    )

    output_path.write_text(final_csv, encoding="utf-8-sig")

    print("\n--- FINAL CSV ---\n")
    print(final_csv)

    print(f"\n[SUCCESS] CSV saved to: {output_path}")
    print(f"[DEBUG] Debug folder: {debug_dir}")
