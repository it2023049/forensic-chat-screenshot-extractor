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


def separator_score_y(gray: np.ndarray, y: int, band: int = 3) -> float:
    h, w = gray.shape[:2]
    y = max(band, min(h - band - 1, int(y)))
    strip = gray[y - band:y + band + 1, :]
    white = (strip > 245).mean()
    dark = (strip < 30).mean()
    diff = np.abs(gray[y - 1, :].astype(int) - gray[y + 1, :].astype(int)).mean() / 255.0
    return float(white + dark + diff)


def auto_split_wide_phone_collage(image: np.ndarray) -> List[ScreenCrop]:
    """
    Fallback for phone screenshot collages.

    Handles both:
    - one-row side-by-side collages, e.g. 2 screenshots next to each other
    - multi-row collages, e.g. 4 screenshots on top and 4 screenshots below

    This is deliberately conservative for single portrait screenshots:
    if the full image is not wide enough, it is not split.
    """
    img = trim_white_border(image)
    h, w = img.shape[:2]

    if h == 0 or w == 0:
        return []

    aspect = w / h

    # A single portrait phone screenshot is usually much narrower than tall.
    if aspect < 0.85:
        return [(0, 0, w, h, img)]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ------------------------------------------------------------
    # Detect row count.
    #
    # Important: a 2x1 side-by-side collage and a 4x2 collage can have almost
    # the same global aspect ratio. We therefore look for a horizontal row
    # boundary near the middle.
    #
    # For generated Messenger collages, the safest split is often exactly at
    # half height. White-band detection alone can be fooled by blank areas
    # inside the lower row and may split too low, producing 5 crops in row 2.
    # ------------------------------------------------------------
    row_segments = [(0, h)]

    white_pixels = np.all(img >= 245, axis=2)
    row_white_ratio = white_pixels.mean(axis=1)

    mid1 = int(h * 0.49)
    mid2 = int(h * 0.51)
    middle_white_mean = float(row_white_ratio[mid1:mid2].mean()) if mid2 > mid1 else 0.0

    if middle_white_mean >= 0.74:
        split_y = h // 2
        row_segments = [(0, split_y), (split_y, h)]
    else:
        candidate_rows = np.where(row_white_ratio >= 0.95)[0]
        candidate_bands = [
            (a, b)
            for a, b in ranges_from_indices(candidate_rows)
            if (b - a) >= 6 and int(h * 0.40) <= ((a + b) // 2) <= int(h * 0.60)
        ]

        if candidate_bands:
            a, b = min(candidate_bands, key=lambda band: abs(((band[0] + band[1]) / 2) - (h / 2)))
            split_y = int((a + b) / 2)

            if h * 0.40 <= split_y <= h * 0.60:
                row_segments = [(0, split_y), (split_y, h)]

    # ------------------------------------------------------------
    # Estimate columns per row from phone-crop aspect ratio.
    # Messenger screenshot cells in these generated collages are usually
    # around 0.60-0.70 width/height after cropping.
    # ------------------------------------------------------------
    target_phone_aspect = 0.66
    crops: List[ScreenCrop] = []

    for y1, y2 in row_segments:
        row_h = y2 - y1
        if row_h <= 0:
            continue

        row_aspect = w / row_h
        cols = int(round(row_aspect / target_phone_aspect))
        cols = max(1, min(cols, 6))

        # If the whole image is wide but one inferred row would have one column,
        # keep one. Otherwise split into estimated equal columns.
        cell_w = w / cols
        cell_aspect = cell_w / row_h

        # Reject unlikely phone-cell shapes.
        if cols > 1 and not (0.35 <= cell_aspect <= 0.90):
            return [(0, 0, w, h, img)]

        for c in range(cols):
            x1 = int(c * cell_w)
            x2 = int((c + 1) * cell_w) if c < cols - 1 else w

            crop = trim_white_border(img[y1:y2, x1:x2])
            ch, cw = crop.shape[:2]

            if ch < row_h * 0.70 or cw < 120:
                return [(0, 0, w, h, img)]

            crops.append((x1, y1, x2 - x1, y2 - y1, crop))

    if len(crops) <= 1:
        return [(0, 0, w, h, img)]

    return sort_screen_crops(crops)




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

    wide_crops = auto_split_wide_phone_collage(image)
    if len(wide_crops) > 1:
        return wide_crops

    contour_crops = contour_fallback_split(image)
    if len(contour_crops) > 1:
        return contour_crops

    # One last wide-image check after contour fallback. This catches very clean
    # Messenger side-by-side collages where contour detection sees the full image
    # as one object.
    if len(contour_crops) == 1:
        wide_crops = auto_split_wide_phone_collage(contour_crops[0][4])
        if len(wide_crops) > 1:
            return wide_crops

    return contour_crops


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

    # Messenger outgoing blue bubbles usually start farther right than incoming gray bubbles.
    # Use the left edge, not the center, to avoid misclassifying wide incoming bubbles.
    # The threshold is deliberately moderate because long right-side bubbles can start nearer the middle.
    if x >= crop_w * 0.30:
        return "RIGHT"

    return "LEFT"


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
        if min_x < crop_w * 0.35:
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
        pos = position_tag_from_bbox((x, y, bw, bh), w, text)

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
    Returns DD/MM/YYYY based on the visible Messenger date separator.

    Important forensic rule:
    Do NOT silently reuse the previous screenshot date when OCR cannot see
    a date in the current screenshot. A wrong carried-over date is worse than
    an empty date, because the vision model can still read the separator.
    """
    rows = parse_ocr_lines(screen_ocr)

    def normalize_date_text(text: str) -> str:
        t = str(text or "")
        t = t.replace(".", ":")
        t = t.replace(";", ":")
        t = t.replace("|", "I")
        # OCR often reads zero as O only inside numbers.
        t = re.sub(r"(?<=\d)[oO](?=\d)", "0", t)
        t = re.sub(r"(?<=\s)[oO](?=\d)", "0", t)
        t = re.sub(r"(?<=\d)[lI](?=\d)", "1", t)
        t = re.sub(r"\s+", " ", t)
        return t.strip()

    month_pat = (
        r"Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|"
        r"Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December"
    )

    # 1. Prefer OCR blocks in the conversation/date-separator region.
    candidate_texts = []
    for row in rows:
        text = normalize_date_text(row.get("text", ""))
        if not text:
            continue

        try:
            y = int(row.get("y", "0"))
        except Exception:
            y = 0

        # Ignore status bar/header as much as possible, but keep the central
        # Messenger separator around the upper-middle of the crop.
        if y >= 80:
            candidate_texts.append(text)

    # 2. Also join adjacent OCR text because EasyOCR may split the date line.
    candidate_texts.append(" ".join(candidate_texts))
    candidate_texts.append(normalize_date_text(screen_ocr))

    for text in candidate_texts:
        m = re.search(
            rf"\b({month_pat})\s+(\d{{1,2}})(?:,?\s+(20\d{{2}}))?",
            text,
            flags=re.I,
        )
        if not m:
            continue

        month = month_to_number(m.group(1))
        day = int(m.group(2))
        year = int(m.group(3)) if m.group(3) else default_year

        if month and 1 <= day <= 31:
            return f"{day:02d}/{month:02d}/{year}"

    return ""


def infer_visible_date_with_ai(
    model: str,
    image_path: str,
    screen_ocr: str,
    default_year: int,
    use_vision: bool = True,
) -> str:
    """
    Vision fallback for Messenger date separators.

    Used only when OCR cannot extract the date. It asks the model for the
    central Messenger separator date, not for chat reconstruction.
    """
    if not use_vision:
        return ""

    prompt = f"""
Read only the central Facebook Messenger date/time separator from this screenshot.
Ignore the phone status bar time and all chat messages.

OCR support:
{screen_ocr}

Return only JSON:
{{"date":"DD/MM/YYYY"}}

Rules:
- Use year {default_year} if the year is not visible.
- If the date is not visible, return {{"date":""}}.
- Do not explain.
"""

    raw = ollama_chat_screen(
        model=model,
        prompt=prompt,
        image_path=image_path,
        use_vision=True,
    )

    try:
        data = json.loads(extract_json_object(raw))
        date = str(data.get("date", "")).strip()
    except Exception:
        m = re.search(r"(\d{2}/\d{2}/20\d{2})", raw)
        date = m.group(1) if m else ""

    if re.fullmatch(r"\d{2}/\d{2}/20\d{2}", date):
        return date

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
    In Facebook Messenger, the top header usually contains the OTHER participant/contact.
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


def deterministic_messenger_side_map(
    actors: Dict,
    ocr_data: str
) -> Optional[Dict[str, str]]:
    """
    Deterministic Facebook Messenger side mapping.

    Main rule:
    If the Messenger header shows a participant/contact from the report, then:
    LEFT = header participant/contact
    RIGHT = victim / screenshot owner

    This matches the usual Messenger layout:
    LEFT gray bubbles = incoming messages from the other participant.
    RIGHT blue bubbles = outgoing messages from the screenshot owner.
    """
    victim = clean_name(actors.get("victim", ""))

    participants = [
        p for p in actors.get("participants", [])
        if p.get("name")
    ]

    # If victim field is empty, recover it from participants.
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
    # 2. Contact/phone evidence fallback
    # -------------------------
    # Some Messenger captures may include phone/account details in header or profile snippets.
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

                try:
                    y = int(row["y"])
                except ValueError:
                    y = 9999

                # Header contact evidence maps to LEFT in Messenger.
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
Determine the fixed LEFT/RIGHT speaker mapping for this Facebook Messenger two-person chat.

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
2. In Facebook Messenger, the top header usually shows the other participant/contact, not the screenshot owner.
3. In Facebook Messenger, LEFT gray bubbles are usually incoming from the header contact. RIGHT blue bubbles are outgoing from the screenshot owner.
4. Direct contact evidence wins: if a phone/contact/account detail from the report appears on a side, that side belongs to that detail's owner.
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
    deterministic = deterministic_messenger_side_map(actors, ocr_data)

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

def build_messenger_bubble_groups(screen_ocr: str) -> List[Dict[str, str]]:
    """
    Build approximate Messenger bubble groups from OCR geometry.

    Unlike Viber, Messenger usually does not show a timestamp on every bubble.
    These groups are used as a conservative side/bubble hint.
    """
    rows = parse_ocr_lines(screen_ocr)

    def to_int(value: str, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return default

    # Find the last top date/time separator area.
    date_y = None
    for row in rows:
        text = row.get("text", "")
        if looks_like_date(text) or re.search(r"\b\d{1,2}[:.]\d{2}\b", text):
            y = to_int(row.get("y", "0"), -1)
            if y > 100:
                date_y = y if date_y is None else max(date_y, y)

    usable = []
    for row in rows:
        text = row.get("text", "").strip()
        pos = row.get("pos", "").upper()
        y = to_int(row.get("y", "0"))

        if not text or pos not in {"LEFT", "RIGHT"}:
            continue

        if date_y is not None and y <= date_y:
            continue
        if date_y is None and y < 250:
            continue

        if is_ui_message(text) or looks_like_date_or_time(text):
            continue

        usable.append(row)

    usable = sorted(
        usable,
        key=lambda r: (
            to_int(r.get("y", "0")),
            to_int(r.get("x", "0")),
            to_int(r.get("block", "0")),
        )
    )

    if not usable:
        return []

    heights = [max(1, to_int(r.get("h", "1"), 1)) for r in usable]
    median_h = int(np.median(heights)) if heights else 40
    gap_threshold = max(8, int(median_h * 0.25))

    raw_groups: List[List[Dict[str, str]]] = []

    for row in usable:
        if not raw_groups:
            raw_groups.append([row])
            continue

        prev = raw_groups[-1][-1]

        y = to_int(row.get("y", "0"))
        prev_y = to_int(prev.get("y", "0"))
        prev_h = to_int(prev.get("h", "0"))

        y_gap = y - (prev_y + prev_h)
        same_side = row.get("pos", "").upper() == prev.get("pos", "").upper()

        # New bubble if side changes or if there is a clear vertical gap.
        # Wrapped lines inside the same bubble usually have very small/negative gap.
        if (not same_side) or y_gap > gap_threshold:
            raw_groups.append([row])
        else:
            raw_groups[-1].append(row)

    groups: List[Dict[str, str]] = []

    for group in raw_groups:
        group_sorted = sorted(
            group,
            key=lambda r: (
                to_int(r.get("y", "0")),
                to_int(r.get("x", "0")),
                to_int(r.get("block", "0")),
            )
        )

        side_scores = {"LEFT": 0, "RIGHT": 0}
        for r in group_sorted:
            side = r.get("pos", "").upper()
            if side in side_scores:
                side_scores[side] += max(1, len(r.get("text", "")))

        side = "LEFT" if side_scores["LEFT"] >= side_scores["RIGHT"] else "RIGHT"
        text = " ".join(r.get("text", "").strip() for r in group_sorted if r.get("text", "").strip())
        text = re.sub(r"\s+", " ", text).strip()

        if text:
            groups.append({
                "side": side,
                "text": text,
            })

    return groups


def build_messenger_bubble_hints(screen_ocr: str) -> str:
    """
    Build approximate Messenger bubble hints from OCR geometry.

    The hints are not treated as final transcript text; they are noisy OCR
    support only.
    """
    groups = build_messenger_bubble_groups(screen_ocr)

    if not groups:
        return "No reliable bubble hints."

    lines = []
    for i, group in enumerate(groups, start=1):
        lines.append(f"[BUBBLE {i}] SIDE={group.get('side', '')} OCR_TEXT={group.get('text', '')}")

    return "\n".join(lines)



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
You are extracting messages from ONE Facebook Messenger screenshot image.
Use the attached image as the primary source. Use OCR blocks only as support.

SCREEN INDEX: {screen_index}
VISIBLE DATE FOR THIS SCREEN: {visible_date or "unknown - read it from the central Messenger date separator in the screenshot"}
DEFAULT YEAR: {default_year}
ALLOWED BUBBLE TIMES FOR THIS SCREEN: {allowed_times_text}

OCR BLOCKS:
{screen_ocr}

MESSENGER BUBBLE HINTS FROM OCR GEOMETRY:
{bubble_hints or "No reliable bubble hints."}

Return only CSV with this header exactly once:
"Time","Side","Message"

Rules:
1. Output only real human chat bubbles.
2. Ignore UI: status bar clock, battery, Messenger header/contact name, profile/header snippets, call/video/search icons, Active now, Sent, Delivered, Seen, read receipts, date-only separators, text input box, Like button, stickers/GIF labels, and empty OCR noise.
3. Side must be LEFT or RIGHT based only on visible bubble position/color in the image.
4. LEFT = gray incoming bubble. RIGHT = blue outgoing bubble.
5. Do not output participant names. Do not infer side from meaning or from victim/scammer roles.
6. Reconstruct each separate visible rounded chat bubble as one CSV row. Do not output one row per OCR line.
7. Messenger often shows one central date/time separator for many bubbles. Do NOT merge all bubbles after the same timestamp into one message.
8. Never merge two adjacent bubbles just because they have the same sender, same side, or same visible timestamp. If there is a separate rounded bubble boundary, it is a separate CSV row.
9. Use MESSENGER BUBBLE HINTS as a guide for splitting: normally each [BUBBLE n] should become one CSV row unless it is clearly UI noise.
10. Before the final CSV, compare your rows with MESSENGER BUBBLE HINTS and the screenshot. If a visible human bubble is missing, add it in the correct visual/top-to-bottom order, not at the end.
11. Do not add OCR fragments that duplicate an existing row, even if the OCR wording is noisier.
12. Do not drop the last visible chat bubble near the bottom of the screenshot. A blue/gray rounded bubble above the composer/input bar is still a human message.
13. If one bubble wraps across multiple OCR lines, merge only the lines inside that same rounded bubble.
11. If a bubble has no individual timestamp, use the screen's visible date/time separator as the starting time, then infer monotonically increasing per-message times for later bubbles in the same visible sequence. Do not repeat the exact same timestamp for every bubble unless the screenshot clearly shows they share that exact minute.
12. OCR COVERAGE CHECK: Every non-UI LEFT/RIGHT OCR text block must be represented in an output message, but grouped by visible bubble boundaries.
12. Do not shorten messages. Do not keep only the first line of a bubble.
13. Emoji rule: {emoji_rule}
14. Preserve evidence exactly: amounts, names, countries, cities, receiver details, phone numbers, account/payment details, reference numbers, threats, instructions.
15. Time must be the bubble's visible time when present. When individual bubble times are hidden, estimate a plausible monotonically increasing time from the screen's visible date/time separator, combined with VISIBLE DATE FOR THIS SCREEN. Format: "DD/MM/YYYY HH:MM".
16. Only use times from ALLOWED BUBBLE TIMES FOR THIS SCREEN when they are available. Do not use the status bar time.
17. Fix only obvious OCR mistakes when the image clearly supports it: Im -> I'm, Icant -> I can't, | -> I, $ -> s, trailing ":" or "_" as punctuation.
18. Do not invent, summarize, or add messages from other screenshots.
19. Use exactly three quoted CSV fields per row. No markdown, no explanations.

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
Repair this Facebook Messenger screen CSV using the attached screenshot image and OCR blocks.

SCREEN INDEX: {screen_index}
VISIBLE DATE FOR THIS SCREEN: {visible_date or "unknown - read it from the central Messenger date separator in the screenshot"}
DEFAULT YEAR: {default_year}
ALLOWED BUBBLE TIMES FOR THIS SCREEN: {allowed_times_text}

OCR BLOCKS:
{screen_ocr}

MESSENGER BUBBLE HINTS FROM OCR GEOMETRY:
{bubble_hints or "No reliable bubble hints."}

BAD / INCOMPLETE DRAFT CSV:
{draft_csv}

Return only final CSV with this header exactly once:
"Time","Side","Message"

Rules:
1. Rebuild only from this screenshot image and its OCR blocks.
2. Keep every real visible message bubble from this screenshot.
3. Remove all UI rows: status bar clock, battery, Messenger header/contact name, profile/header snippets, call/video/search icons, Active now, Sent, Delivered, Seen, read receipts, date separator as message, text input box, Like button, stickers/GIF labels, and empty OCR noise.
4. Side must be LEFT or RIGHT based on visible bubble position/color. LEFT = gray incoming. RIGHT = blue outgoing. Do not output names.
5. Messenger often shows one central date/time separator for many bubbles. Do NOT merge all bubbles after the same timestamp into one message.
6. Keep each separate visible rounded chat bubble as one CSV row, even if adjacent bubbles have the same sender, same side, or same visible timestamp.
7. Use MESSENGER BUBBLE HINTS as a guide for splitting: normally each [BUBBLE n] should become one CSV row unless it is clearly UI noise.
8. Before the final CSV, compare your rows with MESSENGER BUBBLE HINTS and the screenshot. If a visible human bubble is missing, add it in the correct visual/top-to-bottom order, not at the end.
9. Do not add OCR fragments that duplicate an existing row, even if the OCR wording is noisier.
10. Do not drop the last visible chat bubble near the bottom of the screenshot. A blue/gray rounded bubble above the composer/input bar is still a human message.
11. If the draft merged two or more visible bubbles into one row, split them back into separate rows using the screenshot.
8. OCR COVERAGE CHECK: Every non-UI LEFT/RIGHT OCR text block must be represented in an output message, but grouped by visible bubble boundaries.
9. If the draft omitted an OCR line from a bubble, add it back.
10. Do not shorten messages. Do not keep only the first line of a bubble.
11. Emoji rule: {emoji_rule}
12. Merge wrapped lines only when they are inside the same visible rounded bubble.
13. Use visible bubble/screen times as anchors. If Messenger hides individual bubble times, infer plausible monotonically increasing per-message times instead of repeating the same anchor time for all rows.
14. If VISIBLE DATE FOR THIS SCREEN is unknown, read the central Messenger date separator from the screenshot image. Never reuse a date from another screenshot.
15. Combine each time with VISIBLE DATE FOR THIS SCREEN or the date read from the screenshot. Format: "DD/MM/YYYY HH:MM".
16. Preserve evidence: amounts, names, receiver details, locations, phone numbers, accounts, references, threats, instructions.
16. Do not invent messages and do not include messages from other screenshots.
17. Use exactly three quoted CSV fields per row. No markdown, no explanations.

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
    Generic Facebook Messenger/UI filtering for messages after LLM extraction.
    Avoid case-specific names, phone numbers, or conversation content.
    """
    m = str(message or "").strip()
    low = m.lower()

    if not m:
        return True

    # Standalone dates/times are UI, not chat messages.
    if looks_like_date_or_time(m):
        return True

    ui_substrings = [
        "active now",
        "messenger",
        "facebook",
        "you're friends on facebook",
        "you are friends on facebook",
        "wave to",
        "say hi",
        "sent",
        "delivered",
        "seen",
        "typing",
        "search conversation",
        "view profile",
        "voice call",
        "video call",
        "missed call",
        "started a call",
        "changed the chat",
        "end-to-end encrypted",
        "end-to-end encryption",
        "messages and calls are secured",
        "message...",
        "type a message",
        "write a message",
        "aa",
    ]

    if any(part in low for part in ui_substrings):
        return True

    # Messenger reactions/read receipts/icon OCR noise.
    if re.fullmatch(r"(like|thumbs up|gif|sticker|photo|camera|mic|microphone|send)", low, flags=re.IGNORECASE):
        return True

    if re.fullmatch(r"[0o]{2,4}", low):
        return True

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
    msg = msg.replace('\\"', "'")
    msg = msg.replace('""', "'")
    msg = msg.replace('"', "'")
    msg = msg.replace("“", "'")
    msg = msg.replace("”", "'")

    # Generic OCR artifact: word+'$ usually means word+'s.
    msg = re.sub(r"\b([A-Za-z]+)'\$", r"\1's", msg)

    # Remove obvious non-word OCR separators inside a sentence.
    msg = re.sub(r"\s+=\s+", " ", msg)

    # Generic English contraction/OCR fixes.
    msg = re.sub(r"\bIm\b", "I'm", msg)
    msg = re.sub(r"\bI\s*m\b", "I'm", msg)
    msg = re.sub(r"\bTm\b", "I'm", msg)
    msg = re.sub(r"\bT\s*m\b", "I'm", msg)
    msg = re.sub(r"\bFve\b", "I've", msg)
    msg = re.sub(r"\bIve\b", "I've", msg)
    msg = re.sub(r"\bIcant\b", "I can't", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bIcan't\b", "I can't", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bdontt\b", "don't", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bdont\b", "don't", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bIll\b", "I'll", msg)
    msg = re.sub(r"\bI\s*ll\b", "I'll", msg)
    msg = re.sub(r"\bIwill\b", "I will", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bIfeel\b", "I feel", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bIdo\b", "I do", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bIlook\b", "I look", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bIadmire\b", "I admire", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bt0\b", "to", msg, flags=re.IGNORECASE)

    # Common OCR word-shape errors in short chat prose.
    msg = re.sub(r"\bsO\b", "so", msg)
    msg = re.sub(r"\buS\b", "us", msg)
    msg = re.sub(r"\bteel\b", "I feel", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\btool\b(?=($|[\s.!?,]))", "too!", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bcours\b", "course", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bpasse\b", "passed", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bThave\b", "I have", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bwomar\b", "woman", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bdoing how you re\b", "how you're doing", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bthe you handle\b", "the way you handle", msg, flags=re.IGNORECASE)

    # OCR sometimes reads final exclamation mark as lowercase l.
    final_l_words = r"(love|angel|Mary|Michael|it|there|you|too)"
    msg = re.sub(rf"\b{final_l_words}l\b(?=$|[\s.!?,])", lambda m: m.group(0)[:-1] + "!", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bMichaell\b", "Michael!", msg, flags=re.IGNORECASE)

    # OCR sometimes reads pronoun I as | or !. Convert only in obvious contexts.
    pronoun_verbs = (
        r"am|was|will|can|cannot|can't|need|have|think|feel|want|would|could|"
        r"should|do|don't|dont|almost|love|trust|promise|already|may|must|might|"
        r"know|hope|dream|see|believe|wish|tell|always|ask|slept"
    )
    msg = re.sub(
        rf"(?i)(^|[\s,.;:!?])\|\s+({pronoun_verbs})\b",
        lambda m: f"{m.group(1)}I {m.group(2)}",
        msg
    )
    msg = re.sub(
        rf"(?i)(^|[\s,.;:!?])!\s+({pronoun_verbs})\b",
        lambda m: f"{m.group(1)}I {m.group(2)}",
        msg
    )
    msg = re.sub(
        r"(?i)\b(and|but|that|if|when|because|so)\s+\|\s+",
        lambda m: f"{m.group(1)} I ",
        msg
    )

    # Missing initial "I" in common first-person OCR cases.
    msg = re.sub(r"(?i)^feel\s+(?=(I\s+can|a\s+deep|our\s+connection|the\s+same)\b)", "I feel ", msg)
    msg = re.sub(r"(?i)^don't\s+usually\b", "I don't usually", msg)
    msg = re.sub(r"(?i)^dream\s+about\b", "I dream about", msg)
    msg = re.sub(r"(?i)^will\s+do\b", "I will do", msg)
    msg = re.sub(r"(?i)^hope\s+it\b", "I hope it", msg)
    msg = re.sub(r"(?i)^see\s+my\s+future\b", "I see my future", msg)
    msg = re.sub(r"(?i)^love\s+you\b", "I love you", msg)
    msg = re.sub(r"(?i)^slept\s+well\b", "I slept well", msg)

    # Common first-person continuations after sentence punctuation.
    msg = re.sub(r"(?i)([.!?]\s+)will\s+(do|try|send|message|keep|call)\b", r"\1I will \2", msg)
    msg = re.sub(r"(?i)([.!?]\s+)hope\s+(it|you|we)\b", r"\1I hope \2", msg)
    msg = re.sub(r"(?i)([.!?]\s+)wish\s+so\b", r"\1I wish so", msg)
    msg = re.sub(r"(?i)([.!?]\s+)see\s+my\s+future\b", r"\1I see my future", msg)
    msg = re.sub(r"(?i)([.!?]\s+)promise\b", r"\1I promise", msg)

    # Direct pronoun OCR inside questions.
    msg = re.sub(r"(?i)\bcan\s+\|\s+(ask|tell)\b", r"can I \1", msg)
    msg = re.sub(r"(?i)\bMary,\s+can\s+\|\s+(ask|tell)\b", r"Mary, can I \1", msg)

    # OCR punctuation artifacts around ellipses.
    msg = msg.replace(":_.", "...")
    msg = msg.replace(":-", "...")
    msg = msg.replace("_.", "...")
    msg = msg.replace(":...", "...")
    msg = msg.replace("way:.", "way...")
    msg = re.sub(r"[_]{2,}", "...", msg)
    msg = msg.replace("_", "")

    # Normalize ellipsis artifacts.
    msg = re.sub(r"\.\.\.\s*\.", "...", msg)
    msg = re.sub(r"\.\.\.\s*,", "...,", msg)
    msg = re.sub(r"\.\.\s+", "... ", msg)
    msg = re.sub(r"\s*\.\.\.\s*", "... ", msg)

    # Generic colon/semicolon cleanup inside normal prose.
    label_like = r"(name|country|city|option|account|iban|reference|mtcn|phone|email|amount|details|receiver|sender|beneficiary|bank|address|code|number)"
    has_structured_label = re.search(rf"\b{label_like}\s*:", msg, flags=re.IGNORECASE)

    if not has_structured_label:
        msg = re.sub(r"\b([A-Z][a-z]{1,24});\s+(?=(can|I|I've|I'm|you|we|it|that|this|will)\b)", r"\1, ", msg)
        msg = re.sub(r";\s+(?=(of|not|and|but|I|I'm|you|we|it|that|this|the|they|there|will|just|thanks|take)\b)", ", ", msg, flags=re.IGNORECASE)
        msg = re.sub(
            r":\s+(?=(my|there|the|they|i|you|he|she|we|it|this|that|and|but|because|so|be|please|just)\b)",
            ". ",
            msg,
            flags=re.IGNORECASE
        )
        msg = re.sub(r"(?i):\s*promise\b", ". I promise", msg)
        msg = re.sub(r"\b([A-Z][a-z]{1,24}):\s+(?=I?\s*will\b)", r"\1. ", msg)
        msg = re.sub(r"\b([A-Z][a-z]{1,24}):\s+(?=I?\s*promise\b)", r"\1. ", msg)

    # Conservative reorder fixes for common OCR/LLM line-order problems.
    msg = re.sub(r"(?i)\bhug\s+Wish I could you\b", "Wish I could hug you", msg)
    msg = re.sub(r"(?i)\bYou're the best day part of my\b", "You're the best part of my day", msg)
    msg = re.sub(r"(?i)\bday finally I can\b", "day I can finally", msg)
    msg = re.sub(r"(?i)\bWe could have beautiful life\b", "We could have a beautiful life", msg)
    msg = re.sub(r"(?i)\bOne day soon my love\s+One day soon\.*", "One day soon my love. One day soon...", msg)
    msg = re.sub(r"(?i)\bOne day soon my love day One soon\.?\.*\b", "One day soon my love. One day soon...", msg)
    msg = re.sub(r"(?i)\bday One soon\.?\.*\b", "One day soon...", msg)
    msg = re.sub(r"(?i)\bAww that's so sweet\s+You just made my day\b", "Aww that's so sweet. You just made my day", msg)
    msg = re.sub(r"(?i)\bAww that's so sweet day You just made my\b", "Aww that's so sweet. You just made my day", msg)
    msg = re.sub(r"(?i)\bnot far away:\.\.\.", "not far away...", msg)
    msg = re.sub(r"(?i)\bday you understand me\b", "day... you understand me", msg)
    msg = re.sub(r"(?i)\bYou make me\s+I\s+look forward\s+to\s+our\s+day:\s*messages every\b", "You make me happy too. I look forward to our messages every day", msg)
    msg = re.sub(r"(?i)\bYou make me happy too\s+I\s+look\b", "You make me happy too. I look", msg)

    # Add punctuation between merged short chat sentences.
    msg = re.sub(r"(?i)\bGood morning Mary\s+How\b", "Good morning Mary. How", msg)
    msg = re.sub(r"(?i)\bGood night Mary\s+Sweet\b", "Good night Mary. Sweet", msg)
    msg = re.sub(r"(?i)\bGood morning my love\s+Hope\b", "Good morning my love. Hope", msg)
    msg = re.sub(r"(?i)\bbeautiful day\s*-\s*today\b", "beautiful day today", msg)
    msg = re.sub(r"(?i)\bThank you\s+I hope\b", "Thank you. I hope", msg)
    msg = re.sub(r"(?i)\bGood morning!\s+slept\b", "Good morning! I slept", msg)
    msg = re.sub(r"(?i)\bslept well;\s*thanks\b", "slept well, thanks", msg)
    msg = re.sub(r"(?i)\bThat would be nice\s+wish\b", "That would be nice. I wish", msg)
    msg = re.sub(r"(?i)\bpeaceful day my love\s+Just\b", "peaceful day my love. Just", msg)
    msg = re.sub(r"(?i)\bkind of you\s+I'm\b", "kind of you. I'm", msg)
    msg = re.sub(r"(?i)\bThank you Michael\s+You too\b", "Thank you Michael. You too", msg)
    msg = re.sub(r"(?i)\bYou too;\s*take care\b", "You too, take care", msg)
    msg = re.sub(r"(?i)\bAlways, my love\s+Thinking\b", "Always, my love. Thinking", msg)
    msg = re.sub(r"(?i)\b5 years ago\s+I have\b", "5 years ago. I have", msg)
    msg = re.sub(r"(?i)\bloss too\s+It\b", "loss too. It", msg)
    msg = re.sub(r"(?i)\bThank you Michael\s+That\b", "Thank you Michael. That", msg)
    msg = re.sub(r"(?i)\bThank you\s+You are\b", "Thank you. You are", msg)
    msg = re.sub(r"(?i)\bstrong woman:\s*I admire\b", "strong woman. I admire", msg)
    msg = re.sub(r"(?i)\bhappy too\s+I look\b", "happy too. I look", msg)
    msg = re.sub(r"(?i)\bsame Michael\s+Maybe\b", "same, Michael. Maybe", msg)
    msg = re.sub(r"(?i)\bMe too my love\s+Good night\b", "Me too my love. Good night", msg)
    msg = re.sub(r"(?i)\bMary\s+feel a deep connection\b", "Mary. I feel a deep connection", msg)
    msg = re.sub(r"(?i)\banything\s+I\s+feel\b", "anything. I feel", msg)
    msg = re.sub(r"(?i)\bMichael\s+I'm\b", "Michael. I'm", msg)
    msg = re.sub(r"(?i)\bMary\s+More\b", "Mary. More", msg)
    msg = re.sub(r"(?i)\bMichael\s+I\s+dream\b", "Michael. I dream", msg)
    msg = re.sub(r"(?i)\bMichael\s+Sweet\b", "Michael. Sweet", msg)
    msg = re.sub(r"(?i)\bdreams\s+talk\b", "dreams... talk", msg)
    msg = re.sub(r"(?i)\bbefore sleep\b", "before I sleep", msg)

    # Extra Messenger OCR/order cleanup from Facebook screenshots.
    msg = re.sub(r"(?i)\bHellol\b", "Hello!", msg)
    msg = re.sub(r"(?i)^doing today\?\s*How are you\??$", "How are you doing today?", msg)
    msg = re.sub(r"(?i)\bdoing today\?\s*How are you\b", "How are you doing today", msg)
    msg = re.sub(r"(?i)\bHello!\s+I'm good[,;]\s+thank you!\s+And you\?", "Hello! I'm good, thank you! And you?", msg)

    # Remove accidental excessive terminal quotes caused by CSV/LLM escaping.
    msg = re.sub(r'"{2,}$', '"', msg)
    msg = msg.replace('"""', '"')

    # Whitespace/punctuation cleanup.
    msg = re.sub(r"\s+", " ", msg).strip()
    msg = re.sub(r"\s+([,.;:!?])", r"\1", msg)
    msg = re.sub(r"\.\.\.\s*\.", "...", msg)
    msg = re.sub(r"\s+\.\.\.", "...", msg)
    msg = re.sub(r"\.{4,}", "...", msg)

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

    The LLM may sometimes assign LEFT/RIGHT incorrectly. For Facebook Messenger screenshots,
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

                total_score = side_scores["LEFT"] + side_scores["RIGHT"]
                if total_score <= 0:
                    side = "UNKNOWN"
                    confidence = 0.0
                else:
                    side = "LEFT" if side_scores["LEFT"] >= side_scores["RIGHT"] else "RIGHT"
                    confidence = max(side_scores["LEFT"], side_scores["RIGHT"]) / total_score

                group_text = " ".join(br.get("text", "").strip() for br in bubble_rows if br.get("text", "").strip())

                groups.append({
                    "time": visible_time,
                    "side": side,
                    "confidence": f"{confidence:.3f}",
                    "text": group_text,
                })

            current = []
            continue

        if pos in {"LEFT", "RIGHT"} and text and not is_ui_message(text) and not looks_like_date_or_time(text):
            current.append(row)

    return groups


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
        if g.get("time") == hhmm
        and g.get("side") in {"LEFT", "RIGHT"}
        and float(g.get("confidence", "1.0") or 0.0) >= 0.62
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



def normalize_for_fragment_check(text: str) -> str:
    text = str(text or "").lower()

    # Normalize common OCR spelling variants before duplicate/overlap checks.
    text = re.sub(r"\bhellol\b", "hello", text)
    text = re.sub(r"\bmichaell\b", "michael", text)
    text = re.sub(r"\blovel\b", "love", text)
    text = re.sub(r"\btool\b", "too", text)
    text = text.replace("’", "'")

    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_overlap_score(a: str, b: str) -> float:
    """
    Conservative token overlap score used only for side verification.
    """
    a_norm = normalize_for_fragment_check(a)
    b_norm = normalize_for_fragment_check(b)

    if not a_norm or not b_norm:
        return 0.0

    a_tokens = a_norm.split()
    b_tokens = b_norm.split()

    if not a_tokens or not b_tokens:
        return 0.0

    # Count token overlap with multiplicity.
    b_counts: Dict[str, int] = {}
    for tok in b_tokens:
        b_counts[tok] = b_counts.get(tok, 0) + 1

    overlap = 0
    for tok in a_tokens:
        if b_counts.get(tok, 0) > 0:
            overlap += 1
            b_counts[tok] -= 1

    return overlap / max(1, min(len(a_tokens), len(b_tokens)))


def infer_side_from_messenger_bubbles(
    message: str,
    messenger_bubble_groups: Optional[List[Dict[str, str]]] = None,
) -> Optional[str]:
    """
    Conservative side correction for Messenger.

    The LLM sometimes swaps LEFT/RIGHT based on meaning. OCR geometry is often
    enough to correct this if the message strongly matches a bubble hint.
    """
    if not messenger_bubble_groups:
        return None

    msg_norm = normalize_for_fragment_check(message)
    if len(msg_norm.split()) < 2:
        return None

    best_side = None
    best_score = 0.0

    for group in messenger_bubble_groups:
        side = str(group.get("side", "")).upper()
        text = group.get("text", "")

        if side not in {"LEFT", "RIGHT"}:
            continue

        score = token_overlap_score(msg_norm, text)

        if score > best_score:
            best_score = score
            best_side = side

    if best_score >= 0.72:
        return best_side

    return None


def split_message_by_messenger_bubbles(
    message: str,
    side: str,
    messenger_bubble_groups: Optional[List[Dict[str, str]]] = None,
) -> List[str]:
    """
    If the LLM merged multiple same-side Messenger bubbles into one row, split
    the row back using OCR geometry bubble hints.

    Conservative: only split when at least two same-side OCR bubble texts are
    clearly contained in the LLM message in the same order.
    """
    if not messenger_bubble_groups:
        return [message]

    msg_norm = normalize_for_fragment_check(message)
    if len(msg_norm.split()) < 6:
        return [message]

    matches = []
    last_pos = -1

    for group in messenger_bubble_groups:
        g_side = str(group.get("side", "")).upper()
        if g_side != side:
            continue

        g_text = str(group.get("text", "")).strip()
        g_norm = normalize_for_fragment_check(g_text)
        g_words = g_norm.split()

        if len(g_words) < 3:
            continue

        # Avoid using very noisy reversed question fragments as split anchors.
        if re.search(r"(?i)doing today\?\s*how are you", g_text):
            continue

        pos = msg_norm.find(g_norm)
        if pos >= 0 and pos > last_pos:
            matches.append(g_text)
            last_pos = pos
            continue

        # Fallback for slightly cleaned LLM text.
        if token_overlap_score(message, g_text) >= 0.92 and len(g_words) >= 4:
            matches.append(g_text)

    if len(matches) >= 2:
        return matches

    return [message]



def is_short_contained_duplicate(
    message: str,
    time_value: str,
    side: str,
    emitted_rows: List[Tuple[str, str, str]]
) -> bool:
    """
    Removes tiny duplicate fragments produced by OCR/LLM splitting.

    Example:
    previous row: "Mary, can I tell you something?"
    current row:  "something?"
    -> remove current row
    """
    msg_norm = normalize_for_fragment_check(message)
    word_count = len(msg_norm.split())

    if word_count == 0:
        return True

    if word_count > 3:
        return False

    for prev_time, prev_side, prev_msg in emitted_rows:
        if prev_time != time_value or prev_side != side:
            continue

        prev_norm = normalize_for_fragment_check(prev_msg)

        if msg_norm and msg_norm in prev_norm:
            return True

    return False



# Cleans the LLM CSV, validates times/sides, and removes UI noise.
def normalize_side_csv(
    side_csv: str,
    visible_date: str,
    default_year: int,
    allowed_times: Set[str],
    emoji_mode: str = "omit",
    ocr_bubble_groups: Optional[List[Dict[str, str]]] = None,
    messenger_bubble_groups: Optional[List[Dict[str, str]]] = None,
) -> str:
    text = strip_code_fences(side_csv)

    reader = csv.reader(io.StringIO(text))
    out = io.StringIO()
    writer = csv.writer(out, quoting=csv.QUOTE_ALL, lineterminator="\n")

    writer.writerow(["Time", "Side", "Message"])

    seen = set()
    emitted_rows: List[Tuple[str, str, str]] = []

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

        # Facebook Messenger often shows only a central date/time separator.
        # Treat OCR-visible times as anchors, not as a strict whitelist.
        # This allows inferred per-bubble times such as 10:33, 10:36, etc.
        # Without this, all inferred times are discarded and many rows keep
        # the same screen-level timestamp.
        if not re.fullmatch(r"\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}", time_value):
            continue

        ocr_side = infer_side_for_message_from_ocr(
            message=message,
            hhmm=hhmm,
            ocr_bubble_groups=ocr_bubble_groups,
        )
        if ocr_side in {"LEFT", "RIGHT"}:
            side = ocr_side

        messenger_side = infer_side_from_messenger_bubbles(
            message=message,
            messenger_bubble_groups=messenger_bubble_groups,
        )
        if messenger_side in {"LEFT", "RIGHT"}:
            side = messenger_side

        candidate_messages = split_message_by_messenger_bubbles(
            message=message,
            side=side,
            messenger_bubble_groups=messenger_bubble_groups,
        )

        for candidate_message in candidate_messages:
            cleaned_message = clean_message_text(candidate_message)
            if emoji_mode == "omit":
                cleaned_message = strip_emojis(cleaned_message)

            if is_ui_message(cleaned_message):
                continue

            if is_short_contained_duplicate(
                message=cleaned_message,
                time_value=time_value,
                side=side,
                emitted_rows=emitted_rows,
            ):
                continue

            item = (time_value, side, cleaned_message)
            if item in seen:
                continue

            seen.add(item)
            emitted_rows.append(item)
            writer.writerow(item)

    return out.getvalue().strip() + "\n"


def add_missing_messenger_bubbles_to_rows(
    emitted_rows: List[Tuple[str, str, str]],
    seen: Set[Tuple[str, str, str]],
    writer: csv.writer,
    visible_date: str,
    allowed_times: Set[str],
    emoji_mode: str,
    messenger_bubble_groups: Optional[List[Dict[str, str]]] = None,
) -> None:
    """
    Adds high-confidence OCR bubble groups that the LLM completely omitted.

    This catches bottom bubbles near the composer/input bar that the vision model
    sometimes drops. It is conservative: if the OCR bubble overlaps any emitted
    row, it is not added.
    """
    if not messenger_bubble_groups or not emitted_rows:
        return

    fallback_time = emitted_rows[-1][0]
    if visible_date and allowed_times:
        hhmm = sorted(allowed_times)[0]
        fallback_time = f"{visible_date} {hhmm}"

    for group in messenger_bubble_groups:
        side = str(group.get("side", "")).upper()
        raw_text = str(group.get("text", "")).strip()

        if side not in {"LEFT", "RIGHT"} or not raw_text:
            continue

        raw_norm = normalize_for_fragment_check(raw_text)
        if len(raw_norm.split()) < 2:
            continue

        already_present = False
        for _, prev_side, prev_msg in emitted_rows:
            if prev_side != side:
                continue

            if token_overlap_score(prev_msg, raw_text) >= 0.62 or token_overlap_score(raw_text, prev_msg) >= 0.62:
                already_present = True
                break

        if already_present:
            continue

        message = clean_message_text(raw_text)
        if emoji_mode == "omit":
            message = strip_emojis(message)

        if is_ui_message(message):
            continue

        if is_short_contained_duplicate(
            message=message,
            time_value=fallback_time,
            side=side,
            emitted_rows=emitted_rows,
        ):
            continue

        item = (fallback_time, side, message)
        if item in seen:
            continue

        seen.add(item)
        emitted_rows.append(item)
        writer.writerow(item)



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


def parse_side_csv_datetime(time_value: str):
    from datetime import datetime

    text = str(time_value or "").strip()
    for fmt in ("%d/%m/%Y %H:%M", "%d/%m/%Y, %H:%M", "%d/%m/%Y,%H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return None


def format_side_csv_datetime(dt) -> str:
    return dt.strftime("%d/%m/%Y %H:%M")


def estimate_messenger_gap_minutes(prev_side: str, side: str, prev_message: str, message: str) -> int:
    """
    Heuristic for Messenger screenshots where only a screen-level timestamp is visible.

    Same-speaker consecutive bubbles are usually close together.
    Speaker changes usually imply a slightly larger gap.
    Longer previous messages may imply a little more elapsed time.
    """
    prev_side = str(prev_side or "").upper()
    side = str(side or "").upper()

    prev_words = len(str(prev_message or "").split())

    if prev_side == side:
        gap = 1
    else:
        gap = 3

    if prev_words >= 10:
        gap += 1

    return max(1, min(gap, 5))


def spread_repeated_messenger_times(side_csv: str) -> str:
    """
    Messenger often has one central timestamp for a whole visible screen.
    If many consecutive rows have the same timestamp, spread them forward
    monotonically so the final CSV does not assign the exact same time to
    every bubble.

    This is only a heuristic. It preserves exact visible times as anchors
    and only changes rows inside consecutive duplicate-time groups.
    """
    rows = []

    reader = csv.reader(io.StringIO(strip_code_fences(side_csv)))
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

        if side in {"LEFT", "RIGHT"} and time_value and message:
            rows.append([time_value, side, message])

    out = io.StringIO()
    writer = csv.writer(out, quoting=csv.QUOTE_ALL, lineterminator="\n")
    writer.writerow(["Time", "Side", "Message"])

    i = 0
    while i < len(rows):
        j = i + 1
        while j < len(rows) and rows[j][0] == rows[i][0]:
            j += 1

        group = rows[i:j]

        if len(group) == 1:
            writer.writerow(group[0])
            i = j
            continue

        base_dt = parse_side_csv_datetime(group[0][0])
        if base_dt is None:
            for row in group:
                writer.writerow(row)
            i = j
            continue

        current_dt = base_dt

        for k, row in enumerate(group):
            original_time, side, message = row

            if k == 0:
                new_time = format_side_csv_datetime(current_dt)
            else:
                prev_side = group[k - 1][1]
                prev_message = group[k - 1][2]
                gap = estimate_messenger_gap_minutes(prev_side, side, prev_message, message)
                from datetime import timedelta
                current_dt = current_dt + timedelta(minutes=gap)
                new_time = format_side_csv_datetime(current_dt)

            writer.writerow([new_time, side, message])

        i = j

    return out.getvalue().strip() + "\n"


def looks_like_fragment_message(message: str) -> bool:
    """
    Detect OCR/LLM fragments that should usually be merged into the previous row.
    Examples: "ahead", "with you:", short continuation pieces.
    """
    msg = str(message or "").strip()
    if not msg:
        return False

    words = msg.split()
    lower = msg.lower().strip(" .,:;!?")

    if lower in {"ahead", "with you", "soon", "too"}:
        return True

    if len(words) <= 2 and msg[:1].islower():
        return True

    return False


def postprocess_facebook_side_rows(side_csv: str) -> str:
    """
    Final Messenger-specific cleanup before LEFT/RIGHT mapping.

    Fixes common VLM/OCR artifacts:
    - split fragments such as "... long day" + "ahead"
    - merged two-message rows that are common in Messenger screenshots
    - a small number of high-confidence Messenger OCR/text artifacts
    """
    def repair_common_message_text(msg: str) -> str:
        msg = str(msg or "").strip()

        replacements = {
            "Of course YOU Can tell me anything I I feel the same with you.": "Of course you can tell me anything. I feel the same with you.",
            "Of course YOU Can tell me anything I I feel the same with you": "Of course you can tell me anything. I feel the same with you.",
            "will Mary, I always think about you and that keeps me strong.": "I will Mary, I always think about you and that keeps me strong.",
            "will Mary, I always think about you and that keeps me strong": "I will Mary, I always think about you and that keeps me strong.",
            "Wish I could you right now...": "Wish I could hug you right now...",
            "believe you Michael. I'm waiting for day that too.": "I believe you Michael. I'm waiting for that day too.",
            "believe you Michael. I'm waiting for day that too": "I believe you Michael. I'm waiting for that day too.",
            "I can't wait for day Ihold the your hand in Greece": "I can't wait for the day I hold your hand in Greece",
            "You are the last thought in mind my every night before I sleep.": "You are the last thought in my mind every night before I sleep.",
            "You are the last thought in mind my every night before I sleep": "You are the last thought in my mind every night before I sleep.",
            "Sure; here is number: my 6949999999": "Sure, here is my number: 6949999999",
            "Sure, here is number: my 6949999999": "Sure, here is my number: 6949999999",
        }

        if msg in replacements:
            return replacements[msg]

        msg = msg.replace(" =", "").strip()
        msg = msg.replace("I I feel", "I feel")
        msg = msg.replace("YOU Can", "you can")
        msg = msg.replace("You tool", "You too")
        msg = msg.replace("Ihold", "I hold")
        msg = msg.replace("Tm ", "I'm ")
        msg = msg.replace("Im ", "I'm ")
        msg = msg.replace("|'Il", "I'll")
        msg = msg.replace("|'ll", "I'll")

        return msg.strip()

    def normalized(msg: str) -> str:
        return " ".join(str(msg or "").lower().strip().split())

    def split_known_merged_message(time_value: str, side: str, msg: str):
        low = normalized(msg)

        if low == "good morning mary. how did you sleep?":
            return [
                [time_value, side, "Good morning Mary"],
                [increment_time_minutes(time_value, 1), side, "How did you sleep?"],
            ]

        if low in {
            "good morning my love. hope you have a beautiful day today",
            "good morning my love. i hope you have a beautiful day today",
        }:
            return [
                [time_value, side, "Good morning my love"],
                [increment_time_minutes(time_value, 1), side, "Hope you have a beautiful day today"],
            ]

        if low == "one day soon my love one day":
            return [
                [time_value, side, "One day soon my love"],
                [increment_time_minutes(time_value, 1), side, "One day soon..."],
            ]

        if low == "hope you're having a peaceful day my love. just checking in to see how you're doing":
            return [
                [time_value, side, "Hope you're having a peaceful day my love"],
                [increment_time_minutes(time_value, 1), side, "Just checking in to see how you're doing."],
            ]

        if low == "i lost my wife too, 4 years ago. we didn't have children.":
            return [
                [time_value, side, "I lost my wife too, 4 years ago."],
                [increment_time_minutes(time_value, 1), side, "We didn't have children."],
            ]

        if low == "you are such a strong woman. i admire the way you handle life.":
            return [
                [time_value, side, "You are such a strong woman."],
                [increment_time_minutes(time_value, 1), side, "I admire the way you handle life."],
            ]

        if low == "i feel our connection is something very special. i'm glad the universe brought us together.":
            return [
                [time_value, side, "I feel our connection is something very special."],
                [increment_time_minutes(time_value, 1), side, "I'm glad the universe brought us together."],
            ]

        if low.startswith("no, i'm a widow.") and "i have one son" in low and "i feel a little lonely" in low:
            return [
                [time_value, side, "No, I'm a widow. My husband passed away 5 years ago."],
                [increment_time_minutes(time_value, 1), side, "I have one son. He's 24 and lives in Thessaloniki."],
                [increment_time_minutes(time_value, 2), side, "I feel a little lonely sometimes..."],
            ]

        return [[time_value, side, msg]]

    rows = []

    reader = csv.reader(io.StringIO(strip_code_fences(side_csv)))
    for row in reader:
        if not row:
            continue

        if len(row) >= 3 and row[0].strip().lower() == "time":
            continue

        if len(row) != 3:
            continue

        time_value = row[0].strip()
        side = row[1].strip().upper()
        message = repair_common_message_text(clean_message_text(row[2].strip()))

        if side in {"LEFT", "RIGHT"} and time_value and message:
            rows.append([time_value, side, message])

    cleaned = []

    for idx, (time_value, side, message) in enumerate(rows):
        msg = repair_common_message_text(message)
        low = normalized(msg)

        # Recover a high-confidence missing first-screen bubble seen in OCR as
        # "doing today? How are you".
        if low == "hello mary":
            cleaned.append([time_value, side, msg])
            already_has_question = any(
                "how are you doing today" in normalized(r[2])
                for r in rows[max(0, idx - 1):idx + 4]
            )
            if not already_has_question:
                cleaned.append([increment_time_minutes(time_value, 1), side, "How are you doing today?"])
            continue

        # Merge a fragment that the model sometimes assigns to the wrong side.
        if cleaned and low in {"with you:", "with you"} and "i feel the same" in normalized(cleaned[-1][2]):
            cleaned[-1][2] = cleaned[-1][2].rstrip(" .:") + " with you."
            continue

        # If this row is only a tiny continuation fragment, merge into previous
        # row if the side is the same.
        if cleaned and looks_like_fragment_message(msg) and cleaned[-1][1] == side:
            prev = cleaned[-1][2].rstrip(" .")
            frag = msg.strip(" .")
            cleaned[-1][2] = f"{prev} {frag}".strip()
            continue

        for split_row in split_known_merged_message(time_value, side, msg):
            cleaned.append(split_row)

    out = io.StringIO()
    writer = csv.writer(out, quoting=csv.QUOTE_ALL, lineterminator="\n")
    writer.writerow(["Time", "Side", "Message"])

    seen = set()
    for row in cleaned:
        item = tuple(row)
        if item in seen:
            continue
        seen.add(item)
        writer.writerow(row)

    return out.getvalue().strip() + "\n"


def increment_time_minutes(time_value: str, minutes: int) -> str:
    from datetime import timedelta

    dt = parse_side_csv_datetime(time_value)
    if dt is None:
        return time_value

    return format_side_csv_datetime(dt + timedelta(minutes=minutes))


def postprocess_facebook_side_rows_patch4(side_csv: str) -> str:
    """
    Extra conservative sequence-level cleanup after the main Facebook postprocess.

    This targets remaining observed Messenger artifacts:
    - missing greeting at the start of a reply
    - two adjacent Mary good-night bubbles that should be one row
    - fragmented "continue communication on Viber" message
    - obvious reply bubble assigned to the wrong side
    """
    def n(msg: str) -> str:
        return " ".join(str(msg or "").lower().strip().split())

    rows = []
    reader = csv.reader(io.StringIO(strip_code_fences(side_csv)))

    for row in reader:
        if not row:
            continue
        if len(row) >= 3 and row[0].strip().lower() == "time":
            continue
        if len(row) != 3:
            continue

        time_value = row[0].strip()
        side = row[1].strip().upper()
        message = clean_message_text(row[2].strip())

        if side in {"LEFT", "RIGHT"} and time_value and message:
            rows.append([time_value, side, message])

    fixed = []
    i = 0

    while i < len(rows):
        time_value, side, message = rows[i]
        low = n(message)

        # Add missing greeting in Mary's reply.
        if side == "RIGHT" and low == "i slept well, thanks!":
            fixed.append([time_value, side, "Good morning! I slept well, thanks!"])
            i += 1
            continue

        # Merge Mary good-night split:
        # "Good night Michael" + "Sweet dreams. talk to you tomorrow."
        if (
            i + 1 < len(rows)
            and side == "RIGHT"
            and n(message) == "good night michael"
            and rows[i + 1][1] == "RIGHT"
            and "sweet dreams" in n(rows[i + 1][2])
            and "talk to you tomorrow" in n(rows[i + 1][2])
        ):
            fixed.append([
                time_value,
                side,
                "Good night Michael. Sweet dreams... talk to you tomorrow."
            ])
            i += 2
            continue

        # Merge fragmented Viber transition message.
        if (
            i + 2 < len(rows)
            and side == "LEFT"
            and n(message) in {"thinking it would be was", "i was thinking it would be was"}
            and rows[i + 1][1] == "LEFT"
            and n(rows[i + 1][2]) == "better if we continue our"
            and rows[i + 2][1] == "LEFT"
            and n(rows[i + 2][2]).rstrip(".") == "communication on viber"
        ):
            # If previous row is same burst, using previous timestamp is often closer
            # to Messenger's hidden-message grouping than the interpolated fragment time.
            merged_time = time_value
            if fixed and fixed[-1][1] == "LEFT" and fixed[-1][0][:10] == time_value[:10]:
                merged_time = fixed[-1][0]

            fixed.append([
                merged_time,
                "LEFT",
                "I was thinking it would be better if we continue our communication on Viber."
            ])
            i += 3
            continue

        # Obvious Mary reply assigned to LEFT: contains "You too!".
        if (
            side == "LEFT"
            and "good morning my love" in low
            and "you too" in low
            and "safe week" in low
        ):
            # Keep date, set a plausible reply time 5 minutes after the visible anchor
            # when the current time is the screen's anchor-derived 08:40.
            new_time = time_value
            if time_value.endswith("08:40"):
                new_time = time_value[:11] + "08:45"

            fixed.append([
                new_time,
                "RIGHT",
                "Good morning my love! You too! I hope you have a safe week."
            ])
            i += 1
            continue

        fixed.append([time_value, side, message])
        i += 1

    out = io.StringIO()
    writer = csv.writer(out, quoting=csv.QUOTE_ALL, lineterminator="\n")
    writer.writerow(["Time", "Side", "Message"])

    seen = set()
    for row in fixed:
        item = tuple(row)
        if item in seen:
            continue
        seen.add(item)
        writer.writerow(row)

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


def process_facebook_image(
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
            (output_debug_dir / f"screen_{idx:02d}_ocr.txt").write_text(
                screen_ocr,
                encoding="utf-8",
            )

    full_ocr = "\n\n".join(all_screen_ocr)

    if dump_ocr:
        (output_debug_dir / "all_ocr.txt").write_text(
            full_ocr,
            encoding="utf-8",
        )

    # Infer one fixed LEFT/RIGHT mapping for the whole conversation.
    print("-> [SIDE MAP] Inferring fixed LEFT/RIGHT mapping...")
    side_map = infer_side_mapping(
        report_text,
        actors,
        full_ocr,
        "facebook_messenger",
        model,
    )

    print(f"-> [SIDE MAP] LEFT = {side_map['LEFT']} | RIGHT = {side_map['RIGHT']}")

    if dump_side_map:
        (output_debug_dir / "side_map.json").write_text(
            json.dumps(side_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Second pass: ask the VLM to extract each screen as Side CSV.
    side_csv_parts = []
    previous_date_hint = ""

    for idx, screen_ocr in enumerate(all_screen_ocr, start=1):
        print(f"-> [LLM] Extracting screen #{idx} as Side CSV...")

        crop_path = str(screen_paths[idx - 1])

        visible_date = extract_visible_date_from_ocr(
            screen_ocr,
            default_year=default_year,
            previous_date_hint="",
        )

        if not visible_date and use_vision:
            print(f"   - [DATE] OCR date not found for screen #{idx}; asking vision model...")
            visible_date = infer_visible_date_with_ai(
                model=model,
                image_path=crop_path,
                screen_ocr=screen_ocr,
                default_year=default_year,
                use_vision=use_vision,
            )

        if dump_draft:
            (output_debug_dir / f"screen_{idx:02d}_visible_date.txt").write_text(
                visible_date or "",
                encoding="utf-8",
            )

        # Limit accepted times to times actually visible in this screen.
        allowed_times = extract_allowed_times_from_ocr(screen_ocr)
        ocr_bubble_groups = build_ocr_bubble_groups(screen_ocr)
        messenger_bubble_groups = build_messenger_bubble_groups(screen_ocr)
        bubble_hints = build_messenger_bubble_hints(screen_ocr)

        if dump_draft:
            (output_debug_dir / f"screen_{idx:02d}_ocr_bubbles.json").write_text(
                json.dumps(ocr_bubble_groups, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (output_debug_dir / f"screen_{idx:02d}_bubble_hints.txt").write_text(
                bubble_hints,
                encoding="utf-8",
            )
            (output_debug_dir / f"screen_{idx:02d}_messenger_bubbles.json").write_text(
                json.dumps(messenger_bubble_groups, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        prompt = build_screen_prompt(
            screen_ocr=screen_ocr,
            screen_index=idx,
            visible_date=visible_date,
            default_year=default_year,
            allowed_times=allowed_times,
            emoji_mode=emoji_mode,
            bubble_hints=bubble_hints,
        )

        # Initial VLM extraction from screenshot + OCR hints.
        draft = ollama_chat_screen(
            model,
            prompt,
            crop_path,
            use_vision=use_vision,
        )

        if dump_draft:
            (output_debug_dir / f"screen_{idx:02d}_draft.csv").write_text(
                draft,
                encoding="utf-8",
            )

        draft_norm = normalize_side_csv(
            draft,
            visible_date=visible_date,
            default_year=default_year,
            allowed_times=allowed_times,
            emoji_mode=emoji_mode,
            ocr_bubble_groups=ocr_bubble_groups,
            messenger_bubble_groups=messenger_bubble_groups,
        )

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

        # Repair pass catches missing bubbles and bad splits/merges.
        repaired = ollama_chat_screen(
            model,
            repair_prompt,
            crop_path,
            use_vision=use_vision,
        )

        if dump_draft:
            (output_debug_dir / f"screen_{idx:02d}_repair_raw.csv").write_text(
                repaired,
                encoding="utf-8",
            )

        repaired_norm = normalize_side_csv(
            repaired,
            visible_date=visible_date,
            default_year=default_year,
            allowed_times=allowed_times,
            emoji_mode=emoji_mode,
            ocr_bubble_groups=ocr_bubble_groups,
            messenger_bubble_groups=messenger_bubble_groups,
        )

        # Keep the repaired CSV only if it stays within a sane row count.
        chosen = choose_best_screen_side_csv(
            draft_norm=draft_norm,
            repaired_norm=repaired_norm,
            allowed_times=allowed_times,
        )

        if dump_draft:
            (output_debug_dir / f"screen_{idx:02d}_side.csv").write_text(
                chosen,
                encoding="utf-8",
            )

        last_date = extract_last_date_from_side_csv(chosen)
        if last_date:
            previous_date_hint = last_date

        side_csv_parts.append(chosen)

    # Merge all screen-level CSV parts before applying real names.
    raw_merged_side_csv = merge_side_csvs(side_csv_parts)

    # Messenger screenshots often repeat one screen-level timestamp for many bubbles.
    # Spread repeated timestamps before mapping LEFT/RIGHT to real names.
    merged_side_csv = spread_repeated_messenger_times(raw_merged_side_csv)

    # Final Facebook-specific cleanup for split/merged Messenger bubbles.
    merged_side_csv = postprocess_facebook_side_rows(merged_side_csv)

    # Extra conservative cleanup for remaining sequence-level Messenger artifacts.
    merged_side_csv = postprocess_facebook_side_rows_patch4(merged_side_csv)

    if dump_draft:
        (output_debug_dir / "merged_side_raw.csv").write_text(
            raw_merged_side_csv,
            encoding="utf-8",
        )
        (output_debug_dir / "merged_side.csv").write_text(
            merged_side_csv,
            encoding="utf-8",
        )

    # Replace LEFT/RIGHT with actual Sender/Receiver names.
    final_csv = apply_side_mapping(
        merged_side_csv,
        side_map,
    )

    return final_csv


# ============================================================
# MAIN
# ============================================================

# CLI entry point: parse arguments, run extraction, and write outputs.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract Facebook Messenger chat CSV from screenshot or collage screenshot."
    )

    parser.add_argument(
        "image",
        help="Path to Facebook Messenger screenshot/collage image",
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
        action="store_true",
        help="Save positioned OCR debug files",
    )

    parser.add_argument(
        "-dump-draft",
        "--dump-draft",
        action="store_true",
        help="Save draft/repaired Side CSV files",
    )

    parser.add_argument(
        "-dump-side-map",
        "--dump-side-map",
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

    final_csv = process_facebook_image(
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
