"""Optional OCR helpers (EasyOCR). Fail soft when dependency missing."""

from __future__ import annotations

from typing import Any

import numpy as np


_READER = None


def _get_reader():
    global _READER
    if _READER is not None:
        return _READER
    try:
        import easyocr

        _READER = easyocr.Reader(["ch_sim", "en"], gpu=False, verbose=False)
        return _READER
    except Exception:
        return None


def run_ocr(
    image_bgr: np.ndarray,
    *,
    zones: list[tuple[str, int, int, int, int]] | None = None,
) -> list[dict[str, Any]]:
    """Return [{text, conf, x, y, zone}]. Coordinates in full-image pixels."""
    reader = _get_reader()
    if reader is None:
        return []
    h, w = image_bgr.shape[:2]
    if zones is None:
        zones = [
            ("top", 0, 0, w, max(40, h // 14)),
            ("right", int(w * 0.82), 0, w, h),
            ("left", 0, 0, max(40, w // 12), h),
            ("interior", int(w * 0.2), int(h * 0.05), int(w * 0.85), int(h * 0.95)),
        ]
    items: list[dict[str, Any]] = []
    for name, x0, y0, x1, y1 in zones:
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 - x0 < 10 or y1 - y0 < 10:
            continue
        crop = image_bgr[y0:y1, x0:x1]
        try:
            results = reader.readtext(crop, detail=1, paragraph=False)
        except Exception:
            continue
        for box, text, conf in results:
            xs = [p[0] + x0 for p in box]
            ys = [p[1] + y0 for p in box]
            items.append(
                {
                    "text": str(text).strip(),
                    "conf": float(conf),
                    "x": float((min(xs) + max(xs)) / 2),
                    "y": float((min(ys) + max(ys)) / 2),
                    "zone": name,
                }
            )
    return items
