"""Extract RoomPlan from PDF / JPG CAD floorplans.

Pipeline:
  1. Load PDF page(s) via pypdfium2 → RGB, or load JPG/PNG via OpenCV.
  2. Edge / line detection (Canny + Hough) → candidate wall segments.
  3. Detect shorter gaps / perpendicular stubs as openings (doors/windows).
  4. Optional: OpenAI-compatible vision API to refine labels & merge segments.
  5. Emit RoomPlan JSON + write control maps via control.py.
"""

from __future__ import annotations

import base64
import json
import math
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cad_ai_lock.config import get_settings
from cad_ai_lock.geometry import Opening, OpeningType, Point2D, RoomPlan, Wall


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def load_image(path: str | Path, pdf_page: int = 0, pdf_scale: float = 2.0) -> np.ndarray:
    """Load JPG/PNG/PDF → BGR uint8 image."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(str(path))
        if pdf_page < 0 or pdf_page >= len(doc):
            raise IndexError(f"PDF page {pdf_page} out of range (0..{len(doc) - 1})")
        page = doc[pdf_page]
        bitmap = page.render(scale=pdf_scale)
        pil = bitmap.to_pil()
        rgb = np.array(pil.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def _merge_collinear(
    segments: list[tuple[float, float, float, float]],
    angle_tol_deg: float = 8.0,
    dist_tol: float = 15.0,
    gap_tol: float = 25.0,
) -> list[tuple[float, float, float, float]]:
    """Greedy merge of near-collinear Hough segments into longer walls."""
    if not segments:
        return []

    def angle(s: tuple[float, float, float, float]) -> float:
        return math.degrees(math.atan2(s[3] - s[1], s[2] - s[0])) % 180

    def mid(s: tuple[float, float, float, float]) -> tuple[float, float]:
        return ((s[0] + s[2]) / 2, (s[1] + s[3]) / 2)

    def length(s: tuple[float, float, float, float]) -> float:
        return math.hypot(s[2] - s[0], s[3] - s[1])

    segs = sorted(segments, key=length, reverse=True)
    used = [False] * len(segs)
    merged: list[tuple[float, float, float, float]] = []

    for i, s in enumerate(segs):
        if used[i]:
            continue
        used[i] = True
        a = angle(s)
        points = [(s[0], s[1]), (s[2], s[3])]
        changed = True
        while changed:
            changed = False
            for j, t in enumerate(segs):
                if used[j]:
                    continue
                if abs(((angle(t) - a + 90) % 180) - 90) > angle_tol_deg:
                    continue
                # distance between midpoints projected roughly
                mx, my = mid(t)
                # point-line distance to s axis
                x1, y1, x2, y2 = points[0][0], points[0][1], points[-1][0], points[-1][1]
                denom = math.hypot(x2 - x1, y2 - y1) or 1.0
                dist = abs((y2 - y1) * mx - (x2 - x1) * my + x2 * y1 - y2 * x1) / denom
                if dist > dist_tol:
                    continue
                # gap between endpoint clouds
                all_pts = points + [(t[0], t[1]), (t[2], t[3])]
                min_gap = min(
                    math.hypot(p[0] - q[0], p[1] - q[1])
                    for p in points
                    for q in [(t[0], t[1]), (t[2], t[3])]
                )
                if min_gap > gap_tol and length(t) < 40:
                    continue
                if min_gap > gap_tol * 2:
                    continue
                used[j] = True
                points.extend([(t[0], t[1]), (t[2], t[3])])
                changed = True

        # PCA-ish: extreme points along dominant direction
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        # use angle a
        rad = math.radians(a)
        ux, uy = math.cos(rad), math.sin(rad)
        projs = [p[0] * ux + p[1] * uy for p in points]
        i_min, i_max = int(np.argmin(projs)), int(np.argmax(projs))
        merged.append((points[i_min][0], points[i_min][1], points[i_max][0], points[i_max][1]))

    return merged


def _structure_mask(gray: np.ndarray) -> np.ndarray:
    """Dark CAD ink ∪ Canny near mid-tone cores — robust for Chinese scan plans."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    ink = (blur < 120).astype(np.uint8) * 255
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    canny = cv2.Canny(blur, 40, 120)
    near = cv2.dilate((blur < 165).astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
    combo = cv2.bitwise_or(ink, cv2.bitwise_and(canny, near))
    h, w = gray.shape[:2]
    combo[: max(12, h // 50), :] = 0
    combo[-max(10, h // 70) :, :] = 0
    combo[:, : max(10, w // 60)] = 0
    combo[:, -max(40, w // 14) :] = 0
    return combo


def _snap_ortho_seg(
    x1: float, y1: float, x2: float, y2: float, tol_deg: float = 14.0
) -> tuple[float, float, float, float] | None:
    ang = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
    d0 = min(abs(ang - 0), abs(ang - 180))
    d90 = abs(ang - 90)
    if d0 <= tol_deg and d0 <= d90:
        y = (y1 + y2) / 2
        return (x1, y, x2, y)
    if d90 <= tol_deg:
        x = (x1 + x2) / 2
        return (x, y1, x, y2)
    return None


def detect_walls_and_openings(
    image_bgr: np.ndarray,
    min_line_length: int = 40,
    max_line_gap: int = 12,
    canny1: int = 50,
    canny2: int = 150,
) -> tuple[list[Wall], list[Opening], np.ndarray]:
    """Hybrid structure extract: ink/Canny mask + Hough + LSD → ortho walls.

    Prefer long orthogonal segments on the structure mask; short residuals
    become opening candidates. Falls back to image border if empty.
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    structure = _structure_mask(gray)
    edges = structure.copy()

    raw: list[tuple[float, float, float, float]] = []

    # Hough on structure mask
    lines = cv2.HoughLinesP(
        structure,
        rho=1,
        theta=np.pi / 180,
        threshold=max(30, min(w, h) // 45),
        minLineLength=max(28, min_line_length - 10),
        maxLineGap=max(8, max_line_gap),
    )
    if lines is not None:
        for line in lines:
            vals = line.reshape(-1)[:4]
            x1, y1, x2, y2 = map(float, vals)
            if math.hypot(x2 - x1, y2 - y1) >= min_line_length * 0.7:
                snapped = _snap_ortho_seg(x1, y1, x2, y2)
                if snapped:
                    raw.append(snapped)

    # LSD supplement (often recovers long exterior walls)
    try:
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        lsd_lines = lsd.detect(blur)[0]
        if lsd_lines is not None:
            for L in lsd_lines:
                x1, y1, x2, y2 = map(float, np.asarray(L).reshape(-1)[:4])
                ln = math.hypot(x2 - x1, y2 - y1)
                if ln < 40:
                    continue
                mx, my = int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))
                if not (0 <= mx < w and 0 <= my < h):
                    continue
                if structure[
                    max(0, my - 2) : my + 3, max(0, mx - 2) : mx + 3
                ].max() == 0:
                    continue
                snapped = _snap_ortho_seg(x1, y1, x2, y2)
                if snapped:
                    raw.append(snapped)
    except Exception:
        pass

    # legacy Canny path if still sparse
    if len(raw) < 4:
        canny = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), canny1, canny2)
        canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1)
        edges = canny
        lines = cv2.HoughLinesP(
            canny,
            rho=1,
            theta=np.pi / 180,
            threshold=max(40, min(w, h) // 40),
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )
        if lines is not None:
            for line in lines:
                vals = line.reshape(-1)[:4]
                x1, y1, x2, y2 = map(float, vals)
                if math.hypot(x2 - x1, y2 - y1) >= min_line_length * 0.8:
                    raw.append((x1, y1, x2, y2))

    merged = _merge_collinear(raw)
    lengths = [math.hypot(s[2] - s[0], s[3] - s[1]) for s in merged]
    if lengths:
        thr = max(min_line_length * 1.2, float(np.percentile(lengths, 35)))
    else:
        thr = float(min_line_length)

    walls: list[Wall] = []
    openings: list[Opening] = []
    for s, L in zip(merged, lengths or [0] * len(merged)):
        x1, y1, x2, y2 = s
        if L >= thr:
            walls.append(
                Wall(
                    id=_new_id("wall"),
                    start=Point2D(x=x1, y=y1),
                    end=Point2D(x=x2, y=y2),
                    thickness_px=8.0,
                )
            )
        elif L >= 18:
            openings.append(
                Opening(
                    id=_new_id("open"),
                    type=OpeningType.DOOR if L < thr * 0.65 else OpeningType.WINDOW,
                    start=Point2D(x=x1, y=y1),
                    end=Point2D(x=x2, y=y2),
                )
            )

    if not walls:
        margin = 20
        corners = [
            (margin, margin, w - margin, margin),
            (w - margin, margin, w - margin, h - margin),
            (w - margin, h - margin, margin, h - margin),
            (margin, h - margin, margin, margin),
        ]
        for x1, y1, x2, y2 in corners:
            walls.append(
                Wall(
                    id=_new_id("wall"),
                    start=Point2D(x=float(x1), y=float(y1)),
                    end=Point2D(x=float(x2), y=float(y2)),
                    thickness_px=10.0,
                    exterior=True,
                )
            )

    return walls, openings, edges


def refine_with_vision(
    image_bgr: np.ndarray,
    plan: RoomPlan,
) -> RoomPlan:
    """Optional LLM vision refine via OpenAI-compatible chat completions."""
    settings = get_settings()
    if not settings.openai_api_key:
        return plan

    import httpx

    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return plan
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    schema_hint = {
        "walls": [{"id": "wall_1", "start": {"x": 0, "y": 0}, "end": {"x": 100, "y": 0}}],
        "openings": [
            {
                "id": "door_1",
                "type": "door",
                "start": {"x": 40, "y": 0},
                "end": {"x": 60, "y": 0},
            }
        ],
        "notes": "string",
    }
    prompt = (
        "You are a floorplan CAD parser. Given this floorplan image and a draft JSON "
        "of walls/openings in pixel coordinates, refine wall endpoints and label "
        "doors vs windows. Return ONLY valid JSON matching this shape:\n"
        f"{json.dumps(schema_hint)}\n"
        f"Image size: {plan.image_width}x{plan.image_height}. Draft:\n"
        f"{json.dumps(plan.to_json_dict(), ensure_ascii=False)[:6000]}"
    )

    url = settings.openai_base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {
        "model": settings.openai_vision_model,
        "temperature": 0.1,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ],
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        # strip markdown fences if present
        text = content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
        data = json.loads(text)
        walls = [Wall.model_validate(w) for w in data.get("walls", [])]
        openings = [Opening.model_validate(o) for o in data.get("openings", [])]
        if walls:
            plan.walls = walls
        if openings:
            plan.openings = openings
        if data.get("notes"):
            plan.notes = str(data["notes"])
        plan.meta["vision_refined"] = True
    except Exception as exc:  # noqa: BLE001 — optional path must not break CLI
        plan.meta["vision_error"] = str(exc)

    return plan


def extract_room_plan(
    path: str | Path,
    *,
    pdf_page: int = 0,
    use_vision: bool = False,
    scale_px_per_m: float | None = None,
    auto_correct: bool = True,
    use_ocr: bool = True,
) -> tuple[RoomPlan, np.ndarray, np.ndarray]:
    """Main entry: path → (RoomPlan, BGR image, edge/structure map).

    When ``auto_correct`` is True (commercial default), runs assess→repair
    until topology health passes or repair budget is exhausted. Never requires
    hand-edited plan.json for the happy path.
    """
    path = Path(path)
    image = load_image(path, pdf_page=pdf_page)
    h, w = image.shape[:2]
    walls, openings, edges = detect_walls_and_openings(image)

    plan = RoomPlan(
        source_path=str(path),
        image_width=w,
        image_height=h,
        scale_px_per_m=scale_px_per_m,
        walls=walls,
        openings=openings,
        plan_revision="extract-0",
        meta={
            "extractor": "hybrid_ink_hough_lsd",
            "pdf_page": pdf_page,
            "auto_correct": auto_correct,
        },
    )

    ocr_items: list[dict[str, Any]] = []
    if use_ocr:
        try:
            from cad_ai_lock.ocr_utils import run_ocr

            ocr_items = run_ocr(image)
            plan.meta["ocr_items"] = ocr_items
            plan.meta["ocr_count"] = len(ocr_items)
        except Exception as exc:  # noqa: BLE001
            plan.meta["ocr_error"] = str(exc)

    if use_vision:
        plan = refine_with_vision(image, plan)

    if auto_correct:
        from cad_ai_lock.assess import assess_plan
        from cad_ai_lock.repair import auto_repair, mark_exterior

        plan = mark_exterior(plan)
        health = assess_plan(plan)
        plan, health, applied = auto_repair(
            plan,
            health,
            image_bgr=image,
            ocr_items=ocr_items,
        )
        plan.meta["extract_health"] = health.to_dict()
        plan.meta["extract_repairs"] = applied
    elif ocr_items and scale_px_per_m is None:
        from cad_ai_lock.repair import (
            estimate_scale_from_ocr,
            infer_rooms,
            label_rooms_from_ocr,
            mark_exterior,
        )

        plan = mark_exterior(plan)
        plan = infer_rooms(plan, image)
        plan = label_rooms_from_ocr(plan, ocr_items)
        plan = estimate_scale_from_ocr(plan, ocr_items)

    return plan, image, edges


def make_synthetic_floorplan(
    out_path: str | Path,
    width: int = 640,
    height: int = 480,
) -> Path:
    """Draw a simple rectangle room with door + window for demos / smoke tests."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = np.ones((height, width, 3), dtype=np.uint8) * 245
    # outer walls
    margin = 60
    pt1 = (margin, margin)
    pt2 = (width - margin, height - margin)
    cv2.rectangle(img, pt1, pt2, (30, 30, 30), 6)
    # interior partition
    mid_x = width // 2
    cv2.line(img, (mid_x, margin), (mid_x, height - margin - 80), (30, 30, 30), 5)
    # door gap on bottom wall (erase then mark)
    door_cx = width // 2 - 80
    cv2.line(
        img,
        (door_cx - 35, height - margin),
        (door_cx + 35, height - margin),
        (245, 245, 245),
        8,
    )
    cv2.line(
        img,
        (door_cx - 35, height - margin),
        (door_cx + 35, height - margin),
        (0, 120, 255),
        3,
    )
    # window on top wall
    win_cx = width // 2 + 100
    cv2.line(
        img,
        (win_cx - 40, margin),
        (win_cx + 40, margin),
        (245, 245, 245),
        8,
    )
    cv2.line(
        img,
        (win_cx - 40, margin),
        (win_cx + 40, margin),
        (255, 180, 0),
        3,
    )
    # dimension-ish labels
    cv2.putText(
        img,
        "CAD FLOORPLAN DEMO",
        (margin, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (80, 80, 80),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(out_path), img)
    return out_path
