"""Build ControlNet conditioning maps from RoomPlan or source image.

Locking strategy (step 2):
  Render walls as thick lines and openings as colored markers onto:
    - canny / lineart control image (structure lock)
    - optional soft mask (inpaint style regions only)
    - optional depth-like gradient (room enclosure cue)

Generation MUST condition on these maps with high control strength so
walls/doors/windows stay fixed while style/furniture change.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cad_ai_lock.geometry import OpeningType, RoomPlan


def render_structure_map(
    plan: RoomPlan,
    *,
    wall_color: tuple[int, int, int] = (255, 255, 255),
    door_color: tuple[int, int, int] = (0, 255, 255),
    window_color: tuple[int, int, int] = (255, 200, 0),
    bg: int = 0,
) -> np.ndarray:
    """RGB structure map: white walls, cyan doors, amber windows on black."""
    canvas = np.full((plan.image_height, plan.image_width, 3), bg, dtype=np.uint8)
    for wall in plan.walls:
        thickness = max(2, int(round(wall.thickness_px)))
        cv2.line(
            canvas,
            wall.start.as_int_tuple(),
            wall.end.as_int_tuple(),
            wall_color,
            thickness,
            lineType=cv2.LINE_AA,
        )
    for op in plan.openings:
        color = door_color if op.type == OpeningType.DOOR else window_color
        if op.type == OpeningType.OPENING:
            color = (180, 180, 255)
        cv2.line(
            canvas,
            op.start.as_int_tuple(),
            op.end.as_int_tuple(),
            color,
            4,
            lineType=cv2.LINE_AA,
        )
        # end caps
        for pt in (op.start.as_int_tuple(), op.end.as_int_tuple()):
            cv2.circle(canvas, pt, 5, color, -1, lineType=cv2.LINE_AA)
    return canvas


def render_canny_from_plan(plan: RoomPlan) -> np.ndarray:
    """Grayscale lineart suitable as ControlNet Canny / Lineart input."""
    structure = render_structure_map(plan)
    gray = cv2.cvtColor(structure, cv2.COLOR_BGR2GRAY)
    # already binary-ish; slight blur + Canny for provider compatibility
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    # keep strong wall strokes
    edges = cv2.bitwise_or(edges, gray)
    return edges


def render_canny_from_image(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(gray, 50, 150)


def render_depth_proxy(plan: RoomPlan) -> np.ndarray:
    """Simple depth-like map: brighter near room center, dark outside walls."""
    h, w = plan.image_height, plan.image_width
    depth = np.zeros((h, w), dtype=np.float32)
    # fill polygon of wall endpoints if possible
    pts: list[list[int]] = []
    for wall in plan.walls:
        pts.append(list(wall.start.as_int_tuple()))
        pts.append(list(wall.end.as_int_tuple()))
    if len(pts) >= 3:
        hull = cv2.convexHull(np.array(pts, dtype=np.int32))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        # distance transform inside
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        if dist.max() > 0:
            depth = dist / dist.max()
    else:
        yy, xx = np.mgrid[0:h, 0:w]
        depth = 1.0 - np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
        depth = np.clip(depth, 0, 1)
    depth_u8 = (depth * 255).astype(np.uint8)
    return cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)


def render_style_mask(plan: RoomPlan, dilate: int = 12) -> np.ndarray:
    """White = free to restyle (interior); black = locked structure band."""
    h, w = plan.image_height, plan.image_width
    mask = np.ones((h, w), dtype=np.uint8) * 255
    structure = render_canny_from_plan(plan)
    # lock a band around walls/openings
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
    band = cv2.dilate(structure, kernel, iterations=1)
    mask[band > 0] = 0
    return mask


def render_room_semantic(plan: RoomPlan) -> np.ndarray:
    """Color rooms differently for semantic / segmentation conditioning."""
    h, w = plan.image_height, plan.image_width
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    palette = [
        (40, 90, 200),
        (40, 180, 120),
        (200, 140, 40),
        (180, 60, 160),
        (80, 80, 80),
    ]
    for i, room in enumerate(plan.rooms or []):
        poly = room.get("polygon") or []
        if len(poly) < 3:
            continue
        pts = np.array([[int(p["x"]), int(p["y"])] for p in poly], dtype=np.int32)
        color = palette[i % len(palette)]
        cv2.fillPoly(canvas, [pts], color)
    # draw walls on top so structure remains visible
    structure = render_structure_map(plan)
    wall = structure.max(axis=2) > 0
    canvas[wall] = structure[wall]
    return canvas


def render_furniture_map(plan: RoomPlan, layout: Any | None) -> np.ndarray:
    """Occupancy sketch of furniture (soft layer) on black background."""
    h, w = plan.image_height, plan.image_width
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if layout is None:
        return canvas
    items = getattr(layout, "items", None) or []
    for it in items:
        cx, cy = int(round(it.center.x)), int(round(it.center.y))
        hw, hd = max(2, int(it.width_px / 2)), max(2, int(it.depth_px / 2))
        # axis-aligned box approx (rotation drawn as arrow)
        cv2.rectangle(
            canvas,
            (cx - hw, cy - hd),
            (cx + hw, cy + hd),
            (220, 220, 220),
            -1,
            lineType=cv2.LINE_AA,
        )
        rad = math.radians(it.rotation_deg)
        ex = int(cx + math.cos(rad) * max(hw, hd))
        ey = int(cy + math.sin(rad) * max(hw, hd))
        cv2.arrowedLine(canvas, (cx, cy), (ex, ey), (0, 200, 255), 2, tipLength=0.35)
        cv2.putText(
            canvas,
            it.type[:6],
            (cx - hw, max(12, cy - hd - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (180, 255, 180),
            1,
            cv2.LINE_AA,
        )
    return canvas


def write_control_bundle(
    plan: RoomPlan,
    out_dir: str | Path,
    source_image: np.ndarray | None = None,
    layout: Any | None = None,
) -> dict[str, str]:
    """Write canny/structure/depth/mask/semantic/furniture/perspective controls."""
    import json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    structure = render_structure_map(plan)
    canny = render_canny_from_plan(plan)
    depth = render_depth_proxy(plan)
    mask = render_style_mask(plan)
    semantic = render_room_semantic(plan)
    furniture = render_furniture_map(plan, layout)

    paths = {
        "structure": str(out_dir / "control_structure.png"),
        "canny": str(out_dir / "control_canny.png"),
        "depth": str(out_dir / "control_depth.png"),
        "mask": str(out_dir / "control_mask.png"),
        "semantic": str(out_dir / "control_semantic.png"),
        "furniture": str(out_dir / "control_furniture.png"),
    }
    cv2.imwrite(paths["structure"], structure)
    cv2.imwrite(paths["canny"], canny)
    cv2.imwrite(paths["depth"], depth)
    cv2.imwrite(paths["mask"], mask)
    cv2.imwrite(paths["semantic"], semantic)
    cv2.imwrite(paths["furniture"], furniture)

    if source_image is not None:
        src_canny = render_canny_from_image(source_image)
        p = str(out_dir / "control_source_canny.png")
        cv2.imwrite(p, src_canny)
        paths["source_canny"] = p

    # VR hard-lock: plan→one-point perspective wireframe + ViewContract
    try:
        from cad_ai_lock.perspective import place_living_camera, render_perspective_control

        cam = None
        if layout is not None and getattr(layout, "camera", None) is not None:
            cam = layout.camera
        elif plan.camera is not None:
            cam = plan.camera
        else:
            cam = place_living_camera(plan)
        persp, contract = render_perspective_control(plan, cam, layout=layout)
        paths["perspective"] = str(out_dir / "control_perspective.png")
        cv2.imwrite(paths["perspective"], persp)
        contract_path = out_dir / "view_contract.json"
        contract_path.write_text(
            json.dumps(contract.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        paths["view_contract"] = str(contract_path)
    except Exception as exc:  # noqa: BLE001 — controls must not break pipeline
        paths["perspective_error"] = str(exc)

    return paths
