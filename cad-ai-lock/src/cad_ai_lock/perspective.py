"""Plan → one-point perspective hard lock for VR / effect stills.

Commercial rule: VR must be conditioned on a *projected* RoomPlan control,
not free-form text. This module builds:
  - control_perspective.png  (wireframe walls / openings / furniture)
  - ViewContract             (left/right/far/near adjacency checklist)
  - camera placement from living → primary window/balcony

Validate / repair loops consume ViewContract so wrong door counts fail hard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from cad_ai_lock.geometry import (
    CameraSpec,
    FurnitureLayout,
    Opening,
    OpeningType,
    Point2D,
    RoomPlan,
)


@dataclass
class ViewContract:
    """Hard adjacency checklist for a VR camera pose (plan-derived)."""

    preset: str
    look: str  # e.g. "south_to_balcony"
    far_wall: str  # window | balcony | solid | opening
    left_openings: list[str] = field(default_factory=list)  # door/window/opening
    right_openings: list[str] = field(default_factory=list)
    near_openings: list[str] = field(default_factory=list)
    furniture_hints: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    camera: dict[str, Any] = field(default_factory=dict)
    prompt_lock_zh: str = ""
    prompt_lock_en: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "look": self.look,
            "far_wall": self.far_wall,
            "left_openings": self.left_openings,
            "right_openings": self.right_openings,
            "near_openings": self.near_openings,
            "furniture_hints": self.furniture_hints,
            "forbidden": self.forbidden,
            "camera": self.camera,
            "prompt_lock_zh": self.prompt_lock_zh,
            "prompt_lock_en": self.prompt_lock_en,
        }


def _ppm(plan: RoomPlan) -> float:
    if plan.scale_px_per_m and plan.scale_px_per_m > 0:
        return float(plan.scale_px_per_m)
    x0, y0, x1, y1 = plan.bounding_box()
    return max(x1 - x0, y1 - y0, 1.0) / 10.0


def _room_center(room: dict[str, Any]) -> tuple[float, float]:
    c = room.get("center")
    if c:
        return float(c["x"]), float(c["y"])
    poly = room.get("polygon") or []
    if not poly:
        return 0.0, 0.0
    return (
        sum(p["x"] for p in poly) / len(poly),
        sum(p["y"] for p in poly) / len(poly),
    )


def find_living_room(plan: RoomPlan) -> dict[str, Any] | None:
    rooms = plan.rooms or []
    for r in rooms:
        name = str(r.get("name") or "")
        rtype = str(r.get("type") or "")
        if "客厅" in name or rtype == "living":
            return r
    if not rooms:
        return None
    # largest non-balcony non-bath as living proxy
    ranked = sorted(
        rooms,
        key=lambda r: (
            0 if str(r.get("type")) in ("balcony", "bath", "foyer") else 1,
            int(r.get("area_px") or 0),
        ),
        reverse=True,
    )
    return ranked[0] if ranked else None


def find_primary_view_target(plan: RoomPlan) -> tuple[float, float, str]:
    """Return (tx, ty, kind) for main window/balcony opening center."""
    # Prefer balcony room center south of living
    for r in plan.rooms or []:
        if str(r.get("type")) == "balcony" or "阳台" in str(r.get("name") or ""):
            return (*_room_center(r), "balcony")
    wins = [o for o in plan.openings if o.type in (OpeningType.WINDOW, OpeningType.OPENING)]
    if wins:
        best = max(wins, key=lambda o: o.length_px)
        return (
            (best.start.x + best.end.x) / 2,
            (best.start.y + best.end.y) / 2,
            "window",
        )
    x0, y0, x1, y1 = plan.bounding_box()
    return ((x0 + x1) / 2, y1, "balcony")


def place_living_camera(plan: RoomPlan) -> CameraSpec:
    """Camera in living, looking toward balcony/window (one-point).

    Prefers labeled living/balcony rooms; falls back to envelope heuristic
    (eye in mid-south living band, look to southern exterior opening).
    """
    living = find_living_room(plan)
    tx, ty, kind = find_primary_view_target(plan)
    x0, y0, x1, y1 = plan.bounding_box()
    if living:
        cx, cy = _room_center(living)
        # If "living" was mis-labeled as a huge balcony blob, fall back
        if str(living.get("type")) == "balcony" or cy > (y0 + y1) * 0.72:
            cx = (x0 + x1) / 2
            cy = y0 + (y1 - y0) * 0.48
            kind = "balcony"
            tx, ty = (x0 + x1) / 2, y1 - 5
    else:
        cx = (x0 + x1) / 2
        cy = y0 + (y1 - y0) * 0.48
        tx, ty, kind = (x0 + x1) / 2, y1 - 5, "balcony"
    # Ensure we look roughly south (+Y in image) for typical sales plans
    if ty < cy:
        ty = y1 - 5
        kind = "balcony"
    dx, dy = tx - cx, ty - cy
    cx2 = cx - 0.12 * dx
    cy2 = cy - 0.12 * dy
    return CameraSpec(
        preset="living_one_point",
        vanishing_point=Point2D(x=float(tx), y=float(ty)),
        horizon_y=float(ty),
        camera_height_m=1.5,
        wall_height_m=2.8,
        notes=f"living→{kind}; eye=({cx2:.1f},{cy2:.1f}) target=({tx:.1f},{ty:.1f})",
    )


def _basis_from_camera(
    cam_xy: tuple[float, float], target_xy: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return forward (plan), right (plan), up (world z)."""
    fwd = np.array([target_xy[0] - cam_xy[0], target_xy[1] - cam_xy[1]], dtype=np.float64)
    n = np.linalg.norm(fwd) or 1.0
    fwd = fwd / n
    right = np.array([fwd[1], -fwd[0]], dtype=np.float64)  # rotate 90° CW in image coords
    # In image Y-down, for "look south (+Y)", right should be +X
    # rotate so right = perpendicular
    right = np.array([fwd[1], -fwd[0]])
    # Prefer right pointing to increasing X when looking +Y
    if abs(fwd[1]) >= abs(fwd[0]) and fwd[1] > 0:
        right = np.array([1.0, 0.0])
        fwd = np.array([0.0, 1.0])
    elif abs(fwd[1]) >= abs(fwd[0]) and fwd[1] < 0:
        right = np.array([-1.0, 0.0])
        fwd = np.array([0.0, -1.0])
    up = np.array([0.0, 0.0, 1.0])
    return fwd, right, up


def project_point(
    px: float,
    py: float,
    pz: float,
    *,
    cam_xy: tuple[float, float],
    fwd: np.ndarray,
    right: np.ndarray,
    cam_h_m: float,
    ppm: float,
    focal: float,
    canvas_w: int,
    canvas_h: int,
    horizon_y: float,
) -> tuple[float, float, float] | None:
    """Project plan (px,py) + height pz(m) → screen (sx,sy) + depth. None if behind."""
    rel = np.array([px - cam_xy[0], py - cam_xy[1]], dtype=np.float64)
    depth_px = float(rel @ fwd)
    if depth_px < 8.0:
        return None
    lat_px = float(rel @ right)
    # meters
    depth_m = depth_px / ppm
    lat_m = lat_px / ppm
    elev_m = pz - cam_h_m
    sx = canvas_w * 0.5 + focal * (lat_m / depth_m)
    sy = horizon_y - focal * (elev_m / depth_m)
    return sx, sy, depth_m


def _opening_side(
    op: Opening,
    cam_xy: tuple[float, float],
    fwd: np.ndarray,
    right: np.ndarray,
) -> str | None:
    mx = (op.start.x + op.end.x) / 2
    my = (op.start.y + op.end.y) / 2
    rel = np.array([mx - cam_xy[0], my - cam_xy[1]], dtype=np.float64)
    depth = float(rel @ fwd)
    lat = float(rel @ right)
    if depth < 5:
        return "near"
    # far band: deep and near centerline
    if depth > 40 and abs(lat) < depth * 0.35:
        return "far"
    if lat < -15:
        return "left"
    if lat > 15:
        return "right"
    if depth > 50:
        return "far"
    return "near"


def _living_bbox(plan: RoomPlan) -> tuple[float, float, float, float] | None:
    living = find_living_room(plan)
    x0, y0, x1, y1 = plan.bounding_box()
    if living and str(living.get("type")) not in ("balcony", "bath"):
        poly = living.get("polygon") or []
        if len(poly) >= 2:
            xs = [p["x"] for p in poly]
            ys = [p["y"] for p in poly]
            bx0, by0, bx1, by1 = min(xs), min(ys), max(xs), max(ys)
            # reject mislabeled giant blobs
            if (bx1 - bx0) * (by1 - by0) < 0.55 * (x1 - x0) * (y1 - y0):
                return bx0, by0, bx1, by1
    # sales-plan heuristic: living = central band above balcony
    return (
        x0 + (x1 - x0) * 0.28,
        y0 + (y1 - y0) * 0.35,
        x0 + (x1 - x0) * 0.72,
        y0 + (y1 - y0) * 0.78,
    )


def _on_living_perimeter(
    op: Opening, bbox: tuple[float, float, float, float], margin: float = 28.0
) -> bool:
    mx = (op.start.x + op.end.x) / 2
    my = (op.start.y + op.end.y) / 2
    x0, y0, x1, y1 = bbox
    near_l = abs(mx - x0) <= margin and y0 - margin <= my <= y1 + margin
    near_r = abs(mx - x1) <= margin and y0 - margin <= my <= y1 + margin
    near_t = abs(my - y0) <= margin and x0 - margin <= mx <= x1 + margin
    near_b = abs(my - y1) <= margin and x0 - margin <= mx <= x1 + margin
    inside = (x0 - margin <= mx <= x1 + margin) and (y0 - margin <= my <= y1 + margin)
    return inside and (near_l or near_r or near_t or near_b)


def build_view_contract(
    plan: RoomPlan,
    camera: CameraSpec,
    layout: FurnitureLayout | None = None,
) -> ViewContract:
    notes = camera.notes or ""
    look = "south_to_balcony" if "balcony" in notes else "to_window"
    vp = camera.vanishing_point or Point2D(x=plan.image_width / 2, y=plan.image_height * 0.7)
    # parse eye from notes if present
    cam_xy = (plan.image_width / 2.0, plan.image_height * 0.45)
    if "eye=(" in notes:
        try:
            part = notes.split("eye=(")[1].split(")")[0]
            cam_xy = tuple(map(float, part.split(",")))  # type: ignore[assignment]
        except Exception:
            pass
    fwd, right, _ = _basis_from_camera(cam_xy, (vp.x, vp.y))
    living_box = _living_bbox(plan)

    left: list[str] = []
    right_o: list[str] = []
    near: list[str] = []
    far_wall = "solid"
    for op in plan.openings:
        # Only openings on living perimeter count for VR adjacency
        if living_box and not _on_living_perimeter(op, living_box):
            continue
        side = _opening_side(op, cam_xy, fwd, right)
        kind = op.type.value
        # Side walls of living: doors matter; bedroom bay windows are NOT living side windows
        if side in ("left", "right") and kind == "window":
            continue
        if side == "left":
            left.append(kind)
        elif side == "right":
            right_o.append(kind)
        elif side == "near":
            near.append(kind)
        elif side == "far":
            if kind in ("window", "opening"):
                far_wall = "balcony" if "balcony" in notes else "window"
            elif kind == "door":
                far_wall = "opening"

    # Heuristic boost for Chinese living→balcony plans: far is balcony if target says so
    if "balcony" in notes:
        far_wall = "balcony"

    # Clamp noisy extract: living→balcony views rarely have >1 door per side wall
    def _clamp_doors(seq: list[str], max_doors: int = 1) -> list[str]:
        doors = [x for x in seq if x == "door"][:max_doors]
        other = [x for x in seq if x != "door"]
        return doors + other

    left = _clamp_doors(left, 1)
    right_o = _clamp_doors(right_o, 1)
    # If extract missed left door but living→balcony, keep typical 1 door on bedroom side
    if far_wall == "balcony" and not left:
        left = ["door"]

    furniture_hints: list[str] = []
    if layout:
        types = {it.type for it in layout.items}
        if "sofa" in types:
            furniture_hints.append("sofa_in_living")
        if "tv_console" in types or "tv" in types:
            furniture_hints.append("tv_on_side_wall")
        if "coffee_table" in types:
            furniture_hints.append("coffee_table_center")
    if not furniture_hints:
        furniture_hints = ["sofa_facing_tv_or_balcony", "coffee_table", "tv_console_on_right_or_left_wall"]

    forbidden = [
        "do not invent a second door on the left wall",
        "do not invent extra doors on left/right walls beyond the contract counts",
        "far wall must be full-width balcony sliding glass, not a narrow door",
        "do not show bedroom beds in this living view",
        "do not change room adjacency from the floorplan",
        "dining may be visible behind camera / as open passage north — not as side doors",
    ]

    prompt_zh = (
        f"硬锁一点透视：站在客厅看向{'景观阳台' if far_wall=='balcony' else '主窗'}；"
        f"远端墙={far_wall}；"
        f"左墙门窗={left or ['无']}；右墙门窗={right_o or ['无或电视墙']}；"
        f"禁止增减门洞，禁止改户型相邻关系。"
    )
    prompt_en = (
        f"HARD LOCK one-point living VR looking to {far_wall}. "
        f"Far wall MUST be {far_wall} glazing. "
        f"Left wall openings: {left or ['none']}. "
        f"Right wall openings: {right_o or ['none_or_tv_wall']}. "
        f"Do NOT invent doors. Keep exact adjacency from floorplan."
    )

    return ViewContract(
        preset=camera.preset,
        look=look,
        far_wall=far_wall,
        left_openings=left,
        right_openings=right_o,
        near_openings=near,
        furniture_hints=furniture_hints,
        forbidden=forbidden,
        camera={
            "eye": {"x": cam_xy[0], "y": cam_xy[1]},
            "target": {"x": vp.x, "y": vp.y},
            "height_m": camera.camera_height_m,
            "wall_height_m": camera.wall_height_m,
            "notes": camera.notes,
        },
        prompt_lock_zh=prompt_zh,
        prompt_lock_en=prompt_en,
    )


def render_perspective_control(
    plan: RoomPlan,
    camera: CameraSpec,
    *,
    layout: FurnitureLayout | None = None,
    width: int = 720,
    height: int = 1280,
    focal: float | None = None,
) -> tuple[np.ndarray, ViewContract]:
    """Render vertical VR control canvas (BGR) + ViewContract."""
    ppm = _ppm(plan)
    vp = camera.vanishing_point or Point2D(x=plan.image_width / 2, y=plan.image_height * 0.75)
    notes = camera.notes or ""
    cam_xy = (plan.image_width / 2.0, plan.image_height * 0.45)
    if "eye=(" in notes:
        try:
            part = notes.split("eye=(")[1].split(")")[0]
            cam_xy = tuple(map(float, part.split(",")))  # type: ignore[assignment]
        except Exception:
            pass

    fwd, right, _ = _basis_from_camera(cam_xy, (vp.x, vp.y))
    horizon = height * 0.42
    if focal is None:
        focal = width * 0.85

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (18, 18, 18)

    def proj(px: float, py: float, pz: float):
        return project_point(
            px,
            py,
            pz,
            cam_xy=cam_xy,
            fwd=fwd,
            right=right,
            cam_h_m=camera.camera_height_m,
            ppm=ppm,
            focal=float(focal),
            canvas_w=width,
            canvas_h=height,
            horizon_y=horizon,
        )

    wall_h = camera.wall_height_m

    # Floor grid toward VP for depth cue
    for t in np.linspace(0.15, 1.0, 8):
        y = cam_xy[1] + (vp.y - cam_xy[1]) * t
        x0 = cam_xy[0] - 200
        x1 = cam_xy[0] + 200
        p0 = proj(x0, y, 0.0)
        p1 = proj(x1, y, 0.0)
        if p0 and p1:
            cv2.line(
                canvas,
                (int(p0[0]), int(p0[1])),
                (int(p1[0]), int(p1[1])),
                (50, 50, 50),
                1,
                cv2.LINE_AA,
            )

    # Walls as vertical quads
    for wall in plan.walls:
        a = proj(wall.start.x, wall.start.y, 0.0)
        b = proj(wall.end.x, wall.end.y, 0.0)
        a2 = proj(wall.start.x, wall.start.y, wall_h)
        b2 = proj(wall.end.x, wall.end.y, wall_h)
        if not all([a, b, a2, b2]):
            continue
        pts = np.array(
            [
                [a[0], a[1]],
                [b[0], b[1]],
                [b2[0], b2[1]],
                [a2[0], a2[1]],
            ],
            dtype=np.int32,
        )
        cv2.polylines(canvas, [pts], True, (240, 240, 240), 2, cv2.LINE_AA)

    # Openings colored
    for op in plan.openings:
        color = (0, 255, 255) if op.type == OpeningType.DOOR else (0, 200, 255)
        if op.type == OpeningType.WINDOW:
            color = (255, 180, 0)
        if op.type == OpeningType.OPENING:
            color = (180, 180, 255)
        a = proj(op.start.x, op.start.y, 0.05)
        b = proj(op.end.x, op.end.y, 0.05)
        a2 = proj(op.start.x, op.start.y, wall_h * 0.9)
        b2 = proj(op.end.x, op.end.y, wall_h * 0.9)
        if not all([a, b, a2, b2]):
            continue
        pts = np.array(
            [[a[0], a[1]], [b[0], b[1]], [b2[0], b2[1]], [a2[0], a2[1]]],
            dtype=np.int32,
        )
        cv2.polylines(canvas, [pts], True, color, 3, cv2.LINE_AA)

    # Furniture boxes (soft)
    if layout:
        for it in layout.items:
            hw, hd = it.width_px / 2, it.depth_px / 2
            corners = [
                (it.center.x - hw, it.center.y - hd),
                (it.center.x + hw, it.center.y - hd),
                (it.center.x + hw, it.center.y + hd),
                (it.center.x - hw, it.center.y + hd),
            ]
            z = 0.45 if it.type in ("sofa", "bed") else 0.35
            scr = [proj(x, y, 0.0) for x, y in corners]
            scr2 = [proj(x, y, z) for x, y in corners]
            if any(p is None for p in scr + scr2):
                continue
            for i in range(4):
                j = (i + 1) % 4
                cv2.line(
                    canvas,
                    (int(scr[i][0]), int(scr[i][1])),  # type: ignore[index]
                    (int(scr[j][0]), int(scr[j][1])),  # type: ignore[index]
                    (160, 160, 160),
                    1,
                    cv2.LINE_AA,
                )
                cv2.line(
                    canvas,
                    (int(scr[i][0]), int(scr[i][1])),  # type: ignore[index]
                    (int(scr2[i][0]), int(scr2[i][1])),  # type: ignore[index]
                    (120, 200, 120),
                    1,
                    cv2.LINE_AA,
                )

    # Vanishing point marker
    cv2.circle(canvas, (width // 2, int(horizon)), 6, (0, 80, 255), -1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "PERSPECTIVE LOCK",
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )

    contract = build_view_contract(plan, camera, layout)
    # Annotate contract counts on canvas
    cv2.putText(
        canvas,
        f"far={contract.far_wall} L={contract.left_openings} R={contract.right_openings}",
        (16, height - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 255, 180),
        1,
        cv2.LINE_AA,
    )
    return canvas, contract


def living_vr_prompt(contract: ViewContract, style: str) -> str:
    """Compose generation prompt with hard adjacency lock prefix."""
    return (
        f"{contract.prompt_lock_en} Style: {style}. "
        f"Forbidden: {'; '.join(contract.forbidden)}. "
        f"Furniture: {', '.join(contract.furniture_hints) or 'sofa, coffee table, tv console'}."
    )


def assess_vr_against_contract(
    contract: ViewContract,
    *,
    detected_left_doors: int | None = None,
    detected_right_doors: int | None = None,
    detected_far: str | None = None,
) -> dict[str, Any]:
    """Score a VR still vs ViewContract (optional CV detections).

    When detections are None, returns contract-only readiness (control exists).
    """
    issues: list[str] = []
    expect_left = sum(1 for x in contract.left_openings if x == "door")
    expect_right = sum(1 for x in contract.right_openings if x == "door")
    if detected_left_doors is not None and abs(detected_left_doors - expect_left) > 1:
        issues.append(
            f"left_doors {detected_left_doors} != expected~{expect_left}"
        )
    if detected_right_doors is not None and abs(detected_right_doors - expect_right) > 1:
        issues.append(
            f"right_doors {detected_right_doors} != expected~{expect_right}"
        )
    if detected_far and contract.far_wall in ("balcony", "window"):
        if detected_far not in ("balcony", "window", "glass"):
            issues.append(f"far_wall {detected_far} != {contract.far_wall}")
    passed = len(issues) == 0
    return {
        "passed": passed,
        "issues": issues,
        "expected": {
            "far_wall": contract.far_wall,
            "left_doors": expect_left,
            "right_doors": expect_right,
        },
        "detected": {
            "left_doors": detected_left_doors,
            "right_doors": detected_right_doors,
            "far": detected_far,
        },
    }
