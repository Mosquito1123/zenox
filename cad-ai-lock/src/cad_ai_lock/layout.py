"""Soft layout layer: templates + policy checks on top of locked RoomPlan."""

from __future__ import annotations

import math
import uuid
from typing import Any

import numpy as np

from cad_ai_lock.geometry import (
    CameraSpec,
    FurnitureItem,
    FurnitureLayout,
    OpeningType,
    Point2D,
    RoomPlan,
)
from cad_ai_lock.policies_loader import load_packs, resolve_packs


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:6]}"


def _poly_bbox(poly: list[dict[str, float]]) -> tuple[float, float, float, float]:
    xs = [p["x"] for p in poly]
    ys = [p["y"] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _poly_point(poly: list[dict[str, float]], u: float, v: float) -> Point2D:
    x0, y0, x1, y1 = _poly_bbox(poly)
    return Point2D(x=x0 + (x1 - x0) * u, y=y0 + (y1 - y0) * v)


def _room_name(room: dict[str, Any]) -> str:
    return str(room.get("name") or room.get("id") or "")


def _match_template(room: dict[str, Any], templates: dict[str, Any]) -> dict[str, Any] | None:
    name = _room_name(room)
    for tmpl in templates.values():
        keys = tmpl.get("match_room_names") or []
        if any(k.lower() in name.lower() for k in keys):
            return tmpl
    # fallback by id hints
    rid = str(room.get("id") or "").lower()
    if "living" in rid or "客厅" in name:
        return templates.get("living_small")
    if "bed" in rid or "卧" in name:
        return templates.get("bedroom_default")
    if "din" in rid or "餐" in name:
        return templates.get("dining_default")
    return None


def _main_window(plan: RoomPlan) -> tuple[float, float] | None:
    wins = [o for o in plan.openings if o.type == OpeningType.WINDOW]
    if not wins:
        return None
    best = max(wins, key=lambda o: o.length_px)
    return ((best.start.x + best.end.x) / 2, (best.start.y + best.end.y) / 2)


def _doors(plan: RoomPlan) -> list[tuple[float, float]]:
    out = []
    for o in plan.openings:
        if o.type == OpeningType.DOOR:
            out.append(((o.start.x + o.end.x) / 2, (o.start.y + o.end.y) / 2))
    return out


def _px_per_m(plan: RoomPlan) -> float:
    if plan.scale_px_per_m:
        return float(plan.scale_px_per_m)
    # fallback: assume plan bbox ≈ 8m on longer side
    x0, y0, x1, y1 = plan.bounding_box()
    span = max(x1 - x0, y1 - y0, 1.0)
    return span / 8.0


def _rotation_toward(center: Point2D, target: tuple[float, float] | None) -> float:
    if target is None:
        return 0.0
    return math.degrees(math.atan2(target[1] - center.y, target[0] - center.x))


def place_from_templates(
    plan: RoomPlan,
    packs: list[dict[str, Any]],
) -> list[FurnitureItem]:
    templates: dict[str, Any] = {}
    for p in packs:
        if p.get("id") == "layout_templates":
            templates = p.get("templates") or {}
            break
    if not templates:
        return []

    ppm = _px_per_m(plan)
    win = _main_window(plan)
    items: list[FurnitureItem] = []
    rooms = plan.rooms or []
    if not rooms:
        # synthesize one room from bbox so templates still run
        x0, y0, x1, y1 = plan.bounding_box()
        rooms = [
            {
                "id": "living",
                "name": "客厅",
                "polygon": [
                    {"x": x0, "y": y0},
                    {"x": x1, "y": y0},
                    {"x": x1, "y": y1},
                    {"x": x0, "y": y1},
                ],
            }
        ]

    for room in rooms:
        poly = room.get("polygon") or []
        if len(poly) < 3:
            continue
        tmpl = _match_template(room, templates)
        if not tmpl:
            continue
        for spec in tmpl.get("items") or []:
            rel = spec.get("relative") or {"u": 0.5, "v": 0.5}
            center = _poly_point(poly, float(rel.get("u", 0.5)), float(rel.get("v", 0.5)))
            size_m = spec.get("size_m") or [1.0, 0.6]
            w_px = float(size_m[0]) * ppm
            d_px = float(size_m[1]) * ppm
            face = spec.get("face") or "none"
            rot = 0.0
            if face in ("toward_window_or_center", "toward_window"):
                rot = _rotation_toward(center, win)
            elif face == "into_room":
                cx = sum(p["x"] for p in poly) / len(poly)
                cy = sum(p["y"] for p in poly) / len(poly)
                rot = _rotation_toward(center, (cx, cy))
            elif face == "head_to_nearest_solid_wall":
                # approximate: face away from nearest bbox edge (head to wall)
                x0, y0, x1, y1 = _poly_bbox(poly)
                dists = {
                    "n": abs(center.y - y0),
                    "s": abs(y1 - center.y),
                    "w": abs(center.x - x0),
                    "e": abs(x1 - center.x),
                }
                side = min(dists, key=dists.get)  # type: ignore[arg-type]
                rot = {"n": 90, "s": -90, "w": 0, "e": 180}[side]
            items.append(
                FurnitureItem(
                    id=_new_id(spec.get("type", "item")),
                    type=str(spec.get("type", "item")),
                    room_id=str(room.get("id") or ""),
                    center=center,
                    width_px=w_px,
                    depth_px=d_px,
                    rotation_deg=rot,
                    face=face,
                )
            )
    return _push_from_doors(plan, items)


def _push_from_doors(plan: RoomPlan, items: list[FurnitureItem]) -> list[FurnitureItem]:
    """Common-sense: nudge furniture away from door centers."""
    doors = _doors(plan)
    if not doors:
        return items
    ppm = _px_per_m(plan)
    clear = 0.7 * ppm
    out: list[FurnitureItem] = []
    for it in items:
        cx, cy = it.center.x, it.center.y
        for dx, dy in doors:
            dist = math.hypot(cx - dx, cy - dy)
            if dist < clear and dist > 1e-3:
                ux, uy = (cx - dx) / dist, (cy - dy) / dist
                cx += ux * (clear - dist)
                cy += uy * (clear - dist)
        out.append(it.model_copy(update={"center": Point2D(x=cx, y=cy)}))
    return out


def evaluate_policies(
    plan: RoomPlan,
    items: list[FurnitureItem],
    packs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Soft scorecard: 0..1 per rule + overall."""
    win = _main_window(plan)
    doors = _doors(plan)
    ppm = _px_per_m(plan)
    results: list[dict[str, Any]] = []
    total_w = 0.0
    total_s = 0.0

    def add(rule_id: str, weight: float, score: float, detail: str) -> None:
        nonlocal total_w, total_s
        w = float(weight)
        s = float(np.clip(score, 0, 1))
        results.append(
            {"id": rule_id, "weight": w, "score": round(s, 3), "detail": detail}
        )
        total_w += w
        total_s += w * s

    for pack in packs:
        for rule in pack.get("rules") or []:
            rid = rule.get("id", "rule")
            w = float(rule.get("weight", 0.5))
            check = rule.get("check", "")
            if check == "no_furniture_in_door_clearance":
                bad = 0
                for it in items:
                    for dx, dy in doors:
                        if math.hypot(it.center.x - dx, it.center.y - dy) < 0.55 * ppm:
                            bad += 1
                add(rid, w, 1.0 if bad == 0 else max(0.0, 1 - 0.25 * bad), f"conflicts={bad}")
            elif check == "primary_seat_near_main_window":
                sofas = [i for i in items if i.type == "sofa"]
                if not sofas or win is None:
                    add(rid, w, 0.6, "no sofa/window")
                else:
                    d = math.hypot(sofas[0].center.x - win[0], sofas[0].center.y - win[1])
                    # closer is better; normalize by plan diagonal
                    diag = math.hypot(plan.image_width, plan.image_height)
                    add(rid, w, float(np.clip(1.2 - d / (0.45 * diag), 0, 1)), f"dist_px={d:.0f}")
            elif check == "bed_head_on_solid_wall" or check == "bed_not_face_door":
                beds = [i for i in items if i.type == "bed"]
                if not beds:
                    add(rid, w, 0.7, "no bed")
                else:
                    bed = beds[0]
                    # facing door is bad for fengshui / also common sense variant
                    face_door = False
                    for dx, dy in doors:
                        ang = abs(
                            (
                                _rotation_toward(bed.center, (dx, dy))
                                - bed.rotation_deg
                                + 180
                            )
                            % 360
                            - 180
                        )
                        if ang < 35 and math.hypot(bed.center.x - dx, bed.center.y - dy) < 3.5 * ppm:
                            face_door = True
                    if check == "bed_not_face_door":
                        add(rid, w, 0.2 if face_door else 1.0, f"face_door={face_door}")
                    else:
                        add(rid, w, 0.85 if not face_door else 0.45, "head/wall heuristic")
            elif check == "sofa_not_blocking_main_window":
                sofas = [i for i in items if i.type == "sofa"]
                if not sofas or win is None:
                    add(rid, w, 0.75, "skip")
                else:
                    d = math.hypot(sofas[0].center.x - win[0], sofas[0].center.y - win[1])
                    add(rid, w, 1.0 if d > 0.8 * ppm else 0.4, f"dist_px={d:.0f}")
            elif check == "min_corridor_clearance":
                add(rid, w, 0.8, "heuristic pass (door push applied)")
            elif check.startswith("mirror_") or check.startswith("toilet_") or check.startswith("stove_"):
                # optional fengshui stubs: neutral until those items exist
                has = any(i.type in check for i in items)
                add(rid, w, 0.7 if not has else 0.6, "optional / incomplete inventory")
            else:
                add(rid, w, 0.75, f"untyped check={check}")

    overall = (total_s / total_w) if total_w > 0 else 0.0
    return {
        "overall": round(float(overall), 4),
        "rules": results,
        "item_count": len(items),
        "has_scale": plan.scale_px_per_m is not None,
    }


def build_prompt_hints(items: list[FurnitureItem], scorecard: dict[str, Any]) -> list[str]:
    hints: list[str] = []
    types = {i.type for i in items}
    if "sofa" in types:
        hints.append("客厅布置沙发与茶几，动线避开门口")
    if "bed" in types:
        hints.append("卧室床头靠实墙，床不对门")
    if "dining_table" in types:
        hints.append("餐厅桌椅居中，保留通行")
    if "tv_console" in types:
        hints.append("电视柜与沙发相对")
    if scorecard.get("has_scale") is False:
        hints.append("比例尺未知，家具尺度为相对估计")
    return hints


def default_camera(plan: RoomPlan) -> CameraSpec:
    """Living → primary balcony/window one-point camera (VR hard-lock pose)."""
    from cad_ai_lock.perspective import place_living_camera

    return place_living_camera(plan)


def build_layout(
    plan: RoomPlan,
    *,
    daylight: bool = True,
    fengshui: bool = False,
    common_sense: bool = True,
    templates: bool = True,
    extra_packs: list[str] | None = None,
    camera: CameraSpec | None = None,
) -> FurnitureLayout:
    pack_ids = resolve_packs(
        daylight=daylight,
        fengshui=fengshui,
        common_sense=common_sense,
        templates=templates,
        extra=extra_packs,
    )
    # honor plan.policies if set
    if plan.policies:
        pack_ids = list(dict.fromkeys([*plan.policies, *pack_ids]))
    packs = load_packs(pack_ids)
    items = place_from_templates(plan, packs)
    scorecard = evaluate_policies(plan, items, packs)
    hints = build_prompt_hints(items, scorecard)
    cam = camera or plan.camera or default_camera(plan)
    return FurnitureLayout(
        plan_revision=plan.plan_revision,
        policies=pack_ids,
        camera=cam,
        items=items,
        prompt_hints=hints,
        scorecard=scorecard,
        notes="soft layout on locked RoomPlan; walls/doors/windows unchanged",
    )
