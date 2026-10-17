"""Automatic RoomPlan repair strategies driven by assess.PlanHealth issues.

Each repair is idempotent and records itself in plan.meta['repair_log'].
"""

from __future__ import annotations

import math
import re
import uuid
from typing import Any, Callable

import cv2
import numpy as np

from cad_ai_lock.assess import PlanHealth, assess_plan
from cad_ai_lock.geometry import Opening, OpeningType, Point2D, RoomPlan, Wall


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _log(plan: RoomPlan, action: str, detail: dict[str, Any] | None = None) -> None:
    log = plan.meta.setdefault("repair_log", [])
    log.append({"action": action, **(detail or {})})


def _angle(w: Wall) -> float:
    return math.degrees(math.atan2(w.end.y - w.start.y, w.end.x - w.start.x)) % 180


def orthogonal_snap(plan: RoomPlan, tol_deg: float = 14.0) -> RoomPlan:
    walls: list[Wall] = []
    for w in plan.walls:
        a = _angle(w)
        d0 = min(abs(a - 0), abs(a - 180))
        d90 = abs(a - 90)
        if d0 <= tol_deg and d0 <= d90:
            y = (w.start.y + w.end.y) / 2
            walls.append(
                Wall(
                    id=w.id,
                    start=Point2D(x=w.start.x, y=y),
                    end=Point2D(x=w.end.x, y=y),
                    thickness_px=w.thickness_px,
                    exterior=w.exterior,
                )
            )
        elif d90 <= tol_deg:
            x = (w.start.x + w.end.x) / 2
            walls.append(
                Wall(
                    id=w.id,
                    start=Point2D(x=x, y=w.start.y),
                    end=Point2D(x=x, y=w.end.y),
                    thickness_px=w.thickness_px,
                    exterior=w.exterior,
                )
            )
        else:
            walls.append(w)
    plan.walls = walls
    _log(plan, "orthogonal_snap", {"kept": len(walls)})
    return plan


def _as_seg(w: Wall) -> list[float]:
    a = _angle(w)
    if min(abs(a), abs(a - 180)) <= abs(a - 90):
        y = (w.start.y + w.end.y) / 2
        x1, x2 = sorted([w.start.x, w.end.x])
        return [x1, y, x2, y, x2 - x1, 0.0, w.thickness_px, 1.0 if w.exterior else 0.0]
    x = (w.start.x + w.end.x) / 2
    y1, y2 = sorted([w.start.y, w.end.y])
    return [x, y1, x, y2, y2 - y1, 90.0, w.thickness_px, 1.0 if w.exterior else 0.0]


def merge_collinear(
    plan: RoomPlan, dist_tol: float = 10.0, gap_tol: float = 28.0, min_len: float = 40.0
) -> RoomPlan:
    segs = [_as_seg(w) for w in plan.walls]
    segs = sorted(segs, key=lambda s: -s[4])
    used = [False] * len(segs)
    out: list[list[float]] = []
    for i, s in enumerate(segs):
        if used[i]:
            continue
        used[i] = True
        group = [s]
        changed = True
        while changed:
            changed = False
            for j, t in enumerate(segs):
                if used[j] or abs(s[5] - t[5]) > 0.1:
                    continue
                if s[5] == 0:
                    if abs((s[1] + s[3]) / 2 - (t[1] + t[3]) / 2) > dist_tol:
                        continue
                    xs = sorted([s[0], s[2], t[0], t[2]])
                    if xs[2] - xs[1] > gap_tol:
                        continue
                else:
                    if abs((s[0] + s[2]) / 2 - (t[0] + t[2]) / 2) > dist_tol:
                        continue
                    ys = sorted([s[1], s[3], t[1], t[3]])
                    if ys[2] - ys[1] > gap_tol:
                        continue
                used[j] = True
                group.append(t)
                changed = True
        if s[5] == 0:
            y = float(np.median([(g[1] + g[3]) / 2 for g in group]))
            xs = [c for g in group for c in (g[0], g[2])]
            thick = float(np.median([g[6] for g in group]))
            ext = max(g[7] for g in group)
            out.append([min(xs), y, max(xs), y, max(xs) - min(xs), 0.0, thick, ext])
        else:
            x = float(np.median([(g[0] + g[2]) / 2 for g in group]))
            ys = [c for g in group for c in (g[1], g[3])]
            thick = float(np.median([g[6] for g in group]))
            ext = max(g[7] for g in group)
            out.append([x, min(ys), x, max(ys), max(ys) - min(ys), 90.0, thick, ext])

    walls: list[Wall] = []
    for s in out:
        if s[4] < min_len:
            continue
        walls.append(
            Wall(
                id=_new_id("wall"),
                start=Point2D(x=float(s[0]), y=float(s[1])),
                end=Point2D(x=float(s[2]), y=float(s[3])),
                thickness_px=max(4.0, float(s[6])),
                exterior=bool(s[7]),
            )
        )
    plan.walls = walls
    _log(plan, "merge_collinear", {"walls": len(walls)})
    return plan


def dedupe_walls(plan: RoomPlan, tol: float = 18.0) -> RoomPlan:
    kept: list[Wall] = []
    for w in sorted(plan.walls, key=lambda x: -x.length_px):
        dup = False
        for k in kept:
            d1 = math.hypot(w.start.x - k.start.x, w.start.y - k.start.y) + math.hypot(
                w.end.x - k.end.x, w.end.y - k.end.y
            )
            d2 = math.hypot(w.start.x - k.end.x, w.start.y - k.end.y) + math.hypot(
                w.end.x - k.start.x, w.end.y - k.start.y
            )
            if min(d1, d2) < tol * 2 and abs(w.length_px - k.length_px) < 40:
                dup = True
                break
        if not dup:
            kept.append(w)
    plan.walls = kept
    _log(plan, "dedupe_walls", {"walls": len(kept)})
    return plan


def _cluster(vals: list[float], tol: float = 8.0) -> list[float]:
    if not vals:
        return []
    vals = sorted(vals)
    groups: list[list[float]] = [[vals[0]]]
    for v in vals[1:]:
        if v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [float(np.median(g)) for g in groups]


def grid_snap(plan: RoomPlan, tol: float = 8.0) -> RoomPlan:
    xs = [c for w in plan.walls for c in (w.start.x, w.end.x)]
    ys = [c for w in plan.walls for c in (w.start.y, w.end.y)]
    cx, cy = _cluster(xs, tol), _cluster(ys, tol)
    if not cx or not cy:
        return plan

    def nearest(v: float, grid: list[float]) -> float:
        return min(grid, key=lambda g: abs(g - v))

    walls: list[Wall] = []
    for w in plan.walls:
        a = _angle(w)
        if min(abs(a), abs(a - 180)) <= abs(a - 90):
            y = nearest((w.start.y + w.end.y) / 2, cy)
            x1, x2 = nearest(w.start.x, cx), nearest(w.end.x, cx)
            if abs(x2 - x1) < 35:
                continue
            walls.append(
                Wall(
                    id=w.id or _new_id("wall"),
                    start=Point2D(x=min(x1, x2), y=y),
                    end=Point2D(x=max(x1, x2), y=y),
                    thickness_px=w.thickness_px,
                    exterior=w.exterior,
                )
            )
        else:
            x = nearest((w.start.x + w.end.x) / 2, cx)
            y1, y2 = nearest(w.start.y, cy), nearest(w.end.y, cy)
            if abs(y2 - y1) < 35:
                continue
            walls.append(
                Wall(
                    id=w.id or _new_id("wall"),
                    start=Point2D(x=x, y=min(y1, y2)),
                    end=Point2D(x=x, y=max(y1, y2)),
                    thickness_px=w.thickness_px,
                    exterior=w.exterior,
                )
            )
    plan.walls = walls
    plan.meta["grid"] = {"x": cx, "y": cy}
    _log(plan, "grid_snap", {"walls": len(walls), "nx": len(cx), "ny": len(cy)})
    return plan


def snap_junctions(plan: RoomPlan, tol: float = 14.0) -> RoomPlan:
    """Pull nearby endpoints onto shared junctions."""
    pts: list[list[float]] = []
    for w in plan.walls:
        pts.append([w.start.x, w.start.y])
        pts.append([w.end.x, w.end.y])
    # cluster
    centers: list[list[float]] = []
    for p in pts:
        placed = False
        for c in centers:
            if math.hypot(p[0] - c[0], p[1] - c[1]) <= tol:
                c[0] = (c[0] + p[0]) / 2
                c[1] = (c[1] + p[1]) / 2
                c[2] += 1
                placed = True
                break
        if not placed:
            centers.append([p[0], p[1], 1.0])

    def snap_pt(x: float, y: float) -> Point2D:
        best = min(centers, key=lambda c: math.hypot(c[0] - x, c[1] - y))
        if math.hypot(best[0] - x, best[1] - y) <= tol:
            return Point2D(x=best[0], y=best[1])
        return Point2D(x=x, y=y)

    plan.walls = [
        Wall(
            id=w.id,
            start=snap_pt(w.start.x, w.start.y),
            end=snap_pt(w.end.x, w.end.y),
            thickness_px=w.thickness_px,
            exterior=w.exterior,
        )
        for w in plan.walls
        if math.hypot(w.end.x - w.start.x, w.end.y - w.start.y) >= 30
    ]
    _log(plan, "snap_junctions", {"junctions": len(centers)})
    return plan


def mark_exterior(plan: RoomPlan) -> RoomPlan:
    if not plan.walls:
        return plan
    x0, y0, x1, y1 = plan.bounding_box()
    margin = 18.0
    for w in plan.walls:
        xs = [w.start.x, w.end.x]
        ys = [w.start.y, w.end.y]
        if (
            abs(min(xs) - x0) < margin
            or abs(max(xs) - x1) < margin
            or abs(min(ys) - y0) < margin
            or abs(max(ys) - y1) < margin
        ):
            w.exterior = True
    _log(plan, "mark_exterior")
    return plan


def infer_rooms(
    plan: RoomPlan,
    image_bgr: np.ndarray | None = None,
    min_area: int = 900,
) -> RoomPlan:
    """Flood-fill rooms from wall raster; optional texture typing."""
    h, w = plan.image_height, plan.image_width
    wall_mask = np.zeros((h, w), np.uint8)
    for wall in plan.walls:
        thickness = max(4, int(round(wall.thickness_px)))
        cv2.line(
            wall_mask,
            wall.start.as_int_tuple(),
            wall.end.as_int_tuple(),
            255,
            thickness,
        )
    ff = wall_mask.copy()
    cv2.floodFill(ff, None, (1, 1), 128)
    outside = ff == 128
    free = ((wall_mask == 0) & (~outside)).astype(np.uint8) * 255
    n, lab, stats, cents = cv2.connectedComponentsWithStats(free, 8)

    gray = None
    local_std = None
    if image_bgr is not None:
        if image_bgr.shape[0] != h or image_bgr.shape[1] != w:
            image_bgr = cv2.resize(image_bgr, (w, h))
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gf = gray.astype(np.float32)
        local_std = np.sqrt(
            np.maximum(
                cv2.blur(gf**2, (15, 15)) - cv2.blur(gf, (15, 15)) ** 2,
                0,
            )
        )

    rooms: list[dict[str, Any]] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y, ww, hh = map(int, stats[i, :4])
        # skip thin margin artifacts
        if y < 18 and area < 4000:
            continue
        rtype, name = "room", "房间"
        std = mean_g = None
        if gray is not None and local_std is not None:
            mask = lab == i
            mean_g = float(gray[mask].mean())
            std = float(local_std[mask].mean())
            if y > h * 0.78 and hh < 140 and std < 16:
                rtype, name = "balcony", "景观阳台"
            elif std > 17 and mean_g < 218:
                rtype, name = "bedroom", "卧室"
            elif mean_g < 210 and std < 14 and area < 9000:
                rtype, name = "bath", "卫生间"
            elif area > max(18000, 0.08 * h * w):
                rtype, name = "living", "客厅"
            elif area < 2800:
                rtype, name = "foyer", "玄关"
            elif y < h * 0.28 and x < w * 0.5 and std < 16:
                rtype, name = "kitchen", "厨房"
        rooms.append(
            {
                "id": f"{rtype}_{i}",
                "name": name,
                "type": rtype,
                "area_px": area,
                "polygon": [
                    {"x": float(x), "y": float(y)},
                    {"x": float(x + ww), "y": float(y)},
                    {"x": float(x + ww), "y": float(y + hh)},
                    {"x": float(x), "y": float(y + hh)},
                ],
                "center": {"x": float(cents[i][0]), "y": float(cents[i][1])},
                "mean_g": mean_g,
                "std": std,
            }
        )
    plan.rooms = rooms
    _log(plan, "infer_rooms", {"rooms": len(rooms)})
    return plan


_LABEL_ALIASES = [
    ("客厅", "客厅", "living"),
    ("起居", "客厅", "living"),
    ("卧室", "卧室", "bedroom"),
    ("主卧", "主卧", "bedroom"),
    ("次卧", "次卧", "bedroom"),
    ("厨房", "厨房", "kitchen"),
    ("出喜", "厨房", "kitchen"),  # OCR misread of 厨房
    ("卫生间", "卫生间", "bath"),
    ("公卫", "公卫", "bath"),
    ("浴室", "卫生间", "bath"),
    ("玄关", "玄关", "foyer"),
    ("阳台", "景观阳台", "balcony"),
    ("家庭吧", "家庭吧", "bar"),
    ("餐厅", "餐厅", "dining"),
    ("书房", "书房", "study"),
    ("客卧", "客卧", "bedroom"),
]


def label_rooms_from_ocr(
    plan: RoomPlan, ocr_items: list[dict[str, Any]], max_dist: float = 140.0
) -> RoomPlan:
    if not plan.rooms or not ocr_items:
        return plan
    for item in ocr_items:
        text = str(item.get("text") or "")
        matched = None
        for key, name, rtype in _LABEL_ALIASES:
            if key in text:
                matched = (name, rtype)
                break
        if not matched:
            continue
        name, rtype = matched
        x, y = float(item["x"]), float(item["y"])
        best = None
        best_d = 1e18
        for room in plan.rooms:
            c = room.get("center") or {}
            d = (float(c.get("x", 0)) - x) ** 2 + (float(c.get("y", 0)) - y) ** 2
            if d < best_d:
                best_d = d
                best = room
        if best is not None and best_d <= max_dist**2:
            best["name"] = name
            best["type"] = rtype
            best["label_source"] = "ocr"
            best["label_text"] = text
    _log(plan, "label_rooms", {"ocr": len(ocr_items)})
    return plan


def estimate_scale_from_ocr(
    plan: RoomPlan, ocr_items: list[dict[str, Any]]
) -> RoomPlan:
    """Infer px/m from dimension annotations (mm) near envelope edges."""
    dims: list[dict[str, Any]] = []
    for item in ocr_items:
        text = str(item.get("text") or "").replace(" ", "")
        for m in re.findall(r"(\d{3,5})", text):
            v = int(m)
            if 500 <= v <= 30000:
                dims.append(
                    {
                        "mm": v,
                        "x": float(item["x"]),
                        "y": float(item["y"]),
                        "text": text,
                    }
                )
    if not dims:
        _log(plan, "estimate_scale", {"ok": False, "reason": "no_dims"})
        return plan

    x0, y0, x1, y1 = plan.bounding_box()
    env_w, env_h = max(1.0, x1 - x0), max(1.0, y1 - y0)
    top = [d for d in dims if d["y"] < y0 + 40 or d["y"] < plan.image_height * 0.12]
    side = [d for d in dims if d["x"] > x1 - 40 or d["x"] > plan.image_width * 0.82]
    scale = None
    source = None
    if top:
        vals = sorted({d["mm"] for d in top})
        total = max(vals) if max(vals) >= 6000 else sum(sorted(vals, reverse=True)[:2])
        if total >= 2000:
            scale = env_w / (total / 1000.0)
            source = f"top_mm={total}"
    if scale is None and side:
        vals = sorted({d["mm"] for d in side})
        total = max(vals) if max(vals) >= 8000 else sum(vals)
        if total >= 2000:
            scale = env_h / (total / 1000.0)
            source = f"side_mm={total}"
    if scale is None:
        # largest dimension as long side
        biggest = max(dims, key=lambda d: d["mm"])
        long_px = max(env_w, env_h)
        scale = long_px / (biggest["mm"] / 1000.0)
        source = f"largest_mm={biggest['mm']}"
    if scale and 15 <= scale <= 200:
        plan.scale_px_per_m = float(scale)
        plan.meta["scale_source"] = source
        plan.meta["dimension_candidates"] = dims[:20]
        _log(plan, "estimate_scale", {"scale": plan.scale_px_per_m, "source": source})
    else:
        _log(plan, "estimate_scale", {"ok": False, "raw": scale})
    return plan


def attach_openings(plan: RoomPlan, max_dist: float = 18.0) -> RoomPlan:
    for op in plan.openings:
        mx = (op.start.x + op.end.x) / 2
        my = (op.start.y + op.end.y) / 2
        best_id = None
        best_d = 1e18
        for w in plan.walls:
            # point-line distance
            x1, y1, x2, y2 = w.start.x, w.start.y, w.end.x, w.end.y
            denom = math.hypot(x2 - x1, y2 - y1) or 1.0
            dist = abs((y2 - y1) * mx - (x2 - x1) * my + x2 * y1 - y2 * x1) / denom
            if dist < best_d:
                best_d = dist
                best_id = w.id
        if best_id and best_d <= max_dist:
            op.wall_id = best_id
    _log(plan, "attach_openings")
    return plan


def infer_openings(
    plan: RoomPlan, image_bgr: np.ndarray | None = None
) -> RoomPlan:
    """Heuristic openings: gaps on exterior walls + short wall-adjacent strokes."""
    if not plan.walls:
        return plan
    openings: list[Opening] = []
    # use existing short openings if any, else synthesize window on longest exterior
    exteriors = [w for w in plan.walls if w.exterior] or sorted(
        plan.walls, key=lambda w: -w.length_px
    )[:4]
    for w in exteriors:
        if w.length_px < 80:
            continue
        # place a window at mid third on long exterior walls
        t0, t1 = 0.35, 0.65
        sx = w.start.x + (w.end.x - w.start.x) * t0
        sy = w.start.y + (w.end.y - w.start.y) * t0
        ex = w.start.x + (w.end.x - w.start.x) * t1
        ey = w.start.y + (w.end.y - w.start.y) * t1
        openings.append(
            Opening(
                id=_new_id("win"),
                type=OpeningType.WINDOW,
                start=Point2D(x=sx, y=sy),
                end=Point2D(x=ex, y=ey),
                wall_id=w.id,
            )
        )
    # door: shortest-ish exterior or left-side wall mid
    door_wall = min(exteriors, key=lambda w: abs(((w.start.x + w.end.x) / 2) - 0))
    # prefer westernmost vertical
    verts = [w for w in exteriors if abs(w.start.x - w.end.x) < abs(w.start.y - w.end.y)]
    if verts:
        door_wall = min(verts, key=lambda w: (w.start.x + w.end.x) / 2)
    t0, t1 = 0.45, 0.55
    openings.append(
        Opening(
            id=_new_id("door"),
            type=OpeningType.DOOR,
            start=Point2D(
                x=door_wall.start.x + (door_wall.end.x - door_wall.start.x) * t0,
                y=door_wall.start.y + (door_wall.end.y - door_wall.start.y) * t0,
            ),
            end=Point2D(
                x=door_wall.start.x + (door_wall.end.x - door_wall.start.x) * t1,
                y=door_wall.start.y + (door_wall.end.y - door_wall.start.y) * t1,
            ),
            wall_id=door_wall.id,
        )
    )
    # keep prior openings if denser
    if len(plan.openings) >= len(openings):
        attach_openings(plan)
        return plan
    plan.openings = openings
    _log(plan, "infer_openings", {"openings": len(openings)})
    return plan


REPAIR_REGISTRY: dict[str, Callable[..., RoomPlan]] = {
    "orthogonal_snap": lambda plan, **kw: orthogonal_snap(plan),
    "merge_collinear": lambda plan, **kw: merge_collinear(plan),
    "dedupe_walls": lambda plan, **kw: dedupe_walls(plan),
    "grid_snap": lambda plan, **kw: grid_snap(plan),
    "snap_junctions": lambda plan, **kw: snap_junctions(plan),
    "infer_rooms": lambda plan, image=None, **kw: infer_rooms(plan, image),
    "label_rooms": lambda plan, ocr_items=None, **kw: label_rooms_from_ocr(
        plan, ocr_items or []
    ),
    "estimate_scale": lambda plan, ocr_items=None, **kw: estimate_scale_from_ocr(
        plan, ocr_items or []
    ),
    "attach_openings": lambda plan, **kw: attach_openings(plan),
    "infer_openings": lambda plan, image=None, **kw: infer_openings(plan, image),
    "re_extract_walls": lambda plan, **kw: plan,  # handled by pipeline re-extract
}


def auto_repair(
    plan: RoomPlan,
    health: PlanHealth | None = None,
    *,
    image_bgr: np.ndarray | None = None,
    ocr_items: list[dict[str, Any]] | None = None,
    max_steps: int = 8,
) -> tuple[RoomPlan, PlanHealth, list[str]]:
    """Apply suggested repairs until pass or budget exhausted."""
    applied: list[str] = []
    health = health or assess_plan(plan)
    ocr_items = ocr_items or plan.meta.get("ocr_items") or []

    for _ in range(max_steps):
        if health.passed:
            break
        step_applied = False
        for name in health.suggested_repairs:
            if name == "re_extract_walls":
                continue
            fn = REPAIR_REGISTRY.get(name)
            if not fn:
                continue
            before = len(plan.walls), len(plan.rooms or []), plan.scale_px_per_m
            plan = fn(plan, image=image_bgr, ocr_items=ocr_items)
            if name in ("orthogonal_snap", "merge_collinear", "grid_snap"):
                plan = mark_exterior(plan)
            after = len(plan.walls), len(plan.rooms or []), plan.scale_px_per_m
            applied.append(name)
            step_applied = True
            # bump revision
            try:
                rev = int(str(plan.plan_revision).split("-")[-1])
            except ValueError:
                rev = 1
            plan.plan_revision = f"auto-{rev + 1}"
            if after != before:
                break
        if not step_applied:
            break
        health = assess_plan(plan)

    # always ensure rooms/exterior after loop if still missing
    if not plan.rooms:
        plan = infer_rooms(plan, image_bgr)
        applied.append("infer_rooms")
    plan = mark_exterior(plan)
    if plan.scale_px_per_m is None and ocr_items:
        plan = estimate_scale_from_ocr(plan, ocr_items)
        applied.append("estimate_scale")
    # warn-level cleanups that are safe even when already "passed"
    if plan.openings:
        before_orphan = sum(1 for o in plan.openings if not o.wall_id)
        plan = attach_openings(plan)
        # drop ultra-short noise openings not near any wall after attach
        cleaned = []
        for o in plan.openings:
            if o.length_px < 12 and not o.wall_id:
                continue
            cleaned.append(o)
        if len(cleaned) != len(plan.openings):
            plan.openings = cleaned
            applied.append("prune_noise_openings")
        if before_orphan:
            applied.append("attach_openings")
    if ocr_items and plan.rooms:
        plan = label_rooms_from_ocr(plan, ocr_items)
        if "label_rooms" not in applied:
            applied.append("label_rooms")
    health = assess_plan(plan)
    plan.meta["last_health"] = health.to_dict()
    return plan, health, applied
