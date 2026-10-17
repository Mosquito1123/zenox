"""Multi-room VR tour: per-room cameras → parallel render → nav hotspots.

Pipeline:
  RoomPlan.rooms
    → one camera pose per room (eye at center, look to best opening / south)
    → ThreadPool / ProcessPool rasterize beauty+depth
    → adjacency from door proximity between room polygons
    → tour.json + hotspots + simple HTML navigator
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cad_ai_lock.extrude import Scene3D, extrude_room_plan
from cad_ai_lock.geometry import OpeningType, Point2D, RoomPlan
from cad_ai_lock.perspective import _room_center


@dataclass
class RoomCamera:
    room_id: str
    name: str
    room_type: str
    eye_px: tuple[float, float]
    target_px: tuple[float, float]
    eye_m: list[float]
    target_m: list[float]
    height_m: float = 1.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "room_id": self.room_id,
            "name": self.name,
            "room_type": self.room_type,
            "eye_px": {"x": self.eye_px[0], "y": self.eye_px[1]},
            "target_px": {"x": self.target_px[0], "y": self.target_px[1]},
            "eye_m": self.eye_m,
            "target_m": self.target_m,
            "height_m": self.height_m,
        }


@dataclass
class Hotspot:
    from_room: str
    to_room: str
    label: str
    # normalized screen position hint for UI (0..1)
    u: float = 0.5
    v: float = 0.55

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_room,
            "to": self.to_room,
            "label": self.label,
            "u": self.u,
            "v": self.v,
        }


@dataclass
class TourBundle:
    rooms: list[dict[str, Any]] = field(default_factory=list)
    hotspots: list[dict[str, Any]] = field(default_factory=list)
    start_room: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_room": self.start_room,
            "rooms": self.rooms,
            "hotspots": self.hotspots,
            "meta": self.meta,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )


def _room_bbox(room: dict[str, Any]) -> tuple[float, float, float, float]:
    poly = room.get("polygon") or []
    if not poly:
        c = _room_center(room)
        return c[0] - 20, c[1] - 20, c[0] + 20, c[1] + 20
    xs = [p["x"] for p in poly]
    ys = [p["y"] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _px_to_m(
    x: float, y: float, origin: tuple[float, float], ppm: float, z: float = 0.0
) -> list[float]:
    return [(x - origin[0]) / ppm, (y - origin[1]) / ppm, z]


def build_room_cameras(
    plan: RoomPlan,
    scene: Scene3D | None = None,
    *,
    height_m: float = 1.5,
    min_area_px: int = 1200,
) -> list[RoomCamera]:
    """One inward-looking camera per usable room."""
    scene = scene or extrude_room_plan(plan)
    ppm = scene.scale_px_per_m
    origin = scene.origin_px
    rooms = plan.rooms or []
    if not rooms:
        # synthesize single envelope room
        x0, y0, x1, y1 = plan.bounding_box()
        rooms = [
            {
                "id": "whole",
                "name": "全屋",
                "type": "living",
                "area_px": int((x1 - x0) * (y1 - y0)),
                "polygon": [
                    {"x": x0, "y": y0},
                    {"x": x1, "y": y0},
                    {"x": x1, "y": y1},
                    {"x": x0, "y": y1},
                ],
                "center": {"x": (x0 + x1) / 2, "y": (y0 + y1) / 2},
            }
        ]

    cams: list[RoomCamera] = []
    for room in rooms:
        area = int(room.get("area_px") or 0)
        if area and area < min_area_px:
            continue
        rid = str(room.get("id") or f"room_{len(cams)}")
        name = str(room.get("name") or rid)
        rtype = str(room.get("type") or "room")
        cx, cy = _room_center(room)
        x0, y0, x1, y1 = _room_bbox(room)

        # Prefer looking toward a window/door on this room, else toward south (+Y)
        target = None
        best_d = 1e18
        for op in plan.openings:
            mx = (op.start.x + op.end.x) / 2
            my = (op.start.y + op.end.y) / 2
            # opening near room bbox
            if not (x0 - 25 <= mx <= x1 + 25 and y0 - 25 <= my <= y1 + 25):
                continue
            d = (mx - cx) ** 2 + (my - cy) ** 2
            # prefer windows for daylight view
            boost = 0.7 if op.type == OpeningType.WINDOW else 1.0
            d *= boost
            if d < best_d:
                best_d = d
                target = (mx, my)
        if target is None:
            # look toward longer axis outward
            if (y1 - y0) >= (x1 - x0):
                target = (cx, y1 - 5)  # south of room
            else:
                target = (x1 - 5, cy)

        # Pull eye slightly opposite of look direction (stand inside room)
        dx, dy = target[0] - cx, target[1] - cy
        dist = math.hypot(dx, dy) or 1.0
        eye = (cx - 0.15 * dx, cy - 0.15 * dy)
        # clamp eye inside bbox with margin
        mx0, my0 = x0 + 12, y0 + 12
        mx1, my1 = x1 - 12, y1 - 12
        if mx1 > mx0 and my1 > my0:
            eye = (
                float(np.clip(eye[0], mx0, mx1)),
                float(np.clip(eye[1], my0, my1)),
            )

        cams.append(
            RoomCamera(
                room_id=rid,
                name=name,
                room_type=rtype,
                eye_px=eye,
                target_px=target,
                eye_m=_px_to_m(eye[0], eye[1], origin, ppm, height_m),
                target_m=_px_to_m(target[0], target[1], origin, ppm, height_m * 0.92),
                height_m=height_m,
            )
        )
    return cams


def build_hotspots(plan: RoomPlan, cameras: list[RoomCamera]) -> list[Hotspot]:
    """Connect rooms whose polygons are close or share a door between them."""
    by_id = {c.room_id: c for c in cameras}
    rooms = {str(r.get("id")): r for r in (plan.rooms or []) if str(r.get("id")) in by_id}
    if len(rooms) < 2:
        return []

    def center(rid: str) -> tuple[float, float]:
        return _room_center(rooms[rid])

    def near(a: str, b: str, gap: float = 80.0) -> bool:
        ax0, ay0, ax1, ay1 = _room_bbox(rooms[a])
        bx0, by0, bx1, by1 = _room_bbox(rooms[b])
        # expand and test overlap
        ax0 -= gap
        ay0 -= gap
        ax1 += gap
        ay1 += gap
        return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)

    # Door-based edges preferred
    door_links: set[tuple[str, str]] = set()
    for op in plan.openings:
        if op.type != OpeningType.DOOR:
            continue
        mx = (op.start.x + op.end.x) / 2
        my = (op.start.y + op.end.y) / 2
        touching = []
        for rid, room in rooms.items():
            x0, y0, x1, y1 = _room_bbox(room)
            if x0 - 30 <= mx <= x1 + 30 and y0 - 30 <= my <= y1 + 30:
                touching.append(rid)
        if len(touching) >= 2:
            for i in range(len(touching)):
                for j in range(i + 1, len(touching)):
                    a, b = sorted([touching[i], touching[j]])
                    door_links.add((a, b))

    # Fallback: proximity graph
    ids = list(rooms.keys())
    prox: set[tuple[str, str]] = set()
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if near(a, b):
                prox.add(tuple(sorted([a, b])))

    edges = door_links or prox
    hotspots: list[Hotspot] = []
    for a, b in sorted(edges):
        ca, cb = center(a), center(b)
        # screen hint: hotspot on the side toward neighbor
        def side_uv(from_c, to_c):
            dx, dy = to_c[0] - from_c[0], to_c[1] - from_c[1]
            if abs(dx) > abs(dy):
                return (0.85 if dx > 0 else 0.15), 0.55
            return 0.5, (0.82 if dy > 0 else 0.25)

        ua, va = side_uv(ca, cb)
        ub, vb = side_uv(cb, ca)
        hotspots.append(
            Hotspot(a, b, label=by_id[b].name, u=ua, v=va)
        )
        hotspots.append(
            Hotspot(b, a, label=by_id[a].name, u=ub, v=vb)
        )
    return hotspots


def _render_one_room(
    plan: RoomPlan,
    scene: Scene3D,
    cam: RoomCamera,
    out_dir: Path,
    width: int,
    height: int,
) -> dict[str, Any]:
    from cad_ai_lock.geometry import CameraSpec
    from cad_ai_lock.render3d import render_controls_software

    # Encode eye/target into CameraSpec.notes for existing renderer path
    spec = CameraSpec(
        preset=f"room_{cam.room_id}",
        vanishing_point=Point2D(x=cam.target_px[0], y=cam.target_px[1]),
        horizon_y=cam.target_px[1],
        camera_height_m=cam.height_m,
        wall_height_m=plan.camera.wall_height_m if plan.camera else 2.8,
        notes=(
            f"living→room; eye=({cam.eye_px[0]:.1f},{cam.eye_px[1]:.1f}) "
            f"target=({cam.target_px[0]:.1f},{cam.target_px[1]:.1f})"
        ),
    )
    room_dir = out_dir / cam.room_id
    room_dir.mkdir(parents=True, exist_ok=True)
    bundle = render_controls_software(
        plan, scene=scene, camera=spec, width=width, height=height
    )
    # Override camera meta with exact room pose
    bundle.camera = cam.to_dict()
    paths = bundle.save(room_dir)
    # Convenience aliases for tour viewer
    beauty = room_dir / "beauty.png"
    depth = room_dir / "depth.png"
    cv2.imwrite(str(beauty), bundle.beauty)
    cv2.imwrite(str(depth), bundle.depth)
    return {
        "room_id": cam.room_id,
        "name": cam.name,
        "type": cam.room_type,
        "camera": cam.to_dict(),
        "beauty": str(beauty),
        "depth": str(depth),
        "paths": paths,
    }


def render_tour_parallel(
    plan: RoomPlan,
    out_dir: str | Path,
    *,
    scene: Scene3D | None = None,
    max_workers: int | None = None,
    width: int = 720,
    height: int = 1280,
    min_area_px: int = 1200,
) -> TourBundle:
    """Build per-room cameras, parallel-render, stitch hotspots + HTML."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = scene or extrude_room_plan(plan)
    cameras = build_room_cameras(plan, scene, min_area_px=min_area_px)
    if not cameras:
        # fallback single living pose
        from cad_ai_lock.perspective import place_living_camera

        live = place_living_camera(plan)
        x0, y0, x1, y1 = plan.bounding_box()
        cameras = [
            RoomCamera(
                room_id="living",
                name="客厅",
                room_type="living",
                eye_px=((x0 + x1) / 2, y0 + (y1 - y0) * 0.45),
                target_px=(
                    (live.vanishing_point.x if live.vanishing_point else (x0 + x1) / 2),
                    (live.vanishing_point.y if live.vanishing_point else y1 - 5),
                ),
                eye_m=[0, 0, 1.5],
                target_m=[0, 1, 1.4],
            )
        ]
        # fix meters via scene
        cameras = build_room_cameras(
            plan,
            scene,
            min_area_px=0,
        ) or cameras

    workers = max_workers or min(8, max(1, len(cameras)))
    room_results: list[dict[str, Any]] = []
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(
                _render_one_room, plan, scene, cam, out_dir, width, height
            ): cam.room_id
            for cam in cameras
        }
        for fut in as_completed(futs):
            rid = futs[fut]
            try:
                room_results.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{rid}: {exc}")

    room_results.sort(key=lambda r: r["room_id"])
    hotspots = build_hotspots(plan, cameras)

    # Prefer living / largest as start
    start = ""
    for r in room_results:
        if r.get("type") == "living" or "客厅" in str(r.get("name")):
            start = r["room_id"]
            break
    if not start and room_results:
        start = room_results[0]["room_id"]

    tour = TourBundle(
        rooms=room_results,
        hotspots=[h.to_dict() for h in hotspots],
        start_room=start,
        meta={
            "workers": workers,
            "rooms_rendered": len(room_results),
            "errors": errors,
            "width": width,
            "height": height,
        },
    )
    tour.save(out_dir / "tour.json")
    _write_tour_html(tour, out_dir / "tour.html")
    # contact sheet
    _write_contact_sheet(room_results, out_dir / "tour_contact.png")
    return tour


def _write_contact_sheet(rooms: list[dict[str, Any]], path: Path) -> None:
    imgs = []
    labels = []
    for r in rooms:
        p = r.get("beauty")
        if not p:
            continue
        im = cv2.imread(str(p))
        if im is None:
            continue
        im = cv2.resize(im, (180, 320))
        cv2.putText(
            im,
            str(r.get("name") or r["room_id"])[:8],
            (8, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        imgs.append(im)
        labels.append(r["room_id"])
    if not imgs:
        return
    sheet = np.hstack(imgs)
    cv2.imwrite(str(path), sheet)


def _write_tour_html(tour: TourBundle, path: Path) -> None:
    data = json.dumps(tour.to_dict(), ensure_ascii=False)
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>CAD AI Lock · Room Tour</title>
<style>
  :root {{ color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif; }}
  body {{ margin:0; background:#0b0d10; color:#e8eaed; }}
  header {{ padding:12px 16px; border-bottom:1px solid #222; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }}
  #view {{ position:relative; width:min(100vw, 720px); margin:0 auto; background:#000; }}
  #view img {{ width:100%; height:auto; display:block; }}
  .hotspot {{
    position:absolute; transform:translate(-50%,-50%);
    background:rgba(20,140,255,.85); color:#fff; border:0; border-radius:999px;
    padding:8px 12px; cursor:pointer; font-size:13px; white-space:nowrap;
  }}
  .hotspot:hover {{ background:#1a9fff; }}
  #rooms {{ display:flex; gap:8px; flex-wrap:wrap; padding:12px 16px; }}
  #rooms button {{ background:#1c2128; color:#eee; border:1px solid #333; border-radius:8px; padding:8px 10px; cursor:pointer; }}
  #rooms button.active {{ border-color:#3b82f6; background:#132033; }}
  #meta {{ opacity:.7; font-size:12px; padding:0 16px 16px; }}
</style>
</head>
<body>
<header>
  <strong>分房间 VR 漫游</strong>
  <span id="title"></span>
</header>
<div id="view">
  <img id="frame" alt="room view"/>
  <div id="hots"></div>
</div>
<div id="rooms"></div>
<pre id="meta"></pre>
<script>
const TOUR = {data};
let cur = TOUR.start_room || (TOUR.rooms[0] && TOUR.rooms[0].room_id);
const byId = Object.fromEntries(TOUR.rooms.map(r => [r.room_id, r]));
function rel(p) {{
  // paths are absolute in JSON; use basename under same folder as tour.html
  if (!p) return '';
  const parts = p.split(/[/\\\\]/);
  const room = parts[parts.length-2];
  const file = parts[parts.length-1];
  return room + '/' + file;
}}
function go(id) {{
  cur = id;
  const r = byId[id];
  if (!r) return;
  document.getElementById('title').textContent = (r.name || id) + ' · ' + (r.type || '');
  document.getElementById('frame').src = rel(r.beauty);
  document.getElementById('meta').textContent = JSON.stringify(r.camera, null, 2);
  document.querySelectorAll('#rooms button').forEach(b => b.classList.toggle('active', b.dataset.id===id));
  const hots = document.getElementById('hots');
  hots.innerHTML = '';
  TOUR.hotspots.filter(h => h.from === id).forEach(h => {{
    const btn = document.createElement('button');
    btn.className = 'hotspot';
    btn.textContent = '→ ' + h.label;
    btn.style.left = (h.u*100) + '%';
    btn.style.top = (h.v*100) + '%';
    btn.onclick = () => go(h.to);
    hots.appendChild(btn);
  }});
}}
const roomBar = document.getElementById('rooms');
TOUR.rooms.forEach(r => {{
  const b = document.createElement('button');
  b.textContent = r.name || r.room_id;
  b.dataset.id = r.room_id;
  b.onclick = () => go(r.room_id);
  roomBar.appendChild(b);
}});
go(cur);
</script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
