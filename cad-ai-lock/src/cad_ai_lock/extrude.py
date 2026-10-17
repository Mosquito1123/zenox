"""RoomPlan → metric 3D mesh → glTF.

This is the commercial geometry kernel: walls/openings become real solids in
meters. Downstream VR/controls must come from this mesh (or Blender/Unity
import of the glTF), not from free-form image generation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cad_ai_lock.geometry import Opening, OpeningType, RoomPlan, Wall


@dataclass
class MeshPart:
    name: str
    kind: str  # wall | opening_door | opening_window | floor | furniture
    vertices: np.ndarray  # (N,3) float32 meters
    faces: np.ndarray  # (M,3) int32
    color: tuple[float, float, float] = (0.8, 0.8, 0.8)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scene3D:
    parts: list[MeshPart]
    scale_px_per_m: float
    origin_px: tuple[float, float]
    wall_height_m: float
    meta: dict[str, Any] = field(default_factory=dict)

    def to_trimesh_scene(self):
        import trimesh

        scene = trimesh.Scene()
        for part in self.parts:
            if len(part.faces) == 0 or len(part.vertices) == 0:
                continue
            mesh = trimesh.Trimesh(
                vertices=part.vertices.copy(),
                faces=part.faces.copy(),
                process=False,
            )
            rgba = np.array(
                [
                    int(part.color[0] * 255),
                    int(part.color[1] * 255),
                    int(part.color[2] * 255),
                    255,
                ],
                dtype=np.uint8,
            )
            mesh.visual.vertex_colors = np.tile(rgba, (len(mesh.vertices), 1))
            scene.add_geometry(mesh, node_name=part.name, geom_name=part.name)
        return scene

    def export_gltf(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        scene = self.to_trimesh_scene()
        # trimesh exports glb/gltf
        if path.suffix.lower() not in (".glb", ".gltf"):
            path = path.with_suffix(".glb")
        scene.export(str(path))
        meta_path = path.with_suffix(".json")
        meta_path.write_text(
            json.dumps(
                {
                    "scale_px_per_m": self.scale_px_per_m,
                    "origin_px": self.origin_px,
                    "wall_height_m": self.wall_height_m,
                    "parts": [
                        {
                            "name": p.name,
                            "kind": p.kind,
                            "verts": int(len(p.vertices)),
                            "faces": int(len(p.faces)),
                            "meta": p.meta,
                        }
                        for p in self.parts
                    ],
                    "meta": self.meta,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    def merged_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Concatenate all parts → vertices, faces, face_part_index, part_kinds."""
        verts: list[np.ndarray] = []
        faces: list[np.ndarray] = []
        face_kind: list[str] = []
        offset = 0
        for p in self.parts:
            if len(p.faces) == 0:
                continue
            verts.append(p.vertices)
            faces.append(p.faces + offset)
            face_kind.extend([p.kind] * len(p.faces))
            offset += len(p.vertices)
        if not verts:
            return (
                np.zeros((0, 3), np.float32),
                np.zeros((0, 3), np.int32),
                np.zeros((0,), np.int32),
                [],
            )
        V = np.vstack(verts).astype(np.float32)
        F = np.vstack(faces).astype(np.int32)
        kinds = [p.kind for p in self.parts]
        # map face → kind id
        kind_to_id = {k: i for i, k in enumerate(sorted(set(face_kind)))}
        face_ids = np.array([kind_to_id[k] for k in face_kind], dtype=np.int32)
        return V, F, face_ids, [k for k, _ in sorted(kind_to_id.items(), key=lambda x: x[1])]


def _ppm(plan: RoomPlan) -> float:
    if plan.scale_px_per_m and plan.scale_px_per_m > 0:
        return float(plan.scale_px_per_m)
    x0, y0, x1, y1 = plan.bounding_box()
    # assume long side ≈ 12m for apartments if unknown
    return max(x1 - x0, y1 - y0, 1.0) / 12.0


def _px_to_m(
    x: float, y: float, origin: tuple[float, float], ppm: float
) -> tuple[float, float]:
    # Plan Y-down → world Z-up, Y forward (south = +Y)
    return ((x - origin[0]) / ppm, (y - origin[1]) / ppm)


def _box_from_segment(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    z0: float,
    z1: float,
    thickness: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extrude a 2D segment into a vertical box (meters)."""
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy) or 1e-6
    ux, uy = dx / length, dy / length
    # perpendicular in XY
    px, py = -uy, ux
    hx, hy = px * thickness * 0.5, py * thickness * 0.5
    # 8 corners
    c = [
        (x0 - hx, y0 - hy, z0),
        (x0 + hx, y0 + hy, z0),
        (x1 + hx, y1 + hy, z0),
        (x1 - hx, y1 - hy, z0),
        (x0 - hx, y0 - hy, z1),
        (x0 + hx, y0 + hy, z1),
        (x1 + hx, y1 + hy, z1),
        (x1 - hx, y1 - hy, z1),
    ]
    V = np.array(c, dtype=np.float32)
    F = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=np.int32,
    )
    return V, F


def _quad(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    z0: float,
    z1: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vertical opening quad (thin)."""
    V = np.array(
        [
            [x0, y0, z0],
            [x1, y1, z0],
            [x1, y1, z1],
            [x0, y0, z1],
        ],
        dtype=np.float32,
    )
    F = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return V, F


def _floor_slab(
    x0: float, y0: float, x1: float, y1: float, z: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    V = np.array(
        [
            [x0, y0, z],
            [x1, y0, z],
            [x1, y1, z],
            [x0, y1, z],
        ],
        dtype=np.float32,
    )
    F = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return V, F


def extrude_room_plan(
    plan: RoomPlan,
    *,
    wall_height_m: float | None = None,
    wall_thickness_m: float = 0.12,
    include_floor: bool = True,
    include_openings: bool = True,
) -> Scene3D:
    """Extrude locked RoomPlan into a metric Scene3D."""
    ppm = _ppm(plan)
    x0, y0, x1, y1 = plan.bounding_box()
    origin = (x0, y0)
    h = wall_height_m or (
        plan.camera.wall_height_m if plan.camera else 2.8
    )
    parts: list[MeshPart] = []

    if include_floor:
        fx0, fy0 = _px_to_m(x0, y0, origin, ppm)
        fx1, fy1 = _px_to_m(x1, y1, origin, ppm)
        V, F = _floor_slab(fx0, fy0, fx1, fy1, 0.0)
        parts.append(
            MeshPart(
                name="floor",
                kind="floor",
                vertices=V,
                faces=F,
                color=(0.75, 0.72, 0.68),
            )
        )

    for i, wall in enumerate(plan.walls):
        ax, ay = _px_to_m(wall.start.x, wall.start.y, origin, ppm)
        bx, by = _px_to_m(wall.end.x, wall.end.y, origin, ppm)
        thick = max(wall_thickness_m, (wall.thickness_px / ppm) * 0.5)
        V, F = _box_from_segment(ax, ay, bx, by, 0.0, h, thick)
        parts.append(
            MeshPart(
                name=f"wall_{wall.id or i}",
                kind="wall",
                vertices=V,
                faces=F,
                color=(0.92, 0.92, 0.92) if not wall.exterior else (0.55, 0.55, 0.55),
                meta={"wall_id": wall.id, "exterior": wall.exterior},
            )
        )

    if include_openings:
        for i, op in enumerate(plan.openings):
            ax, ay = _px_to_m(op.start.x, op.start.y, origin, ppm)
            bx, by = _px_to_m(op.end.x, op.end.y, origin, ppm)
            if op.type == OpeningType.DOOR:
                z0, z1 = 0.0, min(h * 0.75, 2.1)
                color = (0.2, 0.85, 0.95)
                kind = "opening_door"
            elif op.type == OpeningType.WINDOW:
                z0, z1 = h * 0.35, h * 0.85
                color = (0.95, 0.7, 0.15)
                kind = "opening_window"
            else:
                z0, z1 = 0.0, h * 0.9
                color = (0.7, 0.7, 1.0)
                kind = "opening_pass"
            V, F = _quad(ax, ay, bx, by, z0, z1)
            parts.append(
                MeshPart(
                    name=f"open_{op.id or i}",
                    kind=kind,
                    vertices=V,
                    faces=F,
                    color=color,
                    meta={"opening_id": op.id, "type": op.type.value},
                )
            )

    return Scene3D(
        parts=parts,
        scale_px_per_m=ppm,
        origin_px=origin,
        wall_height_m=h,
        meta={
            "source_path": plan.source_path,
            "plan_revision": plan.plan_revision,
            "walls": len(plan.walls),
            "openings": len(plan.openings),
            "rooms": len(plan.rooms or []),
        },
    )


def extrude_room_shell(
    plan: RoomPlan,
    room: dict[str, Any],
    *,
    wall_height_m: float | None = None,
    base_scene: Scene3D | None = None,
) -> Scene3D:
    """Clean per-room shell (floor + wall quads + nearby openings) for readable tour frames.

    Unlike the full plan extrusion (noisy overlapping wall slabs), this builds one
    open room box from the room polygon so a camera inside sees floor + walls.
    """
    ppm = (base_scene.scale_px_per_m if base_scene else None) or _ppm(plan)
    origin = (base_scene.origin_px if base_scene else None) or (
        plan.bounding_box()[0],
        plan.bounding_box()[1],
    )
    h = wall_height_m or (
        (base_scene.wall_height_m if base_scene else None)
        or (plan.camera.wall_height_m if plan.camera else 2.8)
    )
    poly = list(room.get("polygon") or [])
    if len(poly) < 3:
        x0 = min(p.get("x", 0) for p in poly) if poly else 0
        y0 = min(p.get("y", 0) for p in poly) if poly else 0
        x1 = max(p.get("x", 0) for p in poly) if poly else 100
        y1 = max(p.get("y", 0) for p in poly) if poly else 100
        c = room.get("center") or {}
        if c:
            x0, y0 = c.get("x", 50) - 40, c.get("y", 50) - 40
            x1, y1 = c.get("x", 50) + 40, c.get("y", 50) + 40
        poly = [
            {"x": x0, "y": y0},
            {"x": x1, "y": y0},
            {"x": x1, "y": y1},
            {"x": x0, "y": y1},
        ]

    pts_m = [_px_to_m(float(p["x"]), float(p["y"]), origin, ppm) for p in poly]
    parts: list[MeshPart] = []

    # Floor: fan triangulation around centroid
    cx = sum(p[0] for p in pts_m) / len(pts_m)
    cy = sum(p[1] for p in pts_m) / len(pts_m)
    floor_v = [(cx, cy, 0.0)] + [(p[0], p[1], 0.0) for p in pts_m]
    # close ring
    floor_v.append((pts_m[0][0], pts_m[0][1], 0.0))
    Vf = np.array(floor_v, dtype=np.float32)
    Ff = []
    for i in range(1, len(pts_m) + 1):
        Ff.append([0, i, i + 1 if i < len(pts_m) else 1])
    # fix last: use explicit close vertex index
    Ff = []
    n = len(pts_m)
    for i in range(n):
        i1 = i + 1
        i2 = (i + 1) % n + 1
        Ff.append([0, i1, i2])
    parts.append(
        MeshPart(
            name=f"floor_{room.get('id')}",
            kind="floor",
            vertices=Vf[: n + 1],
            faces=np.array(Ff, dtype=np.int32),
            color=(0.78, 0.70, 0.58),
        )
    )

    # Ceiling (helps frame the room)
    ceil_v = [(cx, cy, h)] + [(p[0], p[1], h) for p in pts_m]
    Vc = np.array(ceil_v, dtype=np.float32)
    Fc = []
    for i in range(n):
        i1 = i + 1
        i2 = (i + 1) % n + 1
        # flip winding so underside faces camera from below
        Fc.append([0, i2, i1])
    parts.append(
        MeshPart(
            name=f"ceil_{room.get('id')}",
            kind="wall",
            vertices=Vc,
            faces=np.array(Fc, dtype=np.int32),
            color=(0.95, 0.95, 0.93),
        )
    )

    # Wall quads along polygon edges (thin inward faces — readable from inside)
    for i in range(n):
        ax, ay = pts_m[i]
        bx, by = pts_m[(i + 1) % n]
        V, F = _quad(ax, ay, bx, by, 0.0, h)
        # also add reverse face so wall is visible from either side
        F2 = np.vstack([F, F[:, ::-1]])
        parts.append(
            MeshPart(
                name=f"shell_wall_{i}",
                kind="wall",
                vertices=V,
                faces=F2.astype(np.int32),
                color=(0.88, 0.88, 0.86),
            )
        )

    # Nearby openings as colored quads
    xs = [float(p["x"]) for p in poly]
    ys = [float(p["y"]) for p in poly]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    for i, op in enumerate(plan.openings):
        mx = (op.start.x + op.end.x) / 2
        my = (op.start.y + op.end.y) / 2
        if not (x0 - 40 <= mx <= x1 + 40 and y0 - 40 <= my <= y1 + 40):
            continue
        ax, ay = _px_to_m(op.start.x, op.start.y, origin, ppm)
        bx, by = _px_to_m(op.end.x, op.end.y, origin, ppm)
        if op.type == OpeningType.DOOR:
            z0, z1 = 0.0, min(h * 0.75, 2.1)
            color = (0.15, 0.75, 0.95)
            kind = "opening_door"
        elif op.type == OpeningType.WINDOW:
            z0, z1 = h * 0.35, h * 0.85
            color = (0.95, 0.75, 0.2)
            kind = "opening_window"
        else:
            z0, z1 = 0.0, h * 0.9
            color = (0.7, 0.7, 1.0)
            kind = "opening_pass"
        V, F = _quad(ax, ay, bx, by, z0, z1)
        F2 = np.vstack([F, F[:, ::-1]])
        parts.append(
            MeshPart(
                name=f"shell_open_{op.id or i}",
                kind=kind,
                vertices=V,
                faces=F2.astype(np.int32),
                color=color,
            )
        )

    return Scene3D(
        parts=parts,
        scale_px_per_m=ppm,
        origin_px=origin,
        wall_height_m=h,
        meta={
            "shell_of": room.get("id"),
            "room_name": room.get("name"),
            "verts": sum(len(p.vertices) for p in parts),
        },
    )


def export_plan_gltf(
    plan: RoomPlan,
    out_path: str | Path,
    **kwargs: Any,
) -> tuple[Path, Scene3D]:
    scene = extrude_room_plan(plan, **kwargs)
    path = scene.export_gltf(out_path)
    return path, scene
