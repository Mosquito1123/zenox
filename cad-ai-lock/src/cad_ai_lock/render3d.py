"""Rasterize Scene3D → depth / normal / segmentation / lineart control maps.

Default path is a pure-NumPy software rasterizer (no GPU/Blender required).
Optional Blender headless script is invoked when ``blender`` is on PATH and
``prefer_blender=True``.

Commercial rule: these maps are ground truth for AI conditioning — not guesses.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cad_ai_lock.extrude import Scene3D, extrude_room_plan
from cad_ai_lock.geometry import CameraSpec, RoomPlan
from cad_ai_lock.perspective import place_living_camera


@dataclass
class RenderBundle:
    depth: np.ndarray  # uint8 HxWx3
    normal: np.ndarray
    segmentation: np.ndarray
    lineart: np.ndarray
    beauty: np.ndarray  # simple shaded preview
    camera: dict[str, Any]
    meta: dict[str, Any]

    def save(self, out_dir: str | Path) -> dict[str, str]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "depth": str(out_dir / "ctrl3d_depth.png"),
            "normal": str(out_dir / "ctrl3d_normal.png"),
            "segmentation": str(out_dir / "ctrl3d_segmentation.png"),
            "lineart": str(out_dir / "ctrl3d_lineart.png"),
            "beauty": str(out_dir / "ctrl3d_beauty.png"),
            "camera": str(out_dir / "ctrl3d_camera.json"),
        }
        cv2.imwrite(paths["depth"], self.depth)
        cv2.imwrite(paths["normal"], self.normal)
        cv2.imwrite(paths["segmentation"], self.segmentation)
        cv2.imwrite(paths["lineart"], self.lineart)
        cv2.imwrite(paths["beauty"], self.beauty)
        Path(paths["camera"]).write_text(
            json.dumps({"camera": self.camera, "meta": self.meta}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return paths


def _camera_eye_target_from_plan(
    plan: RoomPlan, scene: Scene3D, camera: CameraSpec | None
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cam = camera or plan.camera or place_living_camera(plan)
    notes = cam.notes or ""
    ppm = scene.scale_px_per_m
    ox, oy = scene.origin_px

    def px_to_world(x: float, y: float, z: float = 0.0) -> np.ndarray:
        return np.array([(x - ox) / ppm, (y - oy) / ppm, z], dtype=np.float64)

    # defaults from envelope — stand mid-living, look south to balcony band
    x0, y0, x1, y1 = plan.bounding_box()
    eye_px = ((x0 + x1) / 2, y0 + (y1 - y0) * 0.42)
    tgt_px = ((x0 + x1) / 2, y1 - 8)
    per_room = str(cam.preset or "").startswith("room_")
    if "eye=(" in notes:
        try:
            eye_px = tuple(map(float, notes.split("eye=(")[1].split(")")[0].split(",")))  # type: ignore
        except Exception:
            pass
    if "target=(" in notes:
        try:
            tgt_px = tuple(map(float, notes.split("target=(")[1].split(")")[0].split(",")))  # type: ignore
        except Exception:
            pass
    if cam.vanishing_point is not None:
        tgt_px = (cam.vanishing_point.x, cam.vanishing_point.y)

    eye = px_to_world(eye_px[0], eye_px[1], cam.camera_height_m)
    target = px_to_world(tgt_px[0], tgt_px[1], cam.camera_height_m * 0.9)

    # Pull eye backward along view so we are not embedded in a wall slab.
    # Per-room tour cams already sit inside the room — use a short pull-back.
    view = target - eye
    vn = np.linalg.norm(view) or 1.0
    view = view / vn
    # Per-room cams already sit inside the room box — tiny pull only.
    # Whole-plan living cam needs a larger pull so the eye is not in a slab.
    pull = 0.05 if per_room else 1.2
    eye = eye - view * pull
    if not per_room:
        xmin, ymin = px_to_world(x0 + 20, y0 + 20)[:2]
        xmax, ymax = px_to_world(x1 - 20, y1 - 20)[:2]
        eye[0] = float(np.clip(eye[0], min(xmin, xmax) + 0.3, max(xmin, xmax) - 0.3))
        eye[1] = float(np.clip(eye[1], min(ymin, ymax) + 0.3, max(ymin, ymax) - 0.3))
    eye[2] = cam.camera_height_m

    meta = {
        "preset": cam.preset,
        "eye_px": {"x": eye_px[0], "y": eye_px[1]},
        "target_px": {"x": tgt_px[0], "y": tgt_px[1]},
        "eye_m": eye.tolist(),
        "target_m": target.tolist(),
        "wall_height_m": cam.wall_height_m,
        "notes": cam.notes,
    }
    return eye, target, meta


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray | None = None):
    up = np.array([0.0, 0.0, 1.0]) if up is None else up
    f = target - eye
    f = f / (np.linalg.norm(f) or 1.0)
    s = np.cross(f, up)
    if np.linalg.norm(s) < 1e-6:
        s = np.cross(f, np.array([0.0, 1.0, 0.0]))
    s = s / (np.linalg.norm(s) or 1.0)
    u = np.cross(s, f)
    # world → camera: x right, y up, z forward
    R = np.stack([s, u, f], axis=0)
    return R


def _project_vertices(
    V: np.ndarray,
    eye: np.ndarray,
    R: np.ndarray,
    *,
    width: int,
    height: int,
    fov_deg: float = 70.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return screen xy (N,2) and camera-forward depth (N,)."""
    rel = (V - eye) @ R.T  # (N,3) in camera frame: x,y,z_forward
    z = rel[:, 2]
    f = 0.5 * height / math.tan(math.radians(fov_deg) * 0.5)
    # avoid div0
    z_safe = np.where(z > 0.05, z, np.nan)
    sx = width * 0.5 + f * (rel[:, 0] / z_safe)
    sy = height * 0.5 - f * (rel[:, 1] / z_safe)
    return np.stack([sx, sy], axis=1), z


def _rasterize(
    V: np.ndarray,
    F: np.ndarray,
    face_ids: np.ndarray,
    kind_names: list[str],
    eye: np.ndarray,
    R: np.ndarray,
    *,
    width: int,
    height: int,
    fov_deg: float = 70.0,
) -> dict[str, np.ndarray]:
    """CPU barycentric rasterizer."""
    screen, depth_v = _project_vertices(V, eye, R, width=width, height=height, fov_deg=fov_deg)
    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    idbuf = np.full((height, width), -1, dtype=np.int32)
    nbuf = np.zeros((height, width, 3), dtype=np.float32)

    # kind colors for segmentation / beauty (BGR)
    palette = {
        "wall": (210, 210, 205),
        "floor": (90, 140, 180),  # warm wood-ish
        "opening_door": (40, 220, 255),
        "opening_window": (40, 190, 255),
        "opening_pass": (200, 180, 255),
        "furniture": (100, 200, 100),
    }

    for fi, face in enumerate(F):
        i0, i1, i2 = map(int, face)
        pts = screen[[i0, i1, i2]]
        zs = depth_v[[i0, i1, i2]]
        if np.any(~np.isfinite(pts)) or np.any(~np.isfinite(zs)) or np.any(zs <= 0.05):
            continue
        # face normal in world
        v0, v1, v2 = V[i0], V[i1], V[i2]
        n = np.cross(v1 - v0, v2 - v0)
        nn = np.linalg.norm(n)
        if nn < 1e-8:
            continue
        n = n / nn
        # backface in camera: if normal · view < 0 roughly
        view = eye - v0
        # keep both sides for walls thin; skip only if nearly edge-on
        minx = int(max(0, math.floor(pts[:, 0].min())))
        maxx = int(min(width - 1, math.ceil(pts[:, 0].max())))
        miny = int(max(0, math.floor(pts[:, 1].min())))
        maxy = int(min(height - 1, math.ceil(pts[:, 1].max())))
        if minx >= maxx or miny >= maxy:
            continue
        x0, y0 = pts[0]
        x1, y1 = pts[1]
        x2, y2 = pts[2]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-8:
            continue
        kid = int(face_ids[fi]) if fi < len(face_ids) else 0
        xs = np.arange(minx, maxx + 1)
        ys = np.arange(miny, maxy + 1)
        if xs.size == 0 or ys.size == 0:
            continue
        XX, YY = np.meshgrid(xs, ys)
        w0 = ((y1 - y2) * (XX - x2) + (x2 - x1) * (YY - y2)) / denom
        w1 = ((y2 - y0) * (XX - x2) + (x0 - x2) * (YY - y0)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        z = w0 * zs[0] + w1 * zs[1] + w2 * zs[2]
        closer = inside & (z > 0.05) & (z < zbuf[miny : maxy + 1, minx : maxx + 1])
        if not closer.any():
            continue
        patch_z = zbuf[miny : maxy + 1, minx : maxx + 1]
        patch_id = idbuf[miny : maxy + 1, minx : maxx + 1]
        patch_n = nbuf[miny : maxy + 1, minx : maxx + 1]
        patch_z[closer] = z[closer]
        patch_id[closer] = kid
        patch_n[closer] = n

    # depth visualization
    valid = np.isfinite(zbuf) & (zbuf < np.inf)
    depth_u8 = np.zeros((height, width), dtype=np.uint8)
    if valid.any():
        d = zbuf.copy()
        d[~valid] = 0
        dmin, dmax = float(d[valid].min()), float(d[valid].max())
        if dmax > dmin:
            norm = (dmax - d) / (dmax - dmin)
            depth_u8[valid] = (np.clip(norm[valid], 0, 1) * 255).astype(np.uint8)
    depth_rgb = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)

    # normal map (camera-ish encoding)
    normal_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    # encode world normal to RGB
    n_enc = (nbuf + 1.0) * 0.5
    normal_rgb[valid] = (np.clip(n_enc[valid], 0, 1) * 255).astype(np.uint8)

    # segmentation
    seg = np.zeros((height, width, 3), dtype=np.uint8)
    for kid, name in enumerate(kind_names):
        color = palette.get(name, (160, 160, 160))
        mask = idbuf == kid
        seg[mask] = color

    # lineart from depth discontinuities + normal edges
    edges = cv2.Canny(depth_u8, 40, 120)
    ngray = cv2.cvtColor(normal_rgb, cv2.COLOR_BGR2GRAY)
    edges2 = cv2.Canny(ngray, 40, 120)
    line = cv2.bitwise_or(edges, edges2)
    lineart = cv2.cvtColor(line, cv2.COLOR_GRAY2BGR)

    # beauty: ambient + lambert on a soft sky backdrop (readable room box)
    light = np.array([0.35, -0.15, 0.92], dtype=np.float64)
    light = light / np.linalg.norm(light)
    lambert = np.clip((nbuf * light).sum(axis=2), 0.0, 1.0)
    shade = 0.42 + 0.58 * lambert  # lift shadows so walls/floor are visible
    sky = np.array([168, 178, 188], dtype=np.float32)  # BGR soft gray-blue
    beauty = np.zeros((height, width, 3), dtype=np.uint8)
    beauty[:] = sky.astype(np.uint8)
    base = seg.astype(np.float32)
    for c in range(3):
        ch = base[:, :, c] * shade
        beauty[:, :, c][valid] = np.clip(ch[valid], 0, 255).astype(np.uint8)

    return {
        "depth": depth_rgb,
        "normal": normal_rgb,
        "segmentation": seg,
        "lineart": lineart,
        "beauty": beauty,
        "zbuf": zbuf,
        "idbuf": idbuf,
    }


def _filter_faces_by_bbox_m(
    V: np.ndarray,
    F: np.ndarray,
    face_ids: np.ndarray,
    bbox_m: tuple[float, float, float, float],
    *,
    pad_m: float = 0.55,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep faces whose centroid lies inside an expanded XY bbox (meters)."""
    if len(F) == 0:
        return F, face_ids
    x0, y0, x1, y1 = bbox_m
    x0 -= pad_m
    y0 -= pad_m
    x1 += pad_m
    y1 += pad_m
    cents = V[F].mean(axis=1)  # (M,3)
    keep = (
        (cents[:, 0] >= x0)
        & (cents[:, 0] <= x1)
        & (cents[:, 1] >= y0)
        & (cents[:, 1] <= y1)
    )
    if not keep.any():
        return F, face_ids
    return F[keep], face_ids[keep]


def render_controls_software(
    plan: RoomPlan,
    *,
    scene: Scene3D | None = None,
    camera: CameraSpec | None = None,
    width: int = 720,
    height: int = 1280,
    fov_deg: float = 70.0,
    clip_bbox_px: tuple[float, float, float, float] | None = None,
) -> RenderBundle:
    scene = scene or extrude_room_plan(plan)
    V, F, face_ids, kind_names = scene.merged_arrays()
    eye, target, cam_meta = _camera_eye_target_from_plan(plan, scene, camera)
    if clip_bbox_px is not None:
        ox, oy = scene.origin_px
        ppm = scene.scale_px_per_m
        x0, y0, x1, y1 = clip_bbox_px
        bbox_m = (
            (x0 - ox) / ppm,
            (y0 - oy) / ppm,
            (x1 - ox) / ppm,
            (y1 - oy) / ppm,
        )
        F, face_ids = _filter_faces_by_bbox_m(V, F, face_ids, bbox_m)
        cam_meta = {**cam_meta, "clip_bbox_px": {
            "x0": x0, "y0": y0, "x1": x1, "y1": y1
        }}
    R = _look_at(eye, target)
    maps = _rasterize(
        V,
        F,
        face_ids,
        kind_names,
        eye,
        R,
        width=width,
        height=height,
        fov_deg=fov_deg,
    )
    return RenderBundle(
        depth=maps["depth"],
        normal=maps["normal"],
        segmentation=maps["segmentation"],
        lineart=maps["lineart"],
        beauty=maps["beauty"],
        camera=cam_meta,
        meta={
            "renderer": "numpy_software",
            "width": width,
            "height": height,
            "fov_deg": fov_deg,
            "kinds": kind_names,
            "parts": len(scene.parts),
            "faces_rendered": int(len(F)),
        },
    )


_BLENDER_SCRIPT = r'''
import bpy, json, sys, math
from mathutils import Vector, Matrix

argv = sys.argv[sys.argv.index("--")+1:]
gltf_path, out_dir, cam_json, width, height = argv[0], argv[1], argv[2], int(argv[3]), int(argv[4])

# reset
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=gltf_path)
cam = json.load(open(cam_json))
eye = Vector(cam["eye_m"])
target = Vector(cam["target_m"])

# camera
bpy.ops.object.camera_add()
camera = bpy.context.object
camera.location = eye
direction = target - eye
rot = direction.to_track_quat('-Z', 'Y')
camera.rotation_euler = rot.to_euler()
camera.data.lens_unit = 'FOV'
camera.data.angle = math.radians(70)
bpy.context.scene.camera = camera

scene = bpy.context.scene
scene.render.resolution_x = width
scene.render.resolution_y = height
scene.render.engine = 'CYCLES'
scene.cycles.samples = 8
scene.cycles.use_denoising = False

# depth
scene.use_nodes = True
nt = scene.node_tree
nt.nodes.clear()
rl = nt.nodes.new('CompositorNodeRLayers')
# output beauty
scene.render.filepath = out_dir + '/ctrl3d_beauty.png'
bpy.ops.render.render(write_still=True)

# Also write mist/depth via view layer
view_layer = scene.view_layers[0]
view_layer.use_pass_z = True
view_layer.use_pass_normal = True
print("BLENDER_RENDER_OK")
'''


def render_controls_blender(
    plan: RoomPlan,
    gltf_path: str | Path,
    out_dir: str | Path,
    *,
    camera: CameraSpec | None = None,
    width: int = 720,
    height: int = 1280,
) -> RenderBundle | None:
    blender = shutil.which("blender")
    if not blender:
        return None
    scene = extrude_room_plan(plan)
    eye, target, cam_meta = _camera_eye_target_from_plan(plan, scene, camera)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        cam_json = Path(td) / "cam.json"
        cam_json.write_text(json.dumps(cam_meta), encoding="utf-8")
        script = Path(td) / "render.py"
        script.write_text(_BLENDER_SCRIPT, encoding="utf-8")
        cmd = [
            blender,
            "-b",
            "-P",
            str(script),
            "--",
            str(gltf_path),
            str(out_dir),
            str(cam_json),
            str(width),
            str(height),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=180)
        except Exception:
            return None
    # Fall back compose software maps if blender only wrote beauty
    soft = render_controls_software(plan, scene=scene, camera=camera, width=width, height=height)
    beauty_path = out_dir / "ctrl3d_beauty.png"
    if beauty_path.exists():
        img = cv2.imread(str(beauty_path))
        if img is not None:
            soft.beauty = img
            soft.meta["renderer"] = "blender+software_controls"
    soft.camera = cam_meta
    soft.save(out_dir)
    return soft


def render_topdown_qa(
    scene: Scene3D, *, size: int = 768
) -> np.ndarray:
    """Orthographic-ish top-down QA preview (Z-up look-down)."""
    V, F, face_ids, kind_names = scene.merged_arrays()
    if len(V) == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    cx = float((V[:, 0].min() + V[:, 0].max()) / 2)
    cy = float((V[:, 1].min() + V[:, 1].max()) / 2)
    span = float(max(float(np.ptp(V[:, 0])), float(np.ptp(V[:, 1])), 1.0))
    eye = np.array([cx, cy, span * 1.4], dtype=np.float64)
    target = np.array([cx, cy, 0.0], dtype=np.float64)
    # look down -Z with +Y as up in screen
    f = target - eye
    f = f / (np.linalg.norm(f) or 1.0)
    approx_up = np.array([0.0, 1.0, 0.0])
    s = np.cross(f, approx_up)
    s = s / (np.linalg.norm(s) or 1.0)
    u = np.cross(s, f)
    R = np.stack([s, u, f], axis=0)
    maps = _rasterize(
        V, F, face_ids, kind_names, eye, R, width=size, height=size, fov_deg=50.0
    )
    return maps["beauty"]


def build_3d_controls(
    plan: RoomPlan,
    out_dir: str | Path,
    *,
    gltf_path: str | Path | None = None,
    camera: CameraSpec | None = None,
    prefer_blender: bool = False,
    width: int = 720,
    height: int = 1280,
) -> tuple[dict[str, str], RenderBundle, Path]:
    """Extrude → glTF → render control maps. Returns (paths, bundle, gltf)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if gltf_path is None:
        gltf_path = out_dir / "scene.glb"
    from cad_ai_lock.extrude import export_plan_gltf

    gltf, scene = export_plan_gltf(plan, gltf_path)
    bundle = None
    if prefer_blender:
        bundle = render_controls_blender(
            plan, gltf, out_dir, camera=camera, width=width, height=height
        )
    if bundle is None:
        bundle = render_controls_software(
            plan, scene=scene, camera=camera, width=width, height=height
        )
    paths = bundle.save(out_dir)
    paths["gltf"] = str(gltf)
    paths["gltf_meta"] = str(Path(gltf).with_suffix(".json"))
    top = render_topdown_qa(scene)
    top_path = out_dir / "ctrl3d_topdown_qa.png"
    cv2.imwrite(str(top_path), top)
    paths["topdown_qa"] = str(top_path)
    return paths, bundle, Path(gltf)
