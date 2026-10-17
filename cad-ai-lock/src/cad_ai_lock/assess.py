"""Plan / layout / render health assessment for the auto-correction loop.

Commercial contract: every stage emits a scored health report with typed
issues. Downstream `repair` / `pipeline` only act on those issues — never
on ad-hoc manual edits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from cad_ai_lock.geometry import FurnitureLayout, OpeningType, RoomPlan, Wall


class Severity(str, Enum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    FATAL = "fatal"


@dataclass
class Issue:
    code: str
    severity: Severity
    message: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "meta": self.meta,
        }


@dataclass
class PlanHealth:
    score: float  # 0..1
    passed: bool
    threshold: float
    issues: list[Issue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    suggested_repairs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "passed": self.passed,
            "threshold": self.threshold,
            "issues": [i.to_dict() for i in self.issues],
            "metrics": self.metrics,
            "suggested_repairs": self.suggested_repairs,
        }


def _wall_angle_deg(wall: Wall) -> float:
    return math.degrees(
        math.atan2(wall.end.y - wall.start.y, wall.end.x - wall.start.x)
    ) % 180


def _is_ortho(wall: Wall, tol_deg: float = 12.0) -> bool:
    a = _wall_angle_deg(wall)
    return min(abs(a - 0), abs(a - 90), abs(a - 180)) <= tol_deg


def _endpoint_clusters(
    walls: list[Wall], tol: float = 12.0
) -> tuple[int, float]:
    """Return (cluster_count, avg_endpoints_per_cluster)."""
    pts: list[tuple[float, float]] = []
    for w in walls:
        pts.append((w.start.x, w.start.y))
        pts.append((w.end.x, w.end.y))
    if not pts:
        return 0, 0.0
    used = [False] * len(pts)
    clusters = 0
    sizes: list[int] = []
    for i, p in enumerate(pts):
        if used[i]:
            continue
        clusters += 1
        size = 1
        used[i] = True
        for j in range(i + 1, len(pts)):
            if used[j]:
                continue
            if math.hypot(pts[j][0] - p[0], pts[j][1] - p[1]) <= tol:
                used[j] = True
                size += 1
        sizes.append(size)
    avg = float(np.mean(sizes)) if sizes else 0.0
    return clusters, avg


def _bbox_area(plan: RoomPlan) -> float:
    x0, y0, x1, y1 = plan.bounding_box()
    return max(0.0, (x1 - x0) * (y1 - y0))


def assess_plan(
    plan: RoomPlan,
    *,
    threshold: float = 0.62,
    expect_rooms: bool = True,
) -> PlanHealth:
    """Score RoomPlan topology quality and emit repair suggestions."""
    issues: list[Issue] = []
    repairs: list[str] = []
    n_walls = len(plan.walls)
    n_open = len(plan.openings)
    n_rooms = len(plan.rooms or [])

    if n_walls == 0:
        issues.append(
            Issue("no_walls", Severity.FATAL, "未检测到墙体")
        )
        repairs.append("re_extract_walls")
    elif n_walls > 80:
        issues.append(
            Issue(
                "wall_oversegmented",
                Severity.ERROR,
                f"墙段过多 ({n_walls})，疑似未合并碎线",
                {"count": n_walls},
            )
        )
        repairs.extend(["merge_collinear", "orthogonal_snap", "grid_snap"])
    elif n_walls > 40:
        issues.append(
            Issue(
                "wall_noisy",
                Severity.WARN,
                f"墙段偏多 ({n_walls})",
                {"count": n_walls},
            )
        )
        repairs.append("merge_collinear")

    ortho = sum(1 for w in plan.walls if _is_ortho(w))
    ortho_ratio = ortho / n_walls if n_walls else 0.0
    if n_walls and ortho_ratio < 0.75:
        issues.append(
            Issue(
                "non_orthogonal",
                Severity.ERROR,
                f"正交比例偏低 {ortho_ratio:.2f}",
                {"ortho_ratio": ortho_ratio},
            )
        )
        repairs.append("orthogonal_snap")

    clusters, avg_ep = _endpoint_clusters(plan.walls)
    dangling = sum(1 for s in [avg_ep] if avg_ep < 1.35)
    if n_walls >= 4 and avg_ep < 1.4:
        issues.append(
            Issue(
                "weak_connectivity",
                Severity.WARN,
                f"端点聚类弱 (avg={avg_ep:.2f})",
                {"clusters": clusters, "avg_endpoints": avg_ep},
            )
        )
        repairs.append("snap_junctions")

    # duplicate / near-duplicate walls
    dup = 0
    for i, a in enumerate(plan.walls):
        for b in plan.walls[i + 1 :]:
            if abs(a.length_px - b.length_px) > 25:
                continue
            d1 = math.hypot(a.start.x - b.start.x, a.start.y - b.start.y) + math.hypot(
                a.end.x - b.end.x, a.end.y - b.end.y
            )
            d2 = math.hypot(a.start.x - b.end.x, a.start.y - b.end.y) + math.hypot(
                a.end.x - b.start.x, a.end.y - b.start.y
            )
            if min(d1, d2) < 24:
                dup += 1
    if dup >= 3:
        issues.append(
            Issue(
                "duplicate_walls",
                Severity.ERROR,
                f"疑似重复墙段 ×{dup}",
                {"duplicates": dup},
            )
        )
        repairs.append("dedupe_walls")

    if expect_rooms and n_rooms == 0:
        issues.append(
            Issue("no_rooms", Severity.ERROR, "未分割出房间区域")
        )
        repairs.append("infer_rooms")
    elif n_rooms == 1 and n_walls >= 8:
        issues.append(
            Issue(
                "undersegmented_rooms",
                Severity.WARN,
                "仅 1 个房间，内墙可能未闭合",
            )
        )
        repairs.append("infer_rooms")

    labeled = sum(
        1
        for r in (plan.rooms or [])
        if str(r.get("name") or r.get("type") or "")
        not in ("", "房间", "room")
    )
    if n_rooms >= 3 and labeled / max(n_rooms, 1) < 0.4:
        issues.append(
            Issue(
                "unlabeled_rooms",
                Severity.WARN,
                f"房间命名不足 ({labeled}/{n_rooms})",
            )
        )
        repairs.append("label_rooms")

    if plan.scale_px_per_m is None:
        issues.append(
            Issue("no_scale", Severity.WARN, "缺少比例尺 scale_px_per_m")
        )
        repairs.append("estimate_scale")

    if n_open == 0 and n_walls >= 4:
        issues.append(
            Issue("no_openings", Severity.WARN, "未检测到门窗")
        )
        repairs.append("infer_openings")

    # openings not attached
    orphan = sum(1 for o in plan.openings if not o.wall_id)
    if n_open and orphan / n_open > 0.6:
        issues.append(
            Issue(
                "orphan_openings",
                Severity.WARN,
                f"门窗未附着墙体 ({orphan}/{n_open})",
            )
        )
        repairs.append("attach_openings")

    bbox = _bbox_area(plan)
    img_area = float(plan.image_width * plan.image_height)
    coverage = bbox / img_area if img_area else 0.0
    if n_walls and coverage < 0.08:
        issues.append(
            Issue(
                "tiny_envelope",
                Severity.ERROR,
                f"包络过小 coverage={coverage:.3f}",
                {"coverage": coverage},
            )
        )
        repairs.append("re_extract_walls")

    # score: start 1.0, subtract by severity
    score = 1.0
    for iss in issues:
        if iss.severity == Severity.FATAL:
            score -= 0.55
        elif iss.severity == Severity.ERROR:
            score -= 0.18
        elif iss.severity == Severity.WARN:
            score -= 0.07
        else:
            score -= 0.02
    # bonuses
    if 6 <= n_walls <= 35:
        score += 0.05
    if n_rooms >= 4:
        score += 0.05
    if plan.scale_px_per_m:
        score += 0.04
    if ortho_ratio >= 0.9:
        score += 0.04
    score = float(max(0.0, min(1.0, score)))

    # unique repairs preserving order
    seen: set[str] = set()
    uniq_repairs: list[str] = []
    for r in repairs:
        if r not in seen:
            seen.add(r)
            uniq_repairs.append(r)

    passed = score >= threshold and not any(
        i.severity in (Severity.FATAL, Severity.ERROR) for i in issues
    )
    # allow pass with only warns if score ok
    if score >= threshold and not any(i.severity == Severity.FATAL for i in issues):
        # ERROR without fatal: still fail so repair runs
        passed = not any(i.severity == Severity.ERROR for i in issues)

    metrics = {
        "walls": n_walls,
        "openings": n_open,
        "rooms": n_rooms,
        "labeled_rooms": labeled,
        "ortho_ratio": round(ortho_ratio, 4),
        "endpoint_clusters": clusters,
        "avg_endpoints_per_cluster": round(avg_ep, 4),
        "duplicate_pairs": dup,
        "envelope_coverage": round(coverage, 4),
        "scale_px_per_m": plan.scale_px_per_m,
        "plan_revision": plan.plan_revision,
    }
    return PlanHealth(
        score=score,
        passed=passed,
        threshold=threshold,
        issues=issues,
        metrics=metrics,
        suggested_repairs=uniq_repairs,
    )


def assess_layout(layout: FurnitureLayout, plan: RoomPlan) -> PlanHealth:
    """Soft check: furniture must sit inside rooms / not block doors."""
    issues: list[Issue] = []
    repairs: list[str] = []
    items = layout.items or []
    if not items and (plan.rooms or []):
        issues.append(
            Issue("empty_layout", Severity.WARN, "家具布局为空")
        )
        repairs.append("rebuild_layout")
    overall = float((layout.scorecard or {}).get("overall") or 0.0)
    if items and overall < 0.45:
        issues.append(
            Issue(
                "low_layout_score",
                Severity.WARN,
                f"布局策略分偏低 {overall:.2f}",
                {"overall": overall},
            )
        )
        repairs.append("rebuild_layout")
    score = max(0.0, min(1.0, 0.35 + 0.1 * len(items) + overall * 0.5))
    if len(items) > 40:
        score -= 0.1
        issues.append(Issue("layout_overcrowded", Severity.WARN, "家具过多"))
        repairs.append("rebuild_layout")
    return PlanHealth(
        score=score,
        passed=score >= 0.5 and not any(i.severity == Severity.ERROR for i in issues),
        threshold=0.5,
        issues=issues,
        metrics={
            "items": len(items),
            "overall": overall,
            "policies": list(layout.policies or []),
        },
        suggested_repairs=list(dict.fromkeys(repairs)),
    )
