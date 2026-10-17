"""Validate generated images against locked control lines + soft policy scores.

Hard: wall/opening edge alignment (must pass).
Soft: layout policy scorecard (advisory / product UX).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cad_ai_lock.control import render_canny_from_plan
from cad_ai_lock.geometry import FurnitureLayout, RoomPlan


@dataclass
class ValidationResult:
    score: float  # 0..1 geometry
    edge_overlap: float
    control_coverage: float
    passed: bool
    threshold: float
    overlay_path: str | None = None
    message: str = ""
    policy_overall: float | None = None
    policy_passed: bool | None = None
    policy_threshold: float = 0.55
    scorecard: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "edge_overlap": round(self.edge_overlap, 4),
            "control_coverage": round(self.control_coverage, 4),
            "passed": self.passed,
            "threshold": self.threshold,
            "overlay_path": self.overlay_path,
            "message": self.message,
            "policy_overall": self.policy_overall,
            "policy_passed": self.policy_passed,
            "policy_threshold": self.policy_threshold,
            "scorecard": self.scorecard,
        }


def _binary_edges(gray_or_bgr: np.ndarray) -> np.ndarray:
    if gray_or_bgr.ndim == 3:
        gray = cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_or_bgr
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    if float(np.mean(gray > 0)) < 0.15:
        _, bw = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
        return bw
    return cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)


def validate_alignment(
    generated_bgr: np.ndarray,
    plan: RoomPlan,
    *,
    control_canny: np.ndarray | None = None,
    threshold: float = 0.35,
    dilate: int = 3,
    overlay_out: str | Path | None = None,
    layout: FurnitureLayout | None = None,
    policy_threshold: float = 0.55,
) -> ValidationResult:
    h, w = plan.image_height, plan.image_width
    gen = cv2.resize(generated_bgr, (w, h))
    control = control_canny if control_canny is not None else render_canny_from_plan(plan)
    if control.shape[:2] != (h, w):
        control = cv2.resize(control, (w, h))

    gen_e = _binary_edges(gen)
    ctrl_e = _binary_edges(control)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
    gen_d = cv2.dilate(gen_e, kernel, iterations=1)
    ctrl_d = cv2.dilate(ctrl_e, kernel, iterations=1)

    ctrl_px = int(np.count_nonzero(ctrl_e))
    gen_px = int(np.count_nonzero(gen_e))
    if ctrl_px == 0:
        return ValidationResult(
            score=0.0,
            edge_overlap=0.0,
            control_coverage=0.0,
            passed=False,
            threshold=threshold,
            message="empty control edges",
        )

    covered = int(np.count_nonzero((ctrl_e > 0) & (gen_d > 0)))
    control_coverage = covered / ctrl_px
    if gen_px > 0:
        near = int(np.count_nonzero((gen_e > 0) & (ctrl_d > 0)))
        precision = near / gen_px
    else:
        precision = 0.0

    edge_overlap = float(
        2 * control_coverage * precision / (control_coverage + precision + 1e-8)
    )
    score = 0.65 * control_coverage + 0.35 * edge_overlap
    passed = score >= threshold

    policy_overall = None
    policy_passed = None
    scorecard: dict[str, Any] = {}
    if layout is not None:
        scorecard = dict(layout.scorecard or {})
        policy_overall = float(scorecard.get("overall", 0.0))
        policy_passed = policy_overall >= policy_threshold

    overlay_path = None
    if overlay_out:
        overlay_out = Path(overlay_out)
        overlay_out.parent.mkdir(parents=True, exist_ok=True)
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        overlay[ctrl_e > 0] = (255, 80, 80)
        overlay[gen_e > 0] = (80, 255, 80)
        both = (ctrl_e > 0) & (gen_e > 0)
        overlay[both] = (80, 80, 255)
        base = (gen.astype(np.float32) * 0.35).astype(np.uint8)
        mix = cv2.addWeighted(base, 0.6, overlay, 0.9, 0)
        cv2.imwrite(str(overlay_out), mix)
        overlay_path = str(overlay_out)

    if passed and (policy_passed is False):
        msg = "OK structure; soft layout score below preference threshold"
    elif passed:
        msg = "OK — structure locked"
    else:
        msg = "FAIL — walls/openings likely drifted"

    return ValidationResult(
        score=score,
        edge_overlap=edge_overlap,
        control_coverage=control_coverage,
        passed=passed,
        threshold=threshold,
        overlay_path=overlay_path,
        message=msg,
        policy_overall=policy_overall,
        policy_passed=policy_passed,
        policy_threshold=policy_threshold,
        scorecard=scorecard,
    )


def validate_file(
    generated_path: str | Path,
    plan: RoomPlan,
    *,
    threshold: float = 0.35,
    overlay_out: str | Path | None = None,
    layout: FurnitureLayout | None = None,
    policy_threshold: float = 0.55,
) -> ValidationResult:
    img = cv2.imread(str(generated_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(generated_path)
    return validate_alignment(
        img,
        plan,
        threshold=threshold,
        overlay_out=overlay_out,
        layout=layout,
        policy_threshold=policy_threshold,
    )
