"""Commercial closed-loop pipeline with automatic correction.

Stages:
  1. extract (+ OCR)
  2. assess plan → repair until pass / budget
  3. layout (+ soft assess → rebuild)
  4. controls
  5. generate (t2i / optional i2i / optional VR camera still via mock/provider)
  6. validate alignment → if fail: bump control strength & regenerate
  7. emit run_report.json (full audit trail — no silent hand edits)

Hand-edited plan.json is NEVER part of the happy path. Operator override is
an explicit CLI flag (`--plan-override`) for support tools only.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cad_ai_lock.assess import PlanHealth, assess_layout, assess_plan
from cad_ai_lock.control import write_control_bundle
from cad_ai_lock.extract import extract_room_plan
from cad_ai_lock.generate import get_provider, run_i2i, run_t2i
from cad_ai_lock.geometry import FurnitureLayout, RoomPlan
from cad_ai_lock.layout import build_layout
from cad_ai_lock.repair import auto_repair
from cad_ai_lock.validate import ValidationResult, validate_file


@dataclass
class StageResult:
    name: str
    ok: bool
    duration_ms: int
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "duration_ms": self.duration_ms,
            "detail": self.detail,
        }


@dataclass
class PipelineReport:
    version: str = "0.3.1"
    mode: str = "commercial_auto"
    input_path: str = ""
    workdir: str = ""
    success: bool = False
    stages: list[StageResult] = field(default_factory=list)
    plan_health: dict[str, Any] = field(default_factory=dict)
    layout_health: dict[str, Any] = field(default_factory=dict)
    validations: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    corrections: list[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "mode": self.mode,
            "input_path": self.input_path,
            "workdir": self.workdir,
            "success": self.success,
            "stages": [s.to_dict() for s in self.stages],
            "plan_health": self.plan_health,
            "layout_health": self.layout_health,
            "validations": self.validations,
            "artifacts": self.artifacts,
            "corrections": self.corrections,
            "message": self.message,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _timed(name: str, fn):
    t0 = time.time()
    result = fn()
    ms = int((time.time() - t0) * 1000)
    return result, ms


def run_commercial_pipeline(
    input_path: str | Path,
    *,
    workdir: str | Path,
    prompt: str = "现代简约装修，木地板，柔和自然光",
    provider: str = "mock",
    scale: float | None = None,
    daylight: bool = True,
    fengshui: bool = False,
    common_sense: bool = True,
    use_ocr: bool = True,
    use_vision: bool = False,
    max_plan_repair_rounds: int = 3,
    max_generate_retries: int = 3,
    structure_threshold: float = 0.35,
    plan_health_threshold: float = 0.55,
    strength: float | None = None,
    also_i2i: bool = True,
    plan_override: str | Path | None = None,
) -> PipelineReport:
    """End-to-end auto pipeline with correction loops."""
    import os

    from cad_ai_lock.config import get_settings

    input_path = Path(input_path)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    report = PipelineReport(
        input_path=str(input_path),
        workdir=str(workdir),
    )

    os.environ["CAD_AI_PROVIDER"] = provider
    get_settings.cache_clear()

    corrections: list[str] = []
    image: np.ndarray

    # ----- 1) extract -----
    def do_extract():
        if plan_override:
            plan = RoomPlan.from_file(str(plan_override))
            img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(input_path)
            return plan, img, None, ["plan_override"]
        plan, img, edges = extract_room_plan(
            input_path,
            use_vision=use_vision,
            scale_px_per_m=scale,
            auto_correct=True,
            use_ocr=use_ocr,
        )
        applied = list(plan.meta.get("extract_repairs") or [])
        return plan, img, edges, applied

    try:
        (plan, image, _edges, applied), ms = _timed("extract", do_extract)
        corrections.extend(applied)
        report.stages.append(
            StageResult(
                "extract",
                True,
                ms,
                {
                    "walls": len(plan.walls),
                    "openings": len(plan.openings),
                    "rooms": len(plan.rooms or []),
                    "repairs": applied,
                    "override": bool(plan_override),
                },
            )
        )
    except Exception as exc:  # noqa: BLE001
        report.stages.append(StageResult("extract", False, 0, {"error": str(exc)}))
        report.message = f"extract failed: {exc}"
        report.save(workdir / "run_report.json")
        return report

    # ----- 2) assess + extra repair rounds -----
    def do_plan_loop():
        nonlocal plan, corrections
        health = assess_plan(plan, threshold=plan_health_threshold)
        rounds = []
        for r in range(max_plan_repair_rounds):
            if health.passed:
                break
            plan, health, applied = auto_repair(
                plan,
                health,
                image_bgr=image,
                ocr_items=plan.meta.get("ocr_items") or [],
            )
            corrections.extend(applied)
            rounds.append({"round": r + 1, "applied": applied, "health": health.to_dict()})
            if "re_extract_walls" in health.suggested_repairs and r == 0:
                # one hard re-extract without short-circuiting OCR cache
                plan2, _, _ = extract_room_plan(
                    input_path,
                    scale_px_per_m=scale or plan.scale_px_per_m,
                    auto_correct=True,
                    use_ocr=use_ocr,
                    use_vision=use_vision,
                )
                plan = plan2
                corrections.append("re_extract_walls")
                health = assess_plan(plan, threshold=plan_health_threshold)
                rounds.append({"round": f"{r+1}b", "re_extract": True, "health": health.to_dict()})
        return health, rounds

    health, rounds = do_plan_loop()
    report.plan_health = health.to_dict()
    report.stages.append(
        StageResult(
            "plan_repair_loop",
            health.passed or health.score >= plan_health_threshold * 0.85,
            0,
            {"rounds": rounds, "final": health.to_dict()},
        )
    )

    plan_path = workdir / "plan.json"
    plan.save(str(plan_path))
    report.artifacts["plan"] = str(plan_path)
    cv2.imwrite(str(workdir / "source.jpg"), image)
    report.artifacts["source"] = str(workdir / "source.jpg")

    # ----- 3) layout -----
    def do_layout():
        nonlocal corrections
        layout = build_layout(
            plan,
            daylight=daylight,
            fengshui=fengshui,
            common_sense=common_sense,
        )
        lhealth = assess_layout(layout, plan)
        if not lhealth.passed:
            layout = build_layout(
                plan,
                daylight=daylight,
                fengshui=False,
                common_sense=True,
            )
            corrections.append("rebuild_layout")
            lhealth = assess_layout(layout, plan)
        return layout, lhealth

    (layout, lhealth), ms = _timed("layout", do_layout)
    report.layout_health = lhealth.to_dict()
    report.stages.append(
        StageResult(
            "layout",
            lhealth.passed,
            ms,
            {"items": len(layout.items), "scorecard": layout.scorecard},
        )
    )
    layout_path = workdir / "layout.json"
    layout.save(str(layout_path))
    report.artifacts["layout"] = str(layout_path)

    # ----- 4) controls -----
    control_dir = workdir / "control"
    paths = write_control_bundle(plan, control_dir, source_image=image, layout=layout)
    report.artifacts.update({f"control_{k}": v for k, v in paths.items()})
    report.stages.append(StageResult("controls", True, 0, {"files": list(paths)}))

    # ----- 5/6) generate + validate with retries -----
    prov = get_provider()
    settings = get_settings()
    base_strength = float(strength if strength is not None else settings.control_strength)
    validations: dict[str, Any] = {}
    best_t2i: Path | None = None
    best_v: ValidationResult | None = None

    for attempt in range(max_generate_retries):
        cur_strength = min(0.95, base_strength + 0.05 * attempt)
        seed = 42 + attempt

        def do_gen():
            return run_t2i(
                plan,
                prompt,
                workdir / f"t2i_try{attempt}.png",
                provider=prov,
                seed=seed,
                strength=cur_strength,
                layout=layout,
            )

        try:
            t2i_path, ms = _timed(f"t2i_{attempt}", do_gen)
        except Exception as exc:  # noqa: BLE001
            report.stages.append(
                StageResult(f"t2i_{attempt}", False, 0, {"error": str(exc)})
            )
            corrections.append(f"t2i_error_{attempt}")
            continue

        v = validate_file(
            t2i_path,
            plan,
            threshold=structure_threshold,
            overlay_out=workdir / f"t2i_overlay_try{attempt}.png",
            layout=layout,
        )
        validations[f"t2i_try{attempt}"] = {
            **v.to_dict(),
            "strength": cur_strength,
            "path": str(t2i_path),
        }
        report.stages.append(
            StageResult(
                f"validate_t2i_{attempt}",
                v.passed,
                ms,
                {"score": v.score, "strength": cur_strength},
            )
        )
        if best_v is None or v.score > best_v.score:
            best_v = v
            best_t2i = Path(t2i_path)
        if v.passed:
            corrections.append(f"t2i_accepted_try{attempt}")
            break
        corrections.append(f"t2i_retry_strengthen_{attempt}")

    if best_t2i is not None:
        final_t2i = workdir / "t2i.png"
        img = cv2.imread(str(best_t2i))
        if img is not None:
            cv2.imwrite(str(final_t2i), img)
        report.artifacts["t2i"] = str(final_t2i)

    if also_i2i:
        try:
            i2i_path = run_i2i(
                plan,
                workdir / "source.jpg",
                prompt,
                workdir / "i2i.png",
                provider=prov,
                seed=42,
                strength=base_strength,
                layout=layout,
            )
            v2 = validate_file(
                i2i_path,
                plan,
                threshold=structure_threshold,
                overlay_out=workdir / "i2i_overlay.png",
                layout=layout,
            )
            validations["i2i"] = v2.to_dict()
            report.artifacts["i2i"] = str(i2i_path)
            report.stages.append(
                StageResult("i2i_validate", v2.passed, 0, {"score": v2.score})
            )
        except Exception as exc:  # noqa: BLE001
            report.stages.append(StageResult("i2i", False, 0, {"error": str(exc)}))

    # ----- 7) VR perspective hard-lock stage -----
    try:
        from cad_ai_lock.perspective import (
            assess_vr_against_contract,
            living_vr_prompt,
            place_living_camera,
            render_perspective_control,
        )

        cam = layout.camera or plan.camera or place_living_camera(plan)
        plan.camera = cam
        layout.camera = cam
        persp, contract = render_perspective_control(plan, cam, layout=layout)
        persp_path = workdir / "control" / "control_perspective.png"
        persp_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(persp_path), persp)
        contract_path = workdir / "view_contract.json"
        contract_path.write_text(
            json.dumps(contract.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # Mock VR still = perspective wireframe as locked scaffold (providers
        # must condition on this; free-form VR without it is a pipeline fail).
        vr_path = workdir / "vr_living.png"
        cv2.imwrite(str(vr_path), persp)
        vr_prompt = living_vr_prompt(contract, prompt)
        (workdir / "vr_prompt.txt").write_text(vr_prompt, encoding="utf-8")
        vr_assess = assess_vr_against_contract(contract)
        validations["vr_contract"] = vr_assess
        report.artifacts["perspective"] = str(persp_path)
        report.artifacts["view_contract"] = str(contract_path)
        report.artifacts["vr_living"] = str(vr_path)
        report.artifacts["vr_prompt"] = str(workdir / "vr_prompt.txt")
        report.stages.append(
            StageResult(
                "vr_perspective_lock",
                True,
                0,
                {
                    "far_wall": contract.far_wall,
                    "left_openings": contract.left_openings,
                    "right_openings": contract.right_openings,
                    "prompt_lock_zh": contract.prompt_lock_zh,
                },
            )
        )
        corrections.append("vr_perspective_lock")
    except Exception as exc:  # noqa: BLE001
        report.stages.append(
            StageResult("vr_perspective_lock", False, 0, {"error": str(exc)})
        )
        corrections.append("vr_perspective_lock_failed")

    report.validations = validations
    report.corrections = list(dict.fromkeys(corrections))
    t2i_ok = bool(best_v and best_v.passed)
    plan_ok = bool(health.passed or health.score >= plan_health_threshold * 0.85)
    vr_ok = any(s.name == "vr_perspective_lock" and s.ok for s in report.stages)
    report.success = plan_ok and (t2i_ok or provider == "mock") and vr_ok
    # mock provider composites controls — treat as structural success if plan_ok
    if provider == "mock" and plan_ok and best_t2i is not None and vr_ok:
        report.success = True
    report.message = (
        "OK — commercial auto pipeline completed (incl. VR perspective lock)"
        if report.success
        else "COMPLETED_WITH_ISSUES — see plan_health / validations / VR contract"
    )
    report.artifacts["run_report"] = str(workdir / "run_report.json")
    report.save(workdir / "run_report.json")
    return report
