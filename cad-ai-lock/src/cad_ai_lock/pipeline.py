"""Commercial closed-loop pipeline: generate first, then hillclimb corrections.

UX contract:
  - Always emit draft artifacts (plan / controls / t2i / glTF / VR scaffold).
  - If fidelity vs source floorplan is weak, do NOT abort — run correction
    rounds: repair → rebuild layout/controls → regenerate → re-score.
  - Final grade: locked | improved | draft (always downloadable).

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
    version: str = "0.4.3"
    mode: str = "generate_then_correct"
    input_path: str = ""
    workdir: str = ""
    success: bool = False
    locked: bool = False  # True only when fidelity vs source passed
    quality_grade: str = "draft"  # locked | improved | draft
    correction_rounds: int = 0
    stages: list[StageResult] = field(default_factory=list)
    plan_health: dict[str, Any] = field(default_factory=dict)
    layout_health: dict[str, Any] = field(default_factory=dict)
    validations: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    corrections: list[str] = field(default_factory=list)
    iteration_log: list[dict[str, Any]] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "mode": self.mode,
            "input_path": self.input_path,
            "workdir": self.workdir,
            "success": self.success,
            "locked": self.locked,
            "quality_grade": self.quality_grade,
            "correction_rounds": self.correction_rounds,
            "stages": [s.to_dict() for s in self.stages],
            "plan_health": self.plan_health,
            "layout_health": self.layout_health,
            "validations": self.validations,
            "artifacts": self.artifacts,
            "corrections": self.corrections,
            "iteration_log": self.iteration_log,
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
    max_correct_rounds: int = 3,
    structure_threshold: float = 0.35,
    plan_health_threshold: float = 0.55,
    strength: float | None = None,
    also_i2i: bool = True,
    plan_override: str | Path | None = None,
) -> PipelineReport:
    """Generate draft first, then hillclimb corrections — never soft-abort UX."""
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

    # Score plan vs source — weak fidelity triggers hillclimb LATER, never aborts draft
    from cad_ai_lock.fidelity import assess_image_vs_source, assess_plan_vs_source

    fidelity = assess_plan_vs_source(plan, image, expect_min_rooms=5)
    report.stages.append(
        StageResult(
            "fidelity_plan_vs_source",
            fidelity.passed,
            0,
            fidelity.to_dict(),
        )
    )
    if not fidelity.passed:
        corrections.append("fidelity_weak_will_hillclimb")
    fidelity0_score = float(fidelity.score)

    plan_path = workdir / "plan.json"
    plan.save(str(plan_path))
    report.artifacts["plan"] = str(plan_path)
    cv2.imwrite(str(workdir / "source.jpg"), image)
    report.artifacts["source"] = str(workdir / "source.jpg")
    report.artifacts["fidelity_plan"] = str(workdir / "fidelity_plan.json")
    (workdir / "fidelity_plan.json").write_text(
        json.dumps(fidelity.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report.iteration_log.append(
        {"iter": 0, "phase": "draft_plan", "fidelity": fidelity.to_dict()}
    )

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

    # ----- 7) True 3D: extrude glTF + render control maps -----
    try:
        from cad_ai_lock.perspective import (
            assess_vr_against_contract,
            living_vr_prompt,
            place_living_camera,
            render_perspective_control,
        )
        from cad_ai_lock.render3d import build_3d_controls

        cam = layout.camera or plan.camera or place_living_camera(plan)
        plan.camera = cam
        layout.camera = cam

        def do_3d():
            return build_3d_controls(
                plan,
                workdir / "ctrl3d",
                camera=cam,
                prefer_blender=False,
                width=720,
                height=1280,
            )

        (paths3d, bundle3d, glb), ms = _timed("extrude_render3d", do_3d)
        report.artifacts["gltf"] = str(glb)
        report.artifacts.update({f"ctrl3d_{k}": v for k, v in paths3d.items()})
        report.stages.append(
            StageResult(
                "extrude_render3d",
                True,
                ms,
                {
                    "renderer": bundle3d.meta.get("renderer"),
                    "parts": bundle3d.meta.get("parts"),
                    "gltf": str(glb),
                },
            )
        )
        corrections.append("extrude_gltf")
        corrections.append("render3d_controls")

        # 2D perspective wireframe still kept as debug overlay / prompt lock
        persp, contract = render_perspective_control(plan, cam, layout=layout)
        persp_path = workdir / "control" / "control_perspective.png"
        persp_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(persp_path), persp)
        contract_path = workdir / "view_contract.json"
        contract_path.write_text(
            json.dumps(contract.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # VR scaffold = true-3D beauty (geometry-locked), not free-form gen
        vr_path = workdir / "vr_living.png"
        cv2.imwrite(str(vr_path), bundle3d.beauty)
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
                    "source": "ctrl3d_beauty_from_gltf",
                    "prompt_lock_zh": contract.prompt_lock_zh,
                },
            )
        )
        corrections.append("vr_from_3d_mesh")

        # Multi-room tour: per-room cams → parallel render → hotspots
        def do_tour():
            from cad_ai_lock.tour import render_tour_parallel

            return render_tour_parallel(
                plan,
                workdir / "tour",
                width=720,
                height=1280,
            )

        try:
            tour, ms_tour = _timed("tour_parallel", do_tour)
            report.artifacts["tour_json"] = str(workdir / "tour" / "tour.json")
            report.artifacts["tour_html"] = str(workdir / "tour" / "tour.html")
            contact = workdir / "tour" / "tour_contact.png"
            if contact.exists():
                report.artifacts["tour_contact"] = str(contact)
            report.stages.append(
                StageResult(
                    "tour_parallel",
                    True,
                    ms_tour,
                    {
                        "rooms": tour.meta.get("rooms_rendered"),
                        "hotspots": len(tour.hotspots),
                        "workers": tour.meta.get("workers"),
                        "start_room": tour.start_room,
                        "errors": tour.meta.get("errors") or [],
                    },
                )
            )
            corrections.append("tour_parallel_rooms")
        except Exception as tour_exc:  # noqa: BLE001
            report.stages.append(
                StageResult("tour_parallel", False, 0, {"error": str(tour_exc)})
            )
            corrections.append("tour_parallel_failed")
    except Exception as exc:  # noqa: BLE001
        report.stages.append(
            StageResult("extrude_render3d", False, 0, {"error": str(exc)})
        )
        corrections.append("extrude_render3d_failed")

    # Score draft outputs vs source (informational — draft already saved)
    img_fid = None
    if best_t2i is not None and Path(workdir / "t2i.png").exists():
        gen_img = cv2.imread(str(workdir / "t2i.png"))
        if gen_img is not None:
            img_fid = assess_image_vs_source(gen_img, image, label="t2i")
            validations["t2i_vs_source"] = img_fid.to_dict()
            report.stages.append(
                StageResult("fidelity_t2i_vs_source", img_fid.passed, 0, img_fid.to_dict())
            )
            if not img_fid.passed:
                corrections.append("t2i_deviates_will_hillclimb")

    # Snapshot draft iteration for progressive review
    draft_dir = workdir / "iter_0_draft"
    draft_dir.mkdir(exist_ok=True)
    for name in ("plan.json", "layout.json", "t2i.png", "vr_living.png"):
        src = workdir / name
        if not src.exists():
            continue
        if name.endswith(".png"):
            img = cv2.imread(str(src))
            if img is not None:
                cv2.imwrite(str(draft_dir / name), img)
        else:
            (draft_dir / name).write_bytes(src.read_bytes())
    report.artifacts["iter_0_draft"] = str(draft_dir)
    report.iteration_log.append(
        {
            "iter": 0,
            "phase": "draft_outputs",
            "fidelity_plan": fidelity.to_dict(),
            "fidelity_t2i": img_fid.to_dict() if img_fid else None,
        }
    )

    # ----- Hillclimb: generate already done → stepwise correct -----
    best_fidelity_score = float(fidelity.score)
    from cad_ai_lock.repair import auto_repair, infer_rooms, estimate_scale_from_ocr

    for cround in range(1, max_correct_rounds + 1):
        if fidelity.passed and (img_fid is None or img_fid.passed):
            break
        corrections.append(f"correct_round_{cround}")
        iter_dir = workdir / f"iter_{cround}"
        iter_dir.mkdir(exist_ok=True)

        # Stronger plan repair each round
        plan = infer_rooms(plan, image)
        plan, health, applied = auto_repair(
            plan,
            assess_plan(plan, threshold=plan_health_threshold),
            image_bgr=image,
            ocr_items=plan.meta.get("ocr_items") or [],
        )
        corrections.extend(applied)
        if plan.scale_px_per_m is None:
            plan = estimate_scale_from_ocr(plan, plan.meta.get("ocr_items") or [])
        fidelity = assess_plan_vs_source(plan, image, expect_min_rooms=5)
        plan.save(str(workdir / "plan.json"))
        plan.save(str(iter_dir / "plan.json"))

        layout = build_layout(
            plan, daylight=daylight, fengshui=fengshui, common_sense=common_sense
        )
        layout.save(str(workdir / "layout.json"))
        layout.save(str(iter_dir / "layout.json"))
        write_control_bundle(
            plan, workdir / "control", source_image=image, layout=layout
        )

        # Regenerate t2i with higher strength
        cur_strength = min(0.95, base_strength + 0.05 * cround)
        try:
            t2i_path = run_t2i(
                plan,
                prompt,
                workdir / "t2i.png",
                provider=prov,
                seed=42 + cround,
                strength=cur_strength,
                layout=layout,
            )
            # also keep per-iter copy
            img = cv2.imread(str(t2i_path))
            if img is not None:
                cv2.imwrite(str(iter_dir / "t2i.png"), img)
            v = validate_file(
                t2i_path,
                plan,
                threshold=structure_threshold,
                overlay_out=iter_dir / "t2i_overlay.png",
                layout=layout,
            )
            validations[f"t2i_correct_{cround}"] = v.to_dict()
            best_v = v
            best_t2i = Path(t2i_path)
            img_fid = assess_image_vs_source(img, image, label=f"t2i_r{cround}")
            validations[f"t2i_vs_source_r{cround}"] = img_fid.to_dict()
        except Exception as exc:  # noqa: BLE001
            report.stages.append(
                StageResult(f"correct_t2i_{cround}", False, 0, {"error": str(exc)})
            )

        # Refresh 3D / VR scaffold from corrected plan
        try:
            from cad_ai_lock.perspective import (
                living_vr_prompt,
                place_living_camera,
                render_perspective_control,
            )
            from cad_ai_lock.render3d import build_3d_controls

            cam = layout.camera or place_living_camera(plan)
            plan.camera = cam
            paths3d, bundle3d, glb = build_3d_controls(
                plan,
                workdir / "ctrl3d",
                camera=cam,
                prefer_blender=False,
                width=720,
                height=1280,
            )
            report.artifacts["gltf"] = str(glb)
            report.artifacts.update({f"ctrl3d_{k}": v for k, v in paths3d.items()})
            cv2.imwrite(str(workdir / "vr_living.png"), bundle3d.beauty)
            cv2.imwrite(str(iter_dir / "vr_living.png"), bundle3d.beauty)
            persp, contract = render_perspective_control(plan, cam, layout=layout)
            cv2.imwrite(str(workdir / "control" / "control_perspective.png"), persp)
            (workdir / "view_contract.json").write_text(
                json.dumps(contract.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (workdir / "vr_prompt.txt").write_text(
                living_vr_prompt(contract, prompt), encoding="utf-8"
            )
            from cad_ai_lock.tour import render_tour_parallel

            tour = render_tour_parallel(
                plan, workdir / "tour", width=720, height=1280
            )
            report.artifacts["tour_json"] = str(workdir / "tour" / "tour.json")
            report.artifacts["tour_html"] = str(workdir / "tour" / "tour.html")
            report.stages.append(
                StageResult(
                    f"correct_tour_{cround}",
                    True,
                    0,
                    {
                        "rooms": tour.meta.get("rooms_rendered"),
                        "hotspots": len(tour.hotspots),
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001
            report.stages.append(
                StageResult(f"correct_3d_{cround}", False, 0, {"error": str(exc)})
            )

        report.correction_rounds = cround
        report.stages.append(
            StageResult(
                f"correct_round_{cround}",
                fidelity.passed,
                0,
                {
                    "fidelity": fidelity.to_dict(),
                    "t2i_vs_source": img_fid.to_dict() if img_fid else None,
                    "applied": applied,
                },
            )
        )
        report.iteration_log.append(
            {
                "iter": cround,
                "phase": "correct",
                "fidelity_plan": fidelity.to_dict(),
                "fidelity_t2i": img_fid.to_dict() if img_fid else None,
                "strength": cur_strength,
            }
        )
        best_fidelity_score = max(best_fidelity_score, float(fidelity.score))
        (workdir / "fidelity_plan.json").write_text(
            json.dumps(fidelity.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if fidelity.passed and (img_fid is None or img_fid.passed):
            corrections.append(f"correct_round_{cround}_locked")
            break

    report.validations = validations
    report.corrections = list(dict.fromkeys(corrections))
    report.plan_health = health.to_dict() if hasattr(health, "to_dict") else report.plan_health

    fidelity_ok = bool(fidelity.passed)
    t2i_ok = bool(best_v and best_v.passed)
    vr_ok = (workdir / "vr_living.png").exists()
    has_draft = (workdir / "t2i.png").exists() and (workdir / "plan.json").exists()

    # Always succeed at delivering artifacts; grade quality separately
    report.success = has_draft
    report.locked = fidelity_ok and t2i_ok
    if report.locked:
        report.quality_grade = "locked"
        report.message = (
            f"OK — locked to source after {report.correction_rounds} correction round(s)"
        )
    elif best_fidelity_score > fidelity0_score + 0.02 or report.correction_rounds > 0:
        report.quality_grade = "improved"
        report.message = (
            f"DRAFT+IMPROVED — delivered outputs; fidelity "
            f"{fidelity0_score:.2f}→{best_fidelity_score:.2f} after "
            f"{report.correction_rounds} round(s); continue hillclimb or tighten extract"
        )
    else:
        report.quality_grade = "draft"
        report.message = (
            "DRAFT — outputs generated; fidelity still weak vs source. "
            "Artifacts saved under iter_*; next rounds will keep correcting."
        )

    report.artifacts["warning"] = (
        "quality_grade=locked means structure-safe to ship. "
        "draft/improved are still downloadable for review. "
        "Soft GenerateImage ≠ ControlNet."
    )
    report.artifacts["run_report"] = str(workdir / "run_report.json")
    report.save(workdir / "run_report.json")
    return report
