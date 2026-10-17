"""CLI: extract | layout | t2i | i2i | validate | extrude | render3d | tour | pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="cad-ai-lock",
    help="CAD 几何硬锁 + 布局策略软约束，再做文生图 / 图生图。",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


@app.command("make-demo")
def make_demo(
    out: Path = typer.Option(
        Path("examples/sample_floorplan.jpg"),
        "--out",
        "-o",
        help="输出合成户型 JPG 路径",
    ),
) -> None:
    """生成一张带门窗的矩形演示户型图。"""
    from cad_ai_lock.extract import make_synthetic_floorplan

    path = make_synthetic_floorplan(out)
    console.print(f"[green]已写入演示户型[/green]: {path}")


@app.command("policies")
def policies_cmd() -> None:
    """列出可用的规则包。"""
    from cad_ai_lock.policies_loader import list_available_packs, load_pack

    for pid in list_available_packs():
        pack = load_pack(pid)
        n = len(pack.get("rules") or pack.get("templates") or {})
        console.print(
            f"- {pid}  (default={pack.get('enabled_by_default')}, entries≈{n})"
        )


@app.command("extract")
def extract_cmd(
    input_path: Path = typer.Argument(..., help="CAD 平面图 PDF / JPG / PNG"),
    out: Path = typer.Option(Path("plan.json"), "--out", "-o", help="RoomPlan JSON"),
    control_dir: Optional[Path] = typer.Option(
        Path("out/control"),
        "--control",
        "-c",
        help="输出 control maps 目录",
    ),
    pdf_page: int = typer.Option(0, "--pdf-page", help="PDF 页码（从 0 起）"),
    vision: bool = typer.Option(
        False, "--vision", help="用 OpenAI 兼容视觉 API 精修几何"
    ),
    scale: Optional[float] = typer.Option(
        None, "--scale", help="像素/米，用于物理尺寸（可空，OCR 自动估）"
    ),
    auto_correct: bool = typer.Option(
        True, "--auto-correct/--no-auto-correct", help="提取后自动 assess→repair"
    ),
    ocr: bool = typer.Option(True, "--ocr/--no-ocr", help="OCR 读尺寸/房间名"),
) -> None:
    """识图 → RoomPlan JSON + ControlNet 控制图（结构层，默认自动修正）。"""
    from cad_ai_lock.control import write_control_bundle
    from cad_ai_lock.extract import extract_room_plan

    plan, image, _edges = extract_room_plan(
        input_path,
        pdf_page=pdf_page,
        use_vision=vision,
        scale_px_per_m=scale,
        auto_correct=auto_correct,
        use_ocr=ocr,
    )
    plan.save(str(out))
    console.print(
        f"[green]RoomPlan[/green]: {out}  "
        f"(walls={len(plan.walls)}, openings={len(plan.openings)}, "
        f"{plan.image_width}x{plan.image_height})"
    )
    if control_dir is not None:
        paths = write_control_bundle(plan, control_dir, source_image=image)
        for k, v in paths.items():
            console.print(f"  control[{k}] → {v}")


@app.command("layout")
def layout_cmd(
    plan_path: Path = typer.Option(..., "--plan", help="RoomPlan JSON"),
    out: Path = typer.Option(Path("layout.json"), "--out", "-o"),
    control_dir: Optional[Path] = typer.Option(
        Path("out/control"), "--control", "-c", help="刷新含家具的控制图"
    ),
    daylight: bool = typer.Option(True, "--daylight/--no-daylight"),
    fengshui: bool = typer.Option(False, "--fengshui/--no-fengshui"),
    common_sense: bool = typer.Option(True, "--common-sense/--no-common-sense"),
) -> None:
    """在锁定户型上跑常用布局/采光/常识（可选风水）→ FurnitureLayout。"""
    from cad_ai_lock.control import write_control_bundle
    from cad_ai_lock.geometry import RoomPlan
    from cad_ai_lock.layout import build_layout

    plan = RoomPlan.from_file(str(plan_path))
    layout = build_layout(
        plan,
        daylight=daylight,
        fengshui=fengshui,
        common_sense=common_sense,
    )
    layout.save(str(out))
    console.print(
        f"[green]FurnitureLayout[/green]: {out}  "
        f"items={len(layout.items)} policies={layout.policies} "
        f"score={layout.scorecard.get('overall')}"
    )
    for h in layout.prompt_hints:
        console.print(f"  hint: {h}")
    if control_dir is not None:
        paths = write_control_bundle(plan, control_dir, layout=layout)
        for k, v in paths.items():
            console.print(f"  control[{k}] → {v}")


@app.command("t2i")
def t2i_cmd(
    prompt: str = typer.Option(..., "--prompt", "-p", help="风格描述，如：现代简约客厅"),
    plan_path: Path = typer.Option(..., "--plan", help="RoomPlan JSON"),
    out: Path = typer.Option(Path("out/t2i.png"), "--out", "-o"),
    layout_path: Optional[Path] = typer.Option(
        None, "--layout", help="FurnitureLayout JSON（可选）"
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="mock|replicate|fal|local（默认读环境变量）"
    ),
    seed: Optional[int] = typer.Option(None, "--seed"),
    strength: Optional[float] = typer.Option(None, "--strength", help="Control 强度 0~1"),
) -> None:
    """文生图：强制条件于 plan 的 control 线，锁死空间。"""
    import os

    from cad_ai_lock.generate import get_provider, run_t2i
    from cad_ai_lock.geometry import FurnitureLayout, RoomPlan

    if provider:
        os.environ["CAD_AI_PROVIDER"] = provider
        from cad_ai_lock.config import get_settings

        get_settings.cache_clear()

    plan = RoomPlan.from_file(str(plan_path))
    layout = FurnitureLayout.from_file(str(layout_path)) if layout_path else None
    prov = get_provider()
    path = run_t2i(
        plan,
        prompt,
        out,
        provider=prov,
        strength=strength,
        seed=seed,
        layout=layout,
    )
    console.print(f"[green]t2i ({prov.name})[/green] → {path}")


@app.command("i2i")
def i2i_cmd(
    prompt: str = typer.Option(..., "--prompt", "-p"),
    image: Path = typer.Option(..., "--image", "-i", help="原房间图"),
    plan_path: Path = typer.Option(..., "--plan", help="RoomPlan JSON"),
    out: Path = typer.Option(Path("out/i2i.png"), "--out", "-o"),
    layout_path: Optional[Path] = typer.Option(None, "--layout"),
    provider: Optional[str] = typer.Option(None, "--provider"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    strength: Optional[float] = typer.Option(None, "--strength"),
) -> None:
    """图生图：在原图上换风格，墙体门窗由 plan/control 锁定。"""
    import os

    from cad_ai_lock.generate import get_provider, run_i2i
    from cad_ai_lock.geometry import FurnitureLayout, RoomPlan

    if provider:
        os.environ["CAD_AI_PROVIDER"] = provider
        from cad_ai_lock.config import get_settings

        get_settings.cache_clear()

    plan = RoomPlan.from_file(str(plan_path))
    layout = FurnitureLayout.from_file(str(layout_path)) if layout_path else None
    prov = get_provider()
    path = run_i2i(
        plan,
        image,
        prompt,
        out,
        provider=prov,
        strength=strength,
        seed=seed,
        layout=layout,
    )
    console.print(f"[green]i2i ({prov.name})[/green] → {path}")


@app.command("validate")
def validate_cmd(
    image: Path = typer.Argument(..., help="生成结果图"),
    plan_path: Path = typer.Option(..., "--plan", help="RoomPlan JSON"),
    layout_path: Optional[Path] = typer.Option(None, "--layout"),
    threshold: float = typer.Option(0.35, "--threshold", "-t"),
    policy_threshold: float = typer.Option(0.55, "--policy-threshold"),
    overlay: Path = typer.Option(Path("out/validate_overlay.png"), "--overlay"),
) -> None:
    """结构硬校验 +（可选）布局策略软打分。"""
    from cad_ai_lock.geometry import FurnitureLayout, RoomPlan
    from cad_ai_lock.validate import validate_file

    plan = RoomPlan.from_file(str(plan_path))
    layout = FurnitureLayout.from_file(str(layout_path)) if layout_path else None
    result = validate_file(
        image,
        plan,
        threshold=threshold,
        overlay_out=overlay,
        layout=layout,
        policy_threshold=policy_threshold,
    )
    table = Table(title="Validation (hard structure + soft policy)")
    table.add_column("metric")
    table.add_column("value")
    for k, v in result.to_dict().items():
        if k == "scorecard":
            continue
        table.add_row(k, str(v))
    console.print(table)
    if result.scorecard:
        console.print(json.dumps(result.scorecard, ensure_ascii=False, indent=2))
    if not result.passed:
        raise typer.Exit(code=2)


@app.command("extrude")
def extrude_cmd(
    plan_path: Path = typer.Option(..., "--plan", help="RoomPlan JSON"),
    out: Path = typer.Option(Path("out/scene.glb"), "--out", "-o", help="glTF/GLB 输出"),
    height: float = typer.Option(2.8, "--height", help="墙高（米）"),
) -> None:
    """RoomPlan 挤出为真三维 glTF（几何内核）。"""
    from cad_ai_lock.extrude import export_plan_gltf
    from cad_ai_lock.geometry import RoomPlan

    plan = RoomPlan.from_file(str(plan_path))
    path, scene = export_plan_gltf(plan, out, wall_height_m=height)
    console.print(
        f"[green]glTF[/green] → {path}  parts={len(scene.parts)} "
        f"scale={scene.scale_px_per_m:.2f}px/m h={scene.wall_height_m}m"
    )


@app.command("render3d")
def render3d_cmd(
    plan_path: Path = typer.Option(..., "--plan", help="RoomPlan JSON"),
    out_dir: Path = typer.Option(Path("out/ctrl3d"), "--out", "-o"),
    gltf: Optional[Path] = typer.Option(None, "--gltf", help="已有 glTF；空则先挤出"),
    blender: bool = typer.Option(False, "--blender/--software", help="优先 Blender 无头渲染"),
    width: int = typer.Option(720, "--width"),
    height: int = typer.Option(1280, "--height"),
) -> None:
    """从真三维渲 depth/normal/seg/lineart 控制图（VR 硬锁输入）。"""
    from cad_ai_lock.geometry import RoomPlan
    from cad_ai_lock.render3d import build_3d_controls

    plan = RoomPlan.from_file(str(plan_path))
    paths, bundle, glb = build_3d_controls(
        plan,
        out_dir,
        gltf_path=gltf,
        prefer_blender=blender,
        width=width,
        height=height,
    )
    console.print(f"[green]3D controls[/green] gltf={glb} renderer={bundle.meta.get('renderer')}")
    for k, v in paths.items():
        console.print(f"  {k} → {v}")


@app.command("tour")
def tour_cmd(
    plan_path: Path = typer.Option(..., "--plan", help="RoomPlan JSON"),
    out_dir: Path = typer.Option(Path("out/tour"), "--out", "-o"),
    workers: int = typer.Option(0, "--workers", help="并行线程数；0=自动"),
    width: int = typer.Option(720, "--width"),
    height: int = typer.Option(1280, "--height"),
    min_area: int = typer.Option(1200, "--min-area", help="忽略过小房间的面积阈值（px²）"),
) -> None:
    """每房间相机位姿 → 任务队列并行渲 → 再拼导航/热点（tour.html）。"""
    from cad_ai_lock.geometry import RoomPlan
    from cad_ai_lock.tour import render_tour_parallel

    plan = RoomPlan.from_file(str(plan_path))
    tour = render_tour_parallel(
        plan,
        out_dir,
        max_workers=workers or None,
        width=width,
        height=height,
        min_area_px=min_area,
    )
    console.print(
        f"[green]tour[/green] rooms={tour.meta.get('rooms_rendered')} "
        f"hotspots={len(tour.hotspots)} workers={tour.meta.get('workers')} "
        f"start={tour.start_room}"
    )
    console.print(f"  tour.json → {out_dir / 'tour.json'}")
    console.print(f"  tour.html → {out_dir / 'tour.html'}")
    if tour.meta.get("errors"):
        for err in tour.meta["errors"]:
            console.print(f"  [yellow]warn[/yellow] {err}")


@app.command("assess")
def assess_cmd(
    plan_path: Path = typer.Argument(..., help="RoomPlan JSON"),
    threshold: float = typer.Option(0.62, "--threshold", "-t"),
) -> None:
    """评估 RoomPlan 拓扑健康度，并列出建议自动修正动作。"""
    from cad_ai_lock.assess import assess_plan
    from cad_ai_lock.geometry import RoomPlan

    plan = RoomPlan.from_file(str(plan_path))
    health = assess_plan(plan, threshold=threshold)
    console.print(json.dumps(health.to_dict(), ensure_ascii=False, indent=2))
    if not health.passed:
        raise typer.Exit(code=2)


@app.command("repair")
def repair_cmd(
    plan_path: Path = typer.Argument(..., help="RoomPlan JSON"),
    image: Optional[Path] = typer.Option(None, "--image", "-i", help="原户型图（房间分割/纹理）"),
    out: Path = typer.Option(Path("plan.repaired.json"), "--out", "-o"),
    threshold: float = typer.Option(0.62, "--threshold", "-t"),
) -> None:
    """对已有 plan 跑自动修正闭环（不手改坐标）。"""
    import cv2

    from cad_ai_lock.assess import assess_plan
    from cad_ai_lock.geometry import RoomPlan
    from cad_ai_lock.repair import auto_repair

    plan = RoomPlan.from_file(str(plan_path))
    img = cv2.imread(str(image), cv2.IMREAD_COLOR) if image else None
    health = assess_plan(plan, threshold=threshold)
    plan, health2, applied = auto_repair(plan, health, image_bgr=img)
    plan.save(str(out))
    console.print(
        json.dumps(
            {"applied": applied, "before": health.to_dict(), "after": health2.to_dict()},
            ensure_ascii=False,
            indent=2,
        )
    )
    console.print(f"[green]repaired[/green] → {out}")
    if not health2.passed:
        raise typer.Exit(code=2)


@app.command("pipeline")
def pipeline_cmd(
    input_path: Path = typer.Argument(..., help="CAD JPG/PDF"),
    prompt: str = typer.Option("现代简约客厅，木地板，落地灯", "--prompt", "-p"),
    workdir: Path = typer.Option(Path("out/pipeline"), "--workdir", "-w"),
    provider: str = typer.Option("mock", "--provider"),
    scale: Optional[float] = typer.Option(None, "--scale", help="像素/米（可空，OCR 自动估）"),
    daylight: bool = typer.Option(True, "--daylight/--no-daylight"),
    fengshui: bool = typer.Option(False, "--fengshui/--no-fengshui"),
    strength: Optional[float] = typer.Option(None, "--strength"),
    ocr: bool = typer.Option(True, "--ocr/--no-ocr"),
    vision: bool = typer.Option(False, "--vision/--no-vision"),
    retries: int = typer.Option(3, "--retries", help="生成失败自动加强 control 重试次数"),
    correct_rounds: int = typer.Option(
        3, "--correct-rounds", help="出图后逐步修正轮数（不中断，先出草稿）"
    ),
    plan_override: Optional[Path] = typer.Option(
        None,
        "--plan-override",
        help="运维覆盖：跳过自动提取（非主路径）",
    ),
) -> None:
    """先出草稿再逐步修正：extract → 生成 → fidelity → hillclimb 修正轮 → 始终交付产物。"""
    from cad_ai_lock.pipeline import run_commercial_pipeline

    report = run_commercial_pipeline(
        input_path,
        workdir=workdir,
        prompt=prompt,
        provider=provider,
        scale=scale,
        daylight=daylight,
        fengshui=fengshui,
        strength=strength,
        use_ocr=ocr,
        use_vision=vision,
        max_generate_retries=retries,
        max_correct_rounds=correct_rounds,
        plan_override=plan_override,
    )
    console.print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    grade = getattr(report, "quality_grade", "draft")
    if report.success and getattr(report, "locked", False):
        console.print(
            f"[green]pipeline LOCKED[/green] grade={grade} → {workdir}/run_report.json"
        )
    elif report.success:
        console.print(
            f"[cyan]pipeline DRAFT delivered[/cyan] grade={grade} "
            f"rounds={getattr(report, 'correction_rounds', 0)} → {workdir}/run_report.json"
        )
        console.print(f"  {report.message}")
    else:
        console.print(f"[red]pipeline hard failure[/red] → {workdir}/run_report.json")
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
