# CAD AI Lock — 自动化装修流水线

> 完整开源工作流说明见 **[WORKFLOW.md](./WORKFLOW.md)**（阶段、数据契约、Tour、Hillclimb、复现命令）。

```
CAD/PDF/JPG
  → Extract + Assess/Repair
  → **先出草稿** layout / controls / t2i / glTF / VR
  → **分房间漫游** 每房间相机位姿 → 任务队列并行渲 → 拼导航/热点
  → fidelity 对照原户型打分
  → **不中断**：weak 则 hillclimb 修正轮（repair→重建控制→重生成）⟲
  → 交付 grade：locked | improved | draft（均有可下载产物）
  → run_report.json + tour/ + iter_0_draft / iter_N/
```

体验原则：**始终先生成，再逐步修正**；不对齐原户型时标 `draft/improved`，而不是直接失败空白。

```bash
python -m cad_ai_lock.cli extrude --plan plan.json -o out/scene.glb
python -m cad_ai_lock.cli render3d --plan plan.json -o out/ctrl3d
# 有 Blender 时：  --blender
# 分房间并行 VR 漫游（相机位姿 → 队列并行渲 → 热点导航）
python -m cad_ai_lock.cli tour --plan plan.json -o out/tour
# 打开 out/tour/tour.html
```

**主路径不接受手改 `plan.json`。** `--plan-override` 仅供运维兜底。

## 两层锁定 + 自动修正

| 层 | 内容 | 失败时 |
|----|------|--------|
| 硬锁 | 墙门窗、控制图、禁止改户型提示词 | validate fail → 加强 strength 重生成 |
| 软约束 | 布局模板、走道净空、采光、可选风水 | 打分；过低则 rebuild_layout |
| 拓扑 | assess 墙段数/正交/连通/房间/比例尺 | repair 动作表自动修 |

## 安装

```bash
cd cad-ai-lock
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install easyocr   # 可选：尺寸标注 / 中文房间名 OCR
cp .env.example .env
```

## CLI

```bash
# 一键流水线（自动修正闭环）
python -m cad_ai_lock.cli pipeline path/to/floorplan.png \
  -p "现代简约装修" --provider mock -w out/run --ocr

# 分步
python -m cad_ai_lock.cli extract floorplan.png -o plan.json --control out/control
python -m cad_ai_lock.cli assess plan.json
python -m cad_ai_lock.cli repair plan.json -i floorplan.png -o plan.fixed.json
python -m cad_ai_lock.cli layout --plan plan.json -o layout.json --control out/control
python -m cad_ai_lock.cli t2i --plan plan.json --layout layout.json -p "现代简约" -o out/t2i.png
python -m cad_ai_lock.cli validate out/t2i.png --plan plan.json --layout layout.json
```

## 自动修正动作表

| 动作 | 触发 |
|------|------|
| `orthogonal_snap` | 非正交墙过多 |
| `merge_collinear` / `dedupe_walls` | 碎线/重复墙 |
| `grid_snap` / `snap_junctions` | 端点不闭合 |
| `infer_rooms` / `label_rooms` | 无房间或无名 |
| `estimate_scale` | 无比例尺（OCR mm） |
| `infer_openings` / `attach_openings` | 门窗缺失/游离 |
| `re_extract_walls` | 包络异常 → 整段重提 |
| `rebuild_layout` | 布局软分过低 |
| `t2i_retry_strengthen_*` | 生成结构校验失败 |

全程写入 `meta.repair_log` 与 `run_report.json`。

## 环境变量

| 变量 | 说明 |
|------|------|
| `CAD_AI_PROVIDER` | `mock` / `replicate` / `fal` / `local` |
| `REPLICATE_API_TOKEN` / `FAL_KEY` / `OPENAI_API_KEY` | 后端 Key |
| `CONTROL_STRENGTH` | 默认 `0.85` |
| `POLICY_*` / `STRUCTURE_THRESHOLD` / `POLICY_THRESHOLD` | 策略与校验 |

## 目录

```
src/cad_ai_lock/
  extract.py      hybrid ink + Hough + LSD + OCR
  assess.py       拓扑健康度
  repair.py       自动修正动作
  perspective.py  ViewContract + 2D 透视线（调试/提示锁）
  extrude.py      RoomPlan→米制 mesh→glTF
  render3d.py     mesh→depth/normal/seg/lineart（software；可选 Blender）
  tour.py         每房间相机 → ThreadPool 并行渲 → hotspots + tour.html
  ocr_utils.py    EasyOCR（可选）
  pipeline.py     闭环编排（先草稿再 hillclimb）
  layout.py       软布局
  control.py      多控制图
  generate.py     mock / replicate / fal / local
  validate.py     结构硬校验
  cli.py
```

## 注意

- OCR 为可选依赖；无 EasyOCR 时仍可跑 CV 提取 + 纹理房间分类，比例尺可 `--scale` 传入。
- mock 只验证闭环与控制图锁定；上线请接真实 ControlNet（Replicate/FAL/local）。
- VR/透视静帧由 `CameraSpec` + 生成端一点透视 prompt 产出，不是全景 360。
- 多房间漫游：`tour` 为每间房独立相机并行渲 beauty/depth，再按门邻接拼热点导航；非并发全景 VR。
- 风水包默认关闭，不覆盖消防/通行常识。

## 安全

密钥只放本地 `.env`（参考 `.env.example`），详见 [SECURITY.md](./SECURITY.md)。`out/` 与 `.env` 均已 gitignore。

从 monorepo 分支同步到独立仓库（本机已 `gh auth login`）：

```bash
# 先拉最新分支里的脚本，或直接：
gh repo clone Mosquito1123/zenox /tmp/zenox -- -b cursor/cad-ai-lock-auto-pipeline-74b7 --depth 1
bash /tmp/zenox/cad-ai-lock/scripts/sync-to-github.sh
```

## License

MIT
