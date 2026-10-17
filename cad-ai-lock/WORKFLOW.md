# CAD AI Lock — 开源工作流

> 户型图 → 几何硬锁 → 先出草稿 → 分房间并行漫游 → 对照原图打分 → 不中断 hillclimb 修正  
> 版本：`0.4.3` · License：MIT

## 一句话

**不手改坐标。** 从 CAD/JPG 自动提取 `RoomPlan`，挤出真三维 glTF，渲控制图，再按「每房间相机 → 任务队列并行渲 → 拼导航/热点」产出可点的 `tour.html`；生成端必须吃控制图（ControlNet），禁止自由生图跑偏。

---

## 总览图

```
CAD / PDF / JPG
    │
    ▼
① Extract          墨线 + Hough/LSD + OCR → RoomPlan JSON
    │
    ▼
② Assess / Repair  拓扑打分；不达标则自动修正（不手改）
    │
    ▼
③ Layout           软布局（采光 / 常识 / 可选风水）→ FurnitureLayout
    │
    ▼
④ Controls         2D 多控制图（canny / depth / seg / perspective…）
    │
    ├──────────────────────────────┐
    ▼                              ▼
⑤ Generate (t2i/i2i)          ⑥ Extrude → glTF
   ControlNet 硬锁墙门窗          RoomPlan → 米制 mesh
    │                              │
    ▼                              ▼
⑦ Validate                     ⑧ Render3D
   结构硬校验                      depth / normal / seg / lineart / beauty
    │                              │
    │                              ▼
    │                          ⑨ Tour（分房间漫游）
    │                              每房间相机位姿
    │                                → ThreadPool 并行渲
    │                                → 门邻接热点
    │                                → tour.json + tour.html
    │                              │
    └──────────────┬───────────────┘
                   ▼
              ⑩ Fidelity
                 对照原户型打分
                   │
           weak? ──┤
                   ▼
              ⑪ Hillclimb（不中断）
                 repair → 重建控制 → 重生成 → 再打分 ⟲
                   │
                   ▼
              ⑫ Deliver
                 grade: locked | improved | draft
                 产物: run_report.json / iter_* / tour/ / glTF / t2i
```

---

## 设计原则（开源必读）

| 原则 | 说明 |
|------|------|
| **几何硬锁** | 墙/门/窗坐标进 `RoomPlan` 后不可被生成端改写；只允许风格与家具变化 |
| **先出草稿** | 不对齐也不空白失败；始终交付可下载产物，用 `quality_grade` 标质量 |
| **自动修正** | 主路径禁止手改 `plan.json`；`--plan-override` 仅运维兜底 |
| **真三维内核** | VR/控制图来自 mesh（glTF），不是纯 2D 线稿幻想 |
| **分房间并行** | 漫游 = 每间独立相机并行渲，再拼热点；不是单相机串行、也不是全景 360 |
| **软生成 ≠ 锁户型** | `mock` / 无 ControlNet 的 GenerateImage 只能验闭环，不能当正式交付 |

---

## 阶段说明

### ① Extract — `extract.py`

- 输入：户型 JPG / PNG / PDF  
- 手段：墨线分割 + Hough / LSD 线段 +（可选）EasyOCR 读尺寸与房间名  
- 输出：`plan.json`（`RoomPlan`）+ 初始 control 目录  
- CLI：`cad-ai-lock extract floorplan.png -o plan.json --control out/control`

### ② Assess / Repair — `assess.py` / `repair.py`

- 健康度：墙段数、正交率、端点闭合、房间数、比例尺、门窗附着  
- 动作表（自动）：`orthogonal_snap` / `merge_collinear` / `grid_snap` / `infer_rooms` / `estimate_scale` / `infer_openings` / …  
- CLI：`assess plan.json` · `repair plan.json -i floorplan.png -o plan.fixed.json`

### ③ Layout — `layout.py`

- 在锁定户型上放家具模板；采光 / 通行常识默认开；风水默认关  
- 输出：`layout.json` + 刷新后的 control  
- CLI：`layout --plan plan.json -o layout.json`

### ④ Controls — `control.py` / `perspective.py`

- 2D：canny、segmentation、家具占位、一点透视线稿  
- `ViewContract`：远墙/左/右开口清单，写入 prompt 硬锁文案

### ⑤ Generate — `generate.py`

- Provider：`mock` | `replicate` | `fal` | `local`  
- t2i / i2i 必须带 control；`CONTROL_STRENGTH` 默认 `0.85`  
- CLI：`t2i --plan plan.json --layout layout.json -p "现代简约" -o out/t2i.png`

### ⑥ Extrude — `extrude.py`

- `RoomPlan` → 米制墙体/地板/门窗 mesh → `.glb` + meta JSON  
- CLI：`extrude --plan plan.json -o out/scene.glb`

### ⑦ Validate — `validate.py`

- 生成图 vs plan 结构硬校验；可选布局软分  
- 失败 → pipeline 内提高 strength 重试

### ⑧ Render3D — `render3d.py`

- 从 mesh 软件光栅（默认）或 Blender 无头渲：  
  `depth` / `normal` / `segmentation` / `lineart` / `beauty`  
- CLI：`render3d --plan plan.json -o out/ctrl3d`（可选 `--blender`）

### ⑨ Tour — `tour.py` ★ 分房间漫游

```
RoomPlan.rooms
  → 每间：门口内侧相机，看向房间深处
  → ThreadPool 并行：
       · beauty = 一点透视几何示意图（人看）
       · depth/normal/… = mesh 控制图（给 ControlNet）
  → 门邻接（否则贴邻）→ Hotspot 双向边
  → tour.json + tour.html + tour_contact.png
```

- CLI：`tour --plan plan.json -o out/tour`  
- 浏览器打开：`out/tour/tour.html`（房间按钮 + 热点跳转）

### ⑩ Fidelity — `fidelity.py`

- `plan` / 生成图对照原户型：房间数、覆盖、结构一致性  
- 弱分不中断，进入 hillclimb

### ⑪ Hillclimb — `pipeline.py`

- 每轮：`infer_rooms` → `auto_repair` → 重建 layout/controls → 重生成 → 刷新 3D/tour → 再打分  
- 轮次默认 `--correct-rounds 3`

### ⑫ Deliver

| `quality_grade` | 含义 |
|-----------------|------|
| `locked` | 对照原户型通过，可正式交付 |
| `improved` | 有提升但仍弱，可下载继续修 |
| `draft` | 草稿已出，需继续 hillclimb / 加强提取 |

产物目录要点：

```
out/run/
  plan.json  layout.json  t2i.png  vr_living.png
  scene.glb / ctrl3d/
  tour/           ← tour.html tour.json <room_id>/
  iter_0_draft/   ← 首轮草稿快照
  iter_N/         ← 修正轮快照
  run_report.json
  fidelity_plan.json
```

---

## 一键流水线

```bash
cd cad-ai-lock
python -m venv .venv && source .venv/bin/activate
pip install -e .
# 可选 OCR
pip install easyocr

python -m cad_ai_lock.cli pipeline path/to/floorplan.png \
  -p "现代简约装修，木地板，柔和自然光" \
  --provider mock \
  -w out/run \
  --ocr \
  --correct-rounds 3
```

接真实生成（开源下游自行配置 Key）：

```bash
export CAD_AI_PROVIDER=replicate   # 或 fal / local
export REPLICATE_API_TOKEN=...
python -m cad_ai_lock.cli pipeline floorplan.png -p "现代简约" -w out/run
```

---

## 分步调试路径

```bash
# 1) 提取 + 自动修正
python -m cad_ai_lock.cli extract floorplan.png -o plan.json --control out/control

# 2) 健康度
python -m cad_ai_lock.cli assess plan.json

# 3) 布局
python -m cad_ai_lock.cli layout --plan plan.json -o layout.json

# 4) 真三维
python -m cad_ai_lock.cli extrude --plan plan.json -o out/scene.glb
python -m cad_ai_lock.cli render3d --plan plan.json -o out/ctrl3d

# 5) 分房间漫游
python -m cad_ai_lock.cli tour --plan plan.json -o out/tour
# open out/tour/tour.html

# 6) 生成 + 校验（需真实 provider）
python -m cad_ai_lock.cli t2i --plan plan.json --layout layout.json \
  -p "现代简约客厅" -o out/t2i.png
python -m cad_ai_lock.cli validate out/t2i.png --plan plan.json --layout layout.json
```

---

## 数据契约（开源对接点）

### `RoomPlan`（硬锁层）

- `walls[]` / `openings[]` / `rooms[]` / `scale_px_per_m` / `camera`  
- 下游生成**不得**改墙门窗坐标

### `FurnitureLayout`（软层）

- `items[]` + `policies` + `scorecard`  
- 可重建，不影响结构硬锁

### `tour.json`（漫游层）

```json
{
  "start_room": "room_5",
  "rooms": [
    {
      "room_id": "room_5",
      "name": "客厅",
      "type": "living",
      "camera": { "eye_px": {}, "target_px": {}, "eye_m": [], "target_m": [] },
      "beauty": ".../room_5/beauty.png",
      "depth": ".../room_5/depth.png"
    }
  ],
  "hotspots": [
    { "from": "room_5", "to": "room_2", "label": "厨房", "u": 0.85, "v": 0.55 }
  ],
  "meta": { "workers": 4, "rooms_rendered": 13 }
}
```

### `run_report.json`（交付层）

- `success` / `locked` / `quality_grade` / `correction_rounds`  
- `stages[]` / `artifacts{}` / `iteration_log[]`

---

## 模块地图

| 文件 | 职责 |
|------|------|
| `extract.py` | 识图 → RoomPlan |
| `assess.py` / `repair.py` | 拓扑评估与自动修正 |
| `layout.py` / `policies/*.yaml` | 软布局策略包 |
| `control.py` / `perspective.py` | 2D 控制图 + ViewContract |
| `extrude.py` | RoomPlan → glTF；单房间 shell |
| `render3d.py` | mesh → 3D 控制图 |
| `tour.py` | 相机 → 并行渲 → 热点 HTML |
| `generate.py` | mock / replicate / fal / local |
| `validate.py` / `fidelity.py` | 结构校验 / 对照原图 |
| `pipeline.py` | 流水线编排（先草稿再 hillclimb） |
| `cli.py` | Typer 入口 |

---

## 环境变量

| 变量 | 说明 |
|------|------|
| `CAD_AI_PROVIDER` | `mock` / `replicate` / `fal` / `local` |
| `REPLICATE_API_TOKEN` / `FAL_KEY` / `OPENAI_API_KEY` | 后端密钥 |
| `CONTROL_STRENGTH` | 默认 `0.85` |
| `STRUCTURE_THRESHOLD` / `POLICY_THRESHOLD` | 校验阈值 |

---

## 非目标 / 边界

- **不是**全景 360 VR；是分房间一点透视静帧 + 热点跳转  
- **不是**手改 CAD 坐标的工作流  
- **mock provider** 只验证管线，不产生可售卖装修图  
- 无 Blender 时默认 numpy 软件光栅；有 Blender 可 `--blender`  
- OCR（EasyOCR）可选；无 OCR 时用 `--scale` 传像素/米  

---

## 最小复现（给 Contributor）

```bash
python -m cad_ai_lock.cli make-demo -o examples/sample_floorplan.jpg
python -m cad_ai_lock.cli pipeline examples/sample_floorplan.jpg \
  -p "现代简约" --provider mock -w /tmp/cad-demo --correct-rounds 1
python -m cad_ai_lock.cli tour --plan /tmp/cad-demo/plan.json -o /tmp/cad-demo/tour
```

打开 `/tmp/cad-demo/tour/tour.html` 与 `/tmp/cad-demo/run_report.json` 即可验收闭环。
