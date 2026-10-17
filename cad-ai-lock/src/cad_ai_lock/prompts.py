"""Prompt builders that forbid changing walls / doors / windows.

Locking strategy (step 3):
  Positive: style + furniture only (+ layout hints from policy layer).
  Negative: layout changes, moving walls, extra/missing windows/doors.
  Scrub user phrases that conflict with hard geometry lock.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from cad_ai_lock.geometry import FurnitureLayout, RoomPlan

LOCK_POSITIVE_EN = (
    "strictly keep the exact same floor plan layout, wall positions, door and "
    "window openings unchanged, same camera viewpoint if any, architectural "
    "structure locked, only change interior style materials and furniture"
)

LOCK_NEGATIVE_EN = (
    "changing layout, moving walls, moving doors, moving windows, extra window, "
    "extra door, missing door, missing window, different floorplan, warped walls, "
    "bent walls, new openings, demolished wall, open concept conversion, "
    "altered architecture, wrong proportions"
)

LOCK_POSITIVE_ZH = (
    "严格保持户型与墙体门窗位置不变，空间结构锁定，只改变装修风格、材质与家具摆设"
)

LOCK_NEGATIVE_ZH = (
    "改变户型,移动墙体,移动门窗,增减门窗,拆墙,打通空间,扭曲墙线,比例失调,另一套平面图"
)

# User prompt fragments that fight the hard lock — strip / soften
_CONFLICT_PATTERNS = [
    (re.compile(r"打通|拆墙|砸墙|开放式厨房改造墙| bulldoze|knock down wall", re.I), ""),
    (re.compile(r"挪门|改门洞|加一扇窗|封窗", re.I), ""),
    (re.compile(r"change the floorplan|move the walls|remove wall", re.I), ""),
]


@dataclass
class PromptBundle:
    prompt: str
    negative_prompt: str
    prompt_zh: str
    negative_prompt_zh: str
    scrubbed: list[str]


def scrub_user_prompt(user_prompt: str) -> tuple[str, list[str]]:
    text = user_prompt.strip()
    removed: list[str] = []
    for pat, repl in _CONFLICT_PATTERNS:
        if pat.search(text):
            removed.append(pat.pattern)
            text = pat.sub(repl, text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,，。")
    return text, removed


def build_prompts(
    user_prompt: str,
    plan: RoomPlan | None = None,
    *,
    mode: str = "t2i",
    layout: FurnitureLayout | None = None,
) -> PromptBundle:
    """Compose bilingual prompts with hard layout-lock + soft layout hints."""
    user, scrubbed = scrub_user_prompt(user_prompt)
    n_walls = len(plan.walls) if plan else 0
    n_doors = (
        sum(1 for o in plan.openings if o.type.value == "door") if plan else 0
    )
    n_windows = (
        sum(1 for o in plan.openings if o.type.value == "window") if plan else 0
    )
    geo_hint = ""
    if plan:
        geo_hint = (
            f" This plan has {n_walls} wall segments, {n_doors} doors, "
            f"{n_windows} windows — preserve all of them. "
        )

    layout_hint_en = ""
    layout_hint_zh = ""
    if layout and layout.prompt_hints:
        layout_hint_zh = "；".join(layout.prompt_hints)
        layout_hint_en = " Layout: " + "; ".join(layout.prompt_hints) + "."
        if layout.items:
            types = ", ".join(sorted({i.type for i in layout.items}))
            layout_hint_en += f" Place these furnishings: {types}."

    mode_hint = {
        "t2i": "photorealistic interior render from floorplan control lines, ",
        "i2i": "restyle this room photo while locking geometry from control map, ",
    }.get(mode, "")

    prompt_en = (
        f"{mode_hint}{user}. {LOCK_POSITIVE_EN}.{geo_hint}{layout_hint_en}"
    ).strip()
    prompt_zh = f"{user}。{LOCK_POSITIVE_ZH}。"
    if plan:
        prompt_zh += f"平面含{n_walls}段墙、{n_doors}门、{n_windows}窗，全部保留。"
    if layout_hint_zh:
        prompt_zh += layout_hint_zh + "。"

    return PromptBundle(
        prompt=prompt_en,
        negative_prompt=LOCK_NEGATIVE_EN,
        prompt_zh=prompt_zh,
        negative_prompt_zh=LOCK_NEGATIVE_ZH,
        scrubbed=scrubbed,
    )
