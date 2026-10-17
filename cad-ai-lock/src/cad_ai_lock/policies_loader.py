"""Load and select layout / daylight / fengshui rule packs."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

POLICY_DIR = Path(__file__).resolve().parent / "policies"

CORE_PACKS = ("common_sense", "daylight", "layout_templates")
OPTIONAL_PACKS = ("fengshui_optional",)


def policy_path(pack_id: str) -> Path:
    return POLICY_DIR / f"{pack_id}.yaml"


@lru_cache
def load_pack(pack_id: str) -> dict[str, Any]:
    path = policy_path(pack_id)
    if not path.exists():
        raise FileNotFoundError(f"policy pack not found: {pack_id} ({path})")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data["id"] = data.get("id", pack_id)
    return data


def resolve_packs(
    *,
    daylight: bool = True,
    fengshui: bool = False,
    common_sense: bool = True,
    templates: bool = True,
    extra: list[str] | None = None,
) -> list[str]:
    packs: list[str] = []
    if common_sense:
        packs.append("common_sense")
    if daylight:
        packs.append("daylight")
    if templates:
        packs.append("layout_templates")
    if fengshui:
        packs.append("fengshui_optional")
    for p in extra or []:
        if p not in packs:
            packs.append(p)
    return packs


def load_packs(pack_ids: list[str]) -> list[dict[str, Any]]:
    return [load_pack(p) for p in pack_ids]


def list_available_packs() -> list[str]:
    return sorted(p.stem for p in POLICY_DIR.glob("*.yaml"))
