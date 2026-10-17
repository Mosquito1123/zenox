"""Structured room geometry models (walls, openings, RoomPlan JSON schema).

Locking strategy (step 1):
  CAD PDF/JPG → RoomPlan JSON. Downstream generation must treat this as
  immutable structure: walls/doors/windows coordinates stay fixed;
  only style and furniture may change.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class OpeningType(str, Enum):
    DOOR = "door"
    WINDOW = "window"
    OPENING = "opening"  # pass-through / arch


class Point2D(BaseModel):
    x: float
    y: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def as_int_tuple(self) -> tuple[int, int]:
        return (int(round(self.x)), int(round(self.y)))


class Wall(BaseModel):
    """A wall segment in image / plan coordinates (pixels unless scaled)."""

    id: str
    start: Point2D
    end: Point2D
    thickness_px: float = Field(default=8.0, ge=1.0)
    exterior: bool = False

    @property
    def length_px(self) -> float:
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        return float((dx * dx + dy * dy) ** 0.5)


class Opening(BaseModel):
    """Door / window / opening placed on or near a wall."""

    id: str
    type: OpeningType
    start: Point2D
    end: Point2D
    wall_id: str | None = None
    width_m: float | None = None  # physical width if scale known
    swing: str | None = None  # e.g. "left_in" for doors

    @property
    def length_px(self) -> float:
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        return float((dx * dx + dy * dy) ** 0.5)


class CameraSpec(BaseModel):
    """Optional camera for plan→perspective locking (effect views / video)."""

    preset: str = "living_one_point"
    # image-space vanishing point & horizon (pixels in plan or render canvas)
    vanishing_point: Point2D | None = None
    horizon_y: float | None = None
    camera_height_m: float = 1.5
    wall_height_m: float = 2.8
    notes: str | None = None


class FurnitureItem(BaseModel):
    """One furniture instance in plan coordinates (soft layer, not structure)."""

    id: str
    type: str
    room_id: str | None = None
    center: Point2D
    width_px: float
    depth_px: float
    rotation_deg: float = 0.0  # 0 = +x
    face: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class FurnitureLayout(BaseModel):
    """Furniture placement derived from RoomPlan + policy packs."""

    version: str = "1.0"
    plan_revision: str = "1"
    policies: list[str] = Field(default_factory=list)
    camera: CameraSpec | None = None
    items: list[FurnitureItem] = Field(default_factory=list)
    prompt_hints: list[str] = Field(default_factory=list)
    scorecard: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None

    def save(self, path: str) -> None:
        import json
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_file(cls, path: str) -> FurnitureLayout:
        import json
        from pathlib import Path

        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


class RoomPlan(BaseModel):
    """Full structured floorplan used to lock space for generation."""

    version: str = "1.0"
    source_path: str | None = None
    image_width: int = Field(ge=1)
    image_height: int = Field(ge=1)
    # pixels per meter; None if unknown (relative lock only)
    scale_px_per_m: float | None = Field(default=None, gt=0)
    walls: list[Wall] = Field(default_factory=list)
    openings: list[Opening] = Field(default_factory=list)
    rooms: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Optional named regions: {id, name, polygon:[{x,y}]}",
    )
    # soft layer selectors (product toggles); geometry stays immutable
    policies: list[str] = Field(default_factory=list)
    plan_revision: str = "1"
    camera: CameraSpec | None = None
    notes: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("walls")
    @classmethod
    def _need_walls_warn(cls, v: list[Wall]) -> list[Wall]:
        return v

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_file(cls, path: str) -> RoomPlan:
        import json
        from pathlib import Path

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def save(self, path: str) -> None:
        import json
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self.to_json_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def bounding_box(self) -> tuple[float, float, float, float]:
        xs: list[float] = []
        ys: list[float] = []
        for w in self.walls:
            xs.extend([w.start.x, w.end.x])
            ys.extend([w.start.y, w.end.y])
        for o in self.openings:
            xs.extend([o.start.x, o.end.x])
            ys.extend([o.start.y, o.end.y])
        if not xs:
            return (0.0, 0.0, float(self.image_width), float(self.image_height))
        return (min(xs), min(ys), max(xs), max(ys))
