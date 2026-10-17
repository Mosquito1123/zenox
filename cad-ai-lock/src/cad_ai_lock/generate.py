"""Text-to-image and image-to-image with ControlNet-style locking.

Providers:
  - mock: offline deterministic composite (control lines + fill + text)
  - replicate: SDXL / ControlNet canny via Replicate HTTP API
  - fal: fal.ai Flux / ControlNet endpoints
  - local: optional diffusers stub (raises clear error if torch missing)

Locking strategy (step 4):
  Always pass control_canny (or structure) + high control_strength.
  Negative prompt forbids layout changes.
"""

from __future__ import annotations

import hashlib
import io
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any  # noqa: F401 — used by run_t2i/run_i2i layout arg

import cv2
import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cad_ai_lock.config import ProviderName, Settings, get_settings
from cad_ai_lock.control import render_canny_from_plan, render_structure_map
from cad_ai_lock.geometry import RoomPlan
from cad_ai_lock.prompts import PromptBundle, build_prompts


def _seed_from(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _bgr_to_pil(img: np.ndarray) -> Image.Image:
    if img.ndim == 2:
        return Image.fromarray(img)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _encode_png_bytes(img_bgr_or_gray: np.ndarray) -> bytes:
    if img_bgr_or_gray.ndim == 2:
        ok, buf = cv2.imencode(".png", img_bgr_or_gray)
    else:
        ok, buf = cv2.imencode(".png", img_bgr_or_gray)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def _data_uri_png(img: np.ndarray) -> str:
    import base64

    b64 = base64.b64encode(_encode_png_bytes(img)).decode("ascii")
    return f"data:image/png;base64,{b64}"


class ImageProvider(ABC):
    name: str

    @abstractmethod
    def text_to_image(
        self,
        prompts: PromptBundle,
        control_canny: np.ndarray,
        plan: RoomPlan,
        *,
        strength: float,
        steps: int,
        guidance: float,
        seed: int | None,
    ) -> np.ndarray:
        ...

    @abstractmethod
    def image_to_image(
        self,
        prompts: PromptBundle,
        source_bgr: np.ndarray,
        control_canny: np.ndarray,
        plan: RoomPlan,
        *,
        strength: float,
        steps: int,
        guidance: float,
        seed: int | None,
    ) -> np.ndarray:
        ...


class MockProvider(ImageProvider):
    """Deterministic offline demo: control lines + style-tinted fill + overlay text."""

    name = "mock"

    def _compose(
        self,
        prompts: PromptBundle,
        control_canny: np.ndarray,
        plan: RoomPlan,
        source_bgr: np.ndarray | None,
        seed: int,
        mode: str,
    ) -> np.ndarray:
        h, w = plan.image_height, plan.image_width
        rng = np.random.default_rng(seed)

        # style tint from prompt hash
        hue = seed % 180
        base = np.zeros((h, w, 3), dtype=np.uint8)
        base[:] = (int(40 + hue % 40), int(80 + (seed // 7) % 60), int(120 + (seed // 13) % 80))
        # soft noise
        noise = rng.integers(0, 30, (h, w, 3), dtype=np.uint8)
        base = cv2.add(base, noise)

        if source_bgr is not None:
            src = cv2.resize(source_bgr, (w, h))
            base = cv2.addWeighted(src, 0.55, base, 0.45, 0)

        structure = render_structure_map(plan)
        # fill interior lightly
        gray = cv2.cvtColor(structure, cv2.COLOR_BGR2GRAY)
        # overlay locked structure strongly
        mask_lines = gray > 0
        out = base.copy()
        out[mask_lines] = structure[mask_lines]

        # draw furniture placeholders (do NOT touch wall band)
        for i in range(3):
            x = int(80 + (seed + i * 97) % max(1, w - 160))
            y = int(80 + (seed // 3 + i * 53) % max(1, h - 160))
            bw, bh = 50 + i * 10, 30 + i * 8
            roi = out[y : y + bh, x : x + bw]
            if roi.size:
                # only paint where not on structure
                patch = np.full_like(roi, (200 - i * 20, 180, 160 + i * 15))
                local_mask = gray[y : y + bh, x : x + bw] == 0
                roi[local_mask] = patch[local_mask]

        pil = _bgr_to_pil(out)
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except OSError:
            font = ImageFont.load_default()
            font_sm = font
        label = f"[MOCK {mode}] {prompts.prompt_zh[:40]}"
        draw.rectangle((8, 8, w - 8, 52), fill=(0, 0, 0, 180))
        draw.text((14, 12), label, fill=(255, 255, 255), font=font)
        draw.text(
            (14, 32),
            f"walls={len(plan.walls)} openings={len(plan.openings)} seed={seed}",
            fill=(180, 220, 255),
            font=font_sm,
        )
        return _pil_to_bgr(pil)

    def text_to_image(
        self,
        prompts: PromptBundle,
        control_canny: np.ndarray,
        plan: RoomPlan,
        *,
        strength: float,
        steps: int,
        guidance: float,
        seed: int | None,
    ) -> np.ndarray:
        s = seed if seed is not None else _seed_from(prompts.prompt + "t2i")
        return self._compose(prompts, control_canny, plan, None, s, "t2i")

    def image_to_image(
        self,
        prompts: PromptBundle,
        source_bgr: np.ndarray,
        control_canny: np.ndarray,
        plan: RoomPlan,
        *,
        strength: float,
        steps: int,
        guidance: float,
        seed: int | None,
    ) -> np.ndarray:
        s = seed if seed is not None else _seed_from(prompts.prompt + "i2i")
        return self._compose(prompts, control_canny, plan, source_bgr, s, "i2i")


class ReplicateProvider(ImageProvider):
    """Call Replicate ControlNet / img2img models via HTTP."""

    name = "replicate"

    def __init__(self, settings: Settings) -> None:
        if not settings.replicate_api_token:
            raise RuntimeError("REPLICATE_API_TOKEN is not set")
        self.settings = settings
        self.token = settings.replicate_api_token
        self.model = os.environ.get(
            "REPLICATE_MODEL_NAME", settings.replicate_model_name
        )

    def _run(self, input_payload: dict[str, Any]) -> np.ndarray:
        headers = {
            "Authorization": f"Token {self.token}",
            "Content-Type": "application/json",
            "Prefer": "wait",
        }
        # Prefer models.create-style predictions with model name
        # https://replicate.com/docs/reference/http
        body = {"input": input_payload}
        url = f"https://api.replicate.com/v1/models/{self.model}/predictions"
        with httpx.Client(timeout=180.0) as client:
            resp = client.post(url, headers=headers, json=body)
            if resp.status_code == 404:
                # fallback: versioned model from REPLICATE_CONTROLNET_MODEL
                version = (self.settings.replicate_controlnet_model or "").strip()
                if not version:
                    raise RuntimeError(
                        f"Model {self.model!r} not found; set "
                        "REPLICATE_CONTROLNET_MODEL=owner/name:version"
                    )
                if ":" in version:
                    version = version.split(":", 1)[1]
                resp = client.post(
                    "https://api.replicate.com/v1/predictions",
                    headers=headers,
                    json={"version": version, "input": input_payload},
                )
            resp.raise_for_status()
            data = resp.json()
            # poll if not finished
            pred_url = data.get("urls", {}).get("get") or (
                f"https://api.replicate.com/v1/predictions/{data['id']}"
            )
            status = data.get("status")
            while status in ("starting", "processing", "queued"):
                import time

                time.sleep(1.5)
                data = client.get(pred_url, headers=headers).json()
                status = data.get("status")
            if status != "succeeded":
                raise RuntimeError(f"Replicate failed: {data.get('error') or status}")
            output = data.get("output")
            if isinstance(output, list):
                output = output[0]
            if not isinstance(output, str):
                raise RuntimeError(f"Unexpected Replicate output: {output!r}")
            img_bytes = client.get(output).content
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return _pil_to_bgr(pil)

    def text_to_image(
        self,
        prompts: PromptBundle,
        control_canny: np.ndarray,
        plan: RoomPlan,
        *,
        strength: float,
        steps: int,
        guidance: float,
        seed: int | None,
    ) -> np.ndarray:
        # jagilley/controlnet-canny style inputs
        payload: dict[str, Any] = {
            "image": _data_uri_png(control_canny),
            "prompt": prompts.prompt,
            "negative_prompt": prompts.negative_prompt,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "controlnet_conditioning_scale": strength,
        }
        if seed is not None:
            payload["seed"] = seed
        return self._run(payload)

    def image_to_image(
        self,
        prompts: PromptBundle,
        source_bgr: np.ndarray,
        control_canny: np.ndarray,
        plan: RoomPlan,
        *,
        strength: float,
        steps: int,
        guidance: float,
        seed: int | None,
    ) -> np.ndarray:
        payload: dict[str, Any] = {
            "image": _data_uri_png(control_canny),
            "prompt": prompts.prompt,
            "negative_prompt": prompts.negative_prompt,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "controlnet_conditioning_scale": strength,
            # some models accept init image
            "init_image": _data_uri_png(source_bgr),
        }
        if seed is not None:
            payload["seed"] = seed
        return self._run(payload)


class FalProvider(ImageProvider):
    """fal.ai Flux / ControlNet via REST."""

    name = "fal"

    def __init__(self, settings: Settings) -> None:
        if not settings.fal_key:
            raise RuntimeError("FAL_KEY is not set")
        self.settings = settings
        self.key = settings.fal_key
        self.model = settings.fal_controlnet_model

    def _run(self, payload: dict[str, Any]) -> np.ndarray:
        # fal queue API
        headers = {
            "Authorization": f"Key {self.key}",
            "Content-Type": "application/json",
        }
        base = f"https://fal.run/{self.model}"
        with httpx.Client(timeout=180.0) as client:
            resp = client.post(base, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            # common shapes: {images:[{url}]}, {image:{url}}
            url = None
            if "images" in data and data["images"]:
                url = data["images"][0].get("url")
            elif "image" in data:
                img = data["image"]
                url = img.get("url") if isinstance(img, dict) else img
            if not url:
                raise RuntimeError(f"fal unexpected response keys: {list(data.keys())}")
            img_bytes = client.get(url).content
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return _pil_to_bgr(pil)

    def text_to_image(
        self,
        prompts: PromptBundle,
        control_canny: np.ndarray,
        plan: RoomPlan,
        *,
        strength: float,
        steps: int,
        guidance: float,
        seed: int | None,
    ) -> np.ndarray:
        payload: dict[str, Any] = {
            "prompt": prompts.prompt,
            "negative_prompt": prompts.negative_prompt,
            "image_url": _data_uri_png(control_canny),
            "strength": strength,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
        }
        if seed is not None:
            payload["seed"] = seed
        return self._run(payload)

    def image_to_image(
        self,
        prompts: PromptBundle,
        source_bgr: np.ndarray,
        control_canny: np.ndarray,
        plan: RoomPlan,
        *,
        strength: float,
        steps: int,
        guidance: float,
        seed: int | None,
    ) -> np.ndarray:
        payload: dict[str, Any] = {
            "prompt": prompts.prompt,
            "negative_prompt": prompts.negative_prompt,
            "image_url": _data_uri_png(source_bgr),
            "control_image_url": _data_uri_png(control_canny),
            "strength": min(0.95, strength),
            "num_inference_steps": steps,
            "guidance_scale": guidance,
        }
        if seed is not None:
            payload["seed"] = seed
        return self._run(payload)


class LocalDiffusersProvider(ImageProvider):
    """Optional local ControlNet stub — requires [local] extras."""

    name = "local"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        try:
            import torch  # noqa: F401
            from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Local provider needs: pip install 'cad-ai-lock[local]'. "
                "Or set CAD_AI_PROVIDER=mock|replicate|fal."
            ) from exc
        self._torch = __import__("torch")
        # Lazy: do not download weights in constructor for smoke tests
        self._ControlNetModel = ControlNetModel
        self._Pipeline = StableDiffusionControlNetPipeline
        self._pipe = None

    def _get_pipe(self):  # type: ignore[no-untyped-def]
        if self._pipe is None:
            controlnet = self._ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny",
                torch_dtype=self._torch.float16
                if self._torch.cuda.is_available()
                else self._torch.float32,
            )
            pipe = self._Pipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=self._torch.float16
                if self._torch.cuda.is_available()
                else self._torch.float32,
            )
            if self._torch.cuda.is_available():
                pipe = pipe.to("cuda")
            self._pipe = pipe
        return self._pipe

    def text_to_image(
        self,
        prompts: PromptBundle,
        control_canny: np.ndarray,
        plan: RoomPlan,
        *,
        strength: float,
        steps: int,
        guidance: float,
        seed: int | None,
    ) -> np.ndarray:
        pipe = self._get_pipe()
        control = _bgr_to_pil(
            cv2.cvtColor(control_canny, cv2.COLOR_GRAY2BGR)
            if control_canny.ndim == 2
            else control_canny
        )
        g = None
        if seed is not None:
            device = "cuda" if self._torch.cuda.is_available() else "cpu"
            g = self._torch.Generator(device=device).manual_seed(seed)
        result = pipe(
            prompts.prompt,
            image=control,
            negative_prompt=prompts.negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            controlnet_conditioning_scale=strength,
            generator=g,
        )
        return _pil_to_bgr(result.images[0])

    def image_to_image(
        self,
        prompts: PromptBundle,
        source_bgr: np.ndarray,
        control_canny: np.ndarray,
        plan: RoomPlan,
        *,
        strength: float,
        steps: int,
        guidance: float,
        seed: int | None,
    ) -> np.ndarray:
        # Same ControlNet path; source informs prompt only in stub
        _ = source_bgr
        return self.text_to_image(
            prompts,
            control_canny,
            plan,
            strength=strength,
            steps=steps,
            guidance=guidance,
            seed=seed,
        )


def get_provider(settings: Settings | None = None) -> ImageProvider:
    settings = settings or get_settings()
    name = settings.resolve_provider()
    if name == ProviderName.MOCK:
        return MockProvider()
    if name == ProviderName.REPLICATE:
        return ReplicateProvider(settings)
    if name == ProviderName.FAL:
        return FalProvider(settings)
    if name == ProviderName.LOCAL:
        return LocalDiffusersProvider(settings)
    raise ValueError(f"Unknown provider: {name}")


def run_t2i(
    plan: RoomPlan,
    user_prompt: str,
    out_path: str | Path,
    *,
    provider: ImageProvider | None = None,
    control_canny: np.ndarray | None = None,
    strength: float | None = None,
    steps: int | None = None,
    guidance: float | None = None,
    seed: int | None = None,
    layout: Any | None = None,
) -> Path:
    from cad_ai_lock.geometry import FurnitureLayout

    settings = get_settings()
    provider = provider or get_provider(settings)
    layout_obj: FurnitureLayout | None = layout  # type: ignore[assignment]
    prompts = build_prompts(user_prompt, plan, mode="t2i", layout=layout_obj)
    canny = control_canny if control_canny is not None else render_canny_from_plan(plan)
    img = provider.text_to_image(
        prompts,
        canny,
        plan,
        strength=strength if strength is not None else settings.control_strength,
        steps=steps if steps is not None else settings.num_inference_steps,
        guidance=guidance if guidance is not None else settings.guidance_scale,
        seed=seed,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    scrub = f"\nSCRUBBED: {prompts.scrubbed}\n" if prompts.scrubbed else ""
    (out_path.with_suffix(".prompt.txt")).write_text(
        f"PROMPT:\n{prompts.prompt}\n\nNEGATIVE:\n{prompts.negative_prompt}\n"
        f"\nZH:\n{prompts.prompt_zh}\n{scrub}",
        encoding="utf-8",
    )
    return out_path


def run_i2i(
    plan: RoomPlan,
    source_path: str | Path,
    user_prompt: str,
    out_path: str | Path,
    *,
    provider: ImageProvider | None = None,
    control_canny: np.ndarray | None = None,
    strength: float | None = None,
    steps: int | None = None,
    guidance: float | None = None,
    seed: int | None = None,
    layout: Any | None = None,
) -> Path:
    from cad_ai_lock.geometry import FurnitureLayout

    settings = get_settings()
    provider = provider or get_provider(settings)
    layout_obj: FurnitureLayout | None = layout  # type: ignore[assignment]
    prompts = build_prompts(user_prompt, plan, mode="i2i", layout=layout_obj)
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if source is None:
        raise FileNotFoundError(source_path)
    canny = control_canny if control_canny is not None else render_canny_from_plan(plan)
    img = provider.image_to_image(
        prompts,
        source,
        canny,
        plan,
        strength=strength if strength is not None else settings.control_strength,
        steps=steps if steps is not None else settings.num_inference_steps,
        guidance=guidance if guidance is not None else settings.guidance_scale,
        seed=seed,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    scrub = f"\nSCRUBBED: {prompts.scrubbed}\n" if prompts.scrubbed else ""
    (out_path.with_suffix(".prompt.txt")).write_text(
        f"PROMPT:\n{prompts.prompt}\n\nNEGATIVE:\n{prompts.negative_prompt}\n"
        f"\nZH:\n{prompts.prompt_zh}\n{scrub}",
        encoding="utf-8",
    )
    return out_path
