"""Request / event JSON schemas for the MangaTranslator Modal API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from deploy.modal_config import MAX_IMAGES_PER_JOB, PROVIDER_ALIASES

OutputType = Literal["supabase", "volume", "none"]
JobStatus = Literal["queued", "running", "completed", "failed"]


class DetectionConfigIn(BaseModel):
    bubble_detector_model: str = "yolo_2"


class OutsideTextConfigIn(BaseModel):
    enabled: bool = True
    inpainting_method: str = "lama_large"

    @field_validator("inpainting_method")
    @classmethod
    def normalize_inpaint(cls, value: str) -> str:
        method = (value or "lama_large").strip().lower()
        allowed = {"lama_large", "opencv", "none"}
        if method not in allowed:
            raise ValueError("inpainting_method must be lama_large, opencv, or none")
        return method


class RenderingConfigIn(BaseModel):
    rtl: bool = True


class JobConfigIn(BaseModel):
    input_language: str = "Japanese"
    output_language: str = "English"
    provider: str = "deepseek"
    model_name: str = "deepseek-v4-flash"
    translation_mode: str = "one-step"
    ocr_method: str = "LLM"
    font_name: str = "Anime Ace 3.0"
    detection: DetectionConfigIn = Field(default_factory=DetectionConfigIn)
    outside_text: OutsideTextConfigIn = Field(default_factory=OutsideTextConfigIn)
    rendering: RenderingConfigIn = Field(default_factory=RenderingConfigIn)
    output_format: str = "webp"

    @field_validator("provider")
    @classmethod
    def normalize_provider(cls, value: str) -> str:
        raw = (value or "").strip()
        if not raw:
            raise ValueError("provider is required")
        return PROVIDER_ALIASES.get(raw.lower(), raw)

    @field_validator("translation_mode")
    @classmethod
    def normalize_mode(cls, value: str) -> str:
        mode = (value or "").strip().lower()
        if mode not in {"one-step", "two-step"}:
            raise ValueError("translation_mode must be one-step or two-step")
        return mode

    @field_validator("ocr_method")
    @classmethod
    def normalize_ocr(cls, value: str) -> str:
        method = (value or "").strip()
        aliases = {"llm": "LLM", "manga-ocr": "manga-ocr", "manga_ocr": "manga-ocr"}
        mapped = aliases.get(method.lower())
        if mapped is None:
            raise ValueError("ocr_method must be LLM or manga-ocr")
        return mapped

    @field_validator("output_format")
    @classmethod
    def normalize_format(cls, value: str) -> str:
        fmt = (value or "webp").strip().lower().lstrip(".")
        if fmt not in {"webp", "png", "jpg", "jpeg"}:
            raise ValueError("output_format must be webp, png, or jpeg")
        return "jpg" if fmt == "jpeg" else fmt

    @model_validator(mode="after")
    def validate_provider_combo(self) -> JobConfigIn:
        if self.provider == "DeepL":
            if self.translation_mode != "two-step":
                raise ValueError("DeepL has no vision; translation_mode must be two-step")
            if self.ocr_method == "LLM":
                raise ValueError("DeepL has no vision; ocr_method must be manga-ocr")
        return self


class ImageItemIn(BaseModel):
    image_id: str
    url: str | None = None
    index: int = 0

    @field_validator("image_id")
    @classmethod
    def image_id_not_empty(cls, value: str) -> str:
        image_id = (value or "").strip()
        if not image_id:
            raise ValueError("image_id is required")
        return image_id


class OutputSpecIn(BaseModel):
    type: OutputType = "none"
    bucket: str | None = None
    paths: list[str] = Field(default_factory=list)


class CreateJobRequest(BaseModel):
    job_id: str | None = None
    images: list[ImageItemIn]
    output: OutputSpecIn = Field(default_factory=OutputSpecIn)
    config: JobConfigIn = Field(default_factory=JobConfigIn)

    @model_validator(mode="after")
    def validate_images(self) -> CreateJobRequest:
        if not self.images:
            raise ValueError("images must not be empty")
        if len(self.images) > MAX_IMAGES_PER_JOB:
            raise ValueError(f"at most {MAX_IMAGES_PER_JOB} images per job")

        ids = [item.image_id for item in self.images]
        if len(set(ids)) != len(ids):
            raise ValueError("image_id must be unique within a job")

        indexes = [item.index for item in self.images]
        if len(set(indexes)) != len(indexes):
            raise ValueError("image index must be unique within a job")

        if self.output.type == "supabase":
            if not self.output.bucket:
                raise ValueError("output.bucket is required when output.type is supabase")
            if self.output.paths and len(self.output.paths) != len(self.images):
                raise ValueError("output.paths length must match images")
        return self


class CreateJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    image_count: int


class JobEvent(BaseModel):
    seq: int
    event: str
    job_id: str
    data: dict[str, Any] = Field(default_factory=dict)
