import base64
import io
import warnings
from typing import Literal

from pydantic import BaseModel, Field, model_validator, field_validator
from PIL import Image, UnidentifiedImageError

MAX_JPEG_BYTES = 2 * 1024 * 1024
MAX_RESULT_BYTES = 24 * 1024 * 1024


class EvidenceFrame(BaseModel):
    timestamp_sec: float = Field(ge=0, allow_inf_nan=False)
    image_base64: str = Field(max_length=4 * ((MAX_JPEG_BYTES + 2) // 3))

    @field_validator("image_base64")
    @classmethod
    def validate_jpeg(cls, value: str) -> str:
        try:
            image = base64.b64decode(value, validate=True)
        except ValueError as exc:
            raise ValueError("Expected base64 JPEG") from exc
        if len(image) > MAX_JPEG_BYTES:
            raise ValueError("JPEG exceeds 2 MiB")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(image)) as decoded:
                    if decoded.format != "JPEG" or decoded.width * decoded.height > 16_777_216:
                        raise ValueError("Expected JPEG up to 16 megapixels")
                    decoded.verify()
                # verify() alone does not decode JPEG pixel data.
                with Image.open(io.BytesIO(image)) as decoded:
                    decoded.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise ValueError("Invalid or incomplete JPEG") from exc
        return value


class RuleSource(BaseModel):
    status: Literal["provided"] = "provided"
    document: str = Field(min_length=1, max_length=2000)
    section: str | None = Field(default=None, max_length=1000)
    url: str | None = Field(default=None, max_length=2000)

    @field_validator("document")
    @classmethod
    def document_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Rule document is required")
        return value.strip()


class UnmappedRuleSource(BaseModel):
    """An explicit limitation, never a fabricated document or legal conclusion."""
    status: Literal["unmapped"]
    reason: str = Field(min_length=1, max_length=2000)
    document: None = None

    @field_validator("reason")
    @classmethod
    def reason_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("An explanation of the missing rule mapping is required")
        return value.strip()


class Violation(BaseModel):
    type: str = Field(min_length=1, max_length=200)
    timecode_sec: float = Field(ge=0, allow_inf_nan=False)
    description: str = Field(min_length=1, max_length=5000)
    evidence_frames: list[EvidenceFrame] = Field(min_length=1, max_length=3)
    # Required by the user scenario; an additive field in the jury JSON.
    rule_source: RuleSource | UnmappedRuleSource


class AnalysisResult(BaseModel):
    verdict: Literal["no_violation", "violation"]
    violations: list[Violation] = Field(max_length=100)

    @model_validator(mode="after")
    def validate_verdict(self):
        if (self.verdict == "violation") != bool(self.violations):
            raise ValueError("Verdict must agree with violations")
        return self


class JobCreated(BaseModel):
    job_id: str


class JobResponse(BaseModel):
    job_id: str
    status: Literal["processing", "done", "failed"]
    verdict: Literal["no_violation", "violation"] | None = None
    violations: list[Violation] | None = None
    error: str | None = None
