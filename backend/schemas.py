import base64
from typing import Literal

from pydantic import BaseModel, Field, model_validator, field_validator


class EvidenceFrame(BaseModel):
    timestamp_sec: float = Field(ge=0, allow_inf_nan=False)
    image_base64: str

    @field_validator("image_base64")
    @classmethod
    def validate_jpeg(cls, value: str) -> str:
        try:
            image = base64.b64decode(value, validate=True)
        except ValueError as exc:
            raise ValueError("Expected base64 JPEG") from exc
        if not (image.startswith(b"\xff\xd8") and image.endswith(b"\xff\xd9")):
            raise ValueError("Expected JPEG image")
        return value


class RuleSource(BaseModel):
    document: str = Field(min_length=1)
    section: str | None = None
    url: str | None = None


class Violation(BaseModel):
    type: str = Field(min_length=1)
    timecode_sec: float = Field(ge=0, allow_inf_nan=False)
    description: str = Field(min_length=1)
    evidence_frames: list[EvidenceFrame] = Field(min_length=1, max_length=3)
    # Additional UI field; the jury's required fields stay unchanged.
    rule_source: RuleSource | None = None


class AnalysisResult(BaseModel):
    verdict: Literal["no_violation", "violation"]
    violations: list[Violation]

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
