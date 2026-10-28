from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RenderStatusEnum(str, Enum):
    NOT_STARTED = "not_started"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class RenderJobCreate(BaseModel):
    pass


class RenderJobUpdate(BaseModel):
    status: Optional[RenderStatusEnum] = None
    progress: Optional[int] = Field(None, ge=0, le=100)
    gif_path: Optional[str] = None
    error_msg: Optional[str] = None


class RenderJobBase(BaseModel):
    id: str
    status: RenderStatusEnum
    progress: int
    gif_path: Optional[str] = None
    error_msg: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class RenderJobResponse(RenderJobBase):
    pass


class RenderJobListResponse(BaseModel):
    jobs: list[RenderJobResponse]
    total: int


class RenderJobStatusResponse(BaseModel):
    id: str
    status: RenderStatusEnum
    progress: int
    gif_path: Optional[str] = None

    class Config:
        from_attributes = True
