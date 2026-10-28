import uuid
import enum

from sqlalchemy import Column, Integer, String, DateTime, Enum
from sqlalchemy.sql import func

from database import Base


class RenderStatus(enum.Enum):
    NOT_STARTED = "not_started"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class RenderJob(Base):
    __tablename__ = "render_jobs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    status = Column(Enum(RenderStatus), nullable=False, default=RenderStatus.NOT_STARTED)
    progress = Column(Integer, nullable=False, default=0)

    gif_path = Column(String, nullable=True)
    error_msg = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
