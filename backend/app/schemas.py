from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


CaseStatus = Literal["active", "archived"]


class CaseCreate(BaseModel):
    name: str = Field(min_length=1)
    description: Optional[str] = None


class CaseUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[CaseStatus] = None


class CaseResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    status: CaseStatus
    case_slug: str
    created_at: Optional[str] = None
    updated_at: str
    doc_count: Optional[int] = None


class Envelope(BaseModel):
    status: Literal["success", "error"]
    message: str
    data: object | None
    metadata: Optional[dict] = None


ChatMode = Literal["mix", "local", "global", "hybrid", "naive", "bypass"]
AnalysisType = Literal["link", "flow", "event"]


class ChatCreate(BaseModel):
    title: Optional[str] = Field(default=None, max_length=200)


class ChatMessageCreate(BaseModel):
    content: str = Field(min_length=1)
    mode: ChatMode
    options: Optional[dict[str, Any]] = None


class AnalysisCreate(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    analysis_type: AnalysisType


class AnalysisRepairRequest(BaseModel):
    chart_id: str = Field(min_length=1, max_length=100)
    error: str = Field(min_length=1, max_length=8000)
    mermaid_code: Optional[str] = Field(default=None, max_length=30000)


class EntityResolutionMergeRequest(BaseModel):
    source_entities: list[str] = Field(min_length=1)
    target_entity: str = Field(min_length=1)
    reason: Optional[str] = None


class DocumentDuplicateCheckItem(BaseModel):
    client_id: str = Field(min_length=1, max_length=200)
    original_filename: Optional[str] = Field(default=None, max_length=500)
    size_bytes: Optional[int] = Field(default=None, ge=0)
    content_hash_sha256: str = Field(min_length=64, max_length=64)


class DocumentDuplicateCheckRequest(BaseModel):
    files: list[DocumentDuplicateCheckItem] = Field(min_length=1, max_length=200)
