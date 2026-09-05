from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, create_engine, delete, inspect, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

load_dotenv()


class Base(DeclarativeBase):
    pass


class Meeting(Base):
    __tablename__ = "meetings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str] = mapped_column(String(250), nullable=False)
    source_type: Mapped[str] = mapped_column(String(30), nullable=False, default="paste")
    source_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    transcript: Mapped[str] = mapped_column(Text, nullable=False)
    analysis_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class TranscriptChunk(Base):
    __tablename__ = "transcript_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    meeting_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("meetings.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_json: Mapped[str] = mapped_column(Text, nullable=False)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    meeting_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("meetings.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


def _database_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        local_db = Path(__file__).resolve().parents[1] / "meeting_notes.db"
        return f"sqlite:///{local_db}"
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    return url


ENGINE = create_engine(_database_url(), pool_pre_ping=True)


def init_db() -> None:
    Base.metadata.create_all(ENGINE)

    # Small backwards-compatible migration for databases created by the first
    # assignment version. New databases already contain this column.
    columns = {column["name"] for column in inspect(ENGINE).get_columns("meetings")}
    if "source_name" not in columns:
        with ENGINE.begin() as connection:
            connection.execute(text("ALTER TABLE meetings ADD COLUMN source_name VARCHAR(255)"))


def save_meeting(
    title: str,
    source_type: str,
    source_name: str | None,
    transcript: str,
    analysis: dict[str, Any],
    chunks: list[str],
    embeddings: list[list[float]],
) -> str:
    meeting_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    with Session(ENGINE) as session:
        session.add(
            Meeting(
                id=meeting_id,
                title=title,
                source_type=source_type,
                source_name=source_name,
                transcript=transcript,
                analysis_json=json.dumps(analysis, ensure_ascii=False),
                created_at=now,
            )
        )
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            session.add(
                TranscriptChunk(
                    meeting_id=meeting_id,
                    chunk_index=idx,
                    text=chunk,
                    embedding_json=json.dumps(embedding),
                )
            )
        session.commit()
    return meeting_id


def list_meetings() -> list[dict[str, Any]]:
    with Session(ENGINE) as session:
        rows = session.scalars(select(Meeting).order_by(Meeting.created_at.desc())).all()
        return [
            {
                "id": row.id,
                "title": row.title,
                "created_at": row.created_at,
                "source_type": row.source_type,
                "source_name": row.source_name,
            }
            for row in rows
        ]


def get_meeting(meeting_id: str) -> dict[str, Any] | None:
    with Session(ENGINE) as session:
        row = session.get(Meeting, meeting_id)
        if not row:
            return None
        return {
            "id": row.id,
            "title": row.title,
            "source_type": row.source_type,
            "source_name": row.source_name,
            "transcript": row.transcript,
            "analysis": json.loads(row.analysis_json),
            "created_at": row.created_at,
        }


def get_chunks(meeting_id: str) -> list[dict[str, Any]]:
    with Session(ENGINE) as session:
        rows = session.scalars(
            select(TranscriptChunk)
            .where(TranscriptChunk.meeting_id == meeting_id)
            .order_by(TranscriptChunk.chunk_index)
        ).all()
        return [
            {
                "chunk_index": row.chunk_index,
                "text": row.text,
                "embedding": json.loads(row.embedding_json),
            }
            for row in rows
        ]


def add_chat_message(meeting_id: str, role: str, content: str) -> None:
    with Session(ENGINE) as session:
        session.add(
            ChatMessage(
                meeting_id=meeting_id,
                role=role,
                content=content,
                created_at=datetime.now(timezone.utc),
            )
        )
        session.commit()


def get_chat_messages(meeting_id: str) -> list[dict[str, str]]:
    with Session(ENGINE) as session:
        rows = session.scalars(
            select(ChatMessage)
            .where(ChatMessage.meeting_id == meeting_id)
            .order_by(ChatMessage.created_at, ChatMessage.id)
        ).all()
        return [{"role": row.role, "content": row.content} for row in rows]


def delete_meeting(meeting_id: str) -> None:
    with Session(ENGINE) as session:
        session.execute(delete(ChatMessage).where(ChatMessage.meeting_id == meeting_id))
        session.execute(delete(TranscriptChunk).where(TranscriptChunk.meeting_id == meeting_id))
        session.execute(delete(Meeting).where(Meeting.id == meeting_id))
        session.commit()
