from __future__ import annotations

import json
import math
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 5,
        },
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "task": {"type": "string"},
                    "owner": {"type": ["string", "null"]},
                    "due_date": {"type": ["string", "null"]},
                },
                "required": ["task", "owner", "due_date"],
            },
        },
        "decisions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "sentiment": {"type": "string"},
    },
    "required": ["summary", "action_items", "decisions", "sentiment"],
}


class MeetingAI:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")
        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
        self.embedding_model = os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        )

    def analyze(self, transcript: str) -> dict[str, Any]:
        instructions = """
You are a careful meeting-notes analyst.
Use only facts explicitly supported by the meeting transcript.
Return 3-5 concise summary bullets.
For action items, extract the task, owner, and due date. If owner or due date is not
stated, return null instead of guessing.
For decisions, include only decisions that were actually made, not suggestions.
Sentiment should be one short line describing the overall meeting tone.
Do not invent names, dates, commitments, or decisions.
""".strip()

        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=f"Analyze this meeting transcript:\n\n{transcript}",
            text={
                "format": {
                    "type": "json_schema",
                    "name": "meeting_analysis",
                    "schema": ANALYSIS_SCHEMA,
                    "strict": True,
                }
            },
            store=False,
        )
        return json.loads(response.output_text)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        result = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        ordered = sorted(result.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]

    def answer_question(
        self,
        meeting_title: str,
        question: str,
        retrieved_chunks: list[tuple[int, str, float]],
        chat_history: list[dict[str, str]],
    ) -> str:
        context = "\n\n".join(
            f"[Relevant transcript excerpt]\n{text}"
            for _idx, text, _score in retrieved_chunks
        )
        history = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in chat_history[-6:]
        )

        instructions = """
You are a meeting transcript assistant.
Answer only from the supplied transcript excerpts.
If the answer is not supported by the excerpts, say:
"I couldn't find that in this meeting transcript."
Do not use outside knowledge to fill gaps. Be concise and factual.
Do not mention chunks, embeddings, retrieval scores, or internal implementation details.
The application displays the source filename separately, so do not add citations yourself.
""".strip()

        prompt = f"""
Meeting: {meeting_title}

Relevant transcript excerpts:
{context}

Recent chat history:
{history or '(none)'}

User question:
{question}
""".strip()

        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=prompt,
            store=False,
        )
        return response.output_text.strip()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
