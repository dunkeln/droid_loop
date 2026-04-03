from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass

import anthropic
from dotenv import find_dotenv, load_dotenv
from PIL import Image
from pydantic import BaseModel, Field

MODEL_ID = "claude-haiku-4-5-20251001"

load_dotenv(find_dotenv())


class VlmIncidentResult(BaseModel):
    incident_type: str = Field(min_length=1)
    failure_mode: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=1)
    evidence_frame_indices: list[int] = Field(default_factory=list)
    actionability: str = Field(min_length=1)
    recommendations: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class VlmQueryOutput:
    result: VlmIncidentResult
    raw_text: str
    attempts: int
    input_tokens: int
    output_tokens: int


def _usage_tokens(response: object) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    in_tok = int(getattr(usage, "input_tokens", 0) or 0)
    out_tok = int(getattr(usage, "output_tokens", 0) or 0)
    return in_tok, out_tok


def _encode_image(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in model response")
    return text[start : end + 1]


def _build_clip_context_block(clip_metadata: dict | None) -> str:
    """Build a concise clip-identity paragraph injected into the system prompt."""
    if not clip_metadata:
        return "You are analyzing a robot manipulation clip."

    lines: list[str] = []

    ordinal = clip_metadata.get("ordinal")
    total = clip_metadata.get("total_clips")
    clip_id = clip_metadata.get("id", "")

    if ordinal is not None and total is not None:
        lines.append(
            f"You are analyzing clip {ordinal} of {total} flagged incident clips "
            f"in this episode (id: {clip_id})."
        )
    else:
        lines.append("You are analyzing a flagged incident clip from a robot manipulation episode.")

    start_f = clip_metadata.get("start_frame_index")
    end_f = clip_metadata.get("end_frame_index")
    anchor_f = clip_metadata.get("anchor_frame_index")
    if start_f is not None and end_f is not None:
        anchor_note = f" — anchor (peak anomaly) at frame {anchor_f}" if anchor_f is not None else ""
        lines.append(f"Frame range: {start_f}–{end_f}{anchor_note}.")

    start_ts = clip_metadata.get("start_timestamp_s")
    end_ts = clip_metadata.get("end_timestamp_s")
    anchor_ts = clip_metadata.get("anchor_timestamp_s")
    if start_ts is not None and end_ts is not None:
        anchor_ts_note = f" (peak at {anchor_ts:.2f}s)" if anchor_ts is not None else ""
        lines.append(f"Time window: {start_ts:.2f}s – {end_ts:.2f}s{anchor_ts_note}.")

    # Sibling clips give the model inter-clip comparison context
    siblings: list[dict] = clip_metadata.get("sibling_clips", [])
    if siblings:
        sibling_summary = ", ".join(
            f"clip {s.get('ordinal')} frames {s.get('start_frame_index')}–{s.get('end_frame_index')}"
            for s in siblings[:4]
        )
        lines.append(f"Other clips in this episode: [{sibling_summary}].")

    return " ".join(lines)


def _prompt_schema(user_query: str | None, clip_metadata: dict | None = None) -> str:
    query_text = (
        user_query.strip()
        if user_query
        else "Describe the most likely incident/failure mode visible in the frames."
    )
    context_block = _build_clip_context_block(clip_metadata)
    return (
        f"{context_block} "
        "Return ONLY one JSON object with keys exactly:\n"
        "{"
        '"incident_type": string,'
        '"failure_mode": string,'
        '"confidence": number_between_0_and_1,'
        '"summary": string,'
        '"evidence_frame_indices": array_of_integers,'
        '"actionability": string,'
        '"recommendations": array_of_strings'
        "}\n"
        "No markdown. No prose outside JSON. "
        f"Task: {query_text}"
    )


def _tool_spec() -> dict:
    return {
        "name": "respond_incident_json",
        "description": (
            "Return the final incident analysis as structured JSON after analyzing the provided clip frames."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "incident_type": {"type": "string"},
                "failure_mode": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "summary": {"type": "string"},
                "evidence_frame_indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "default": [],
                },
                "actionability": {"type": "string"},
                "recommendations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                },
            },
            "required": [
                "incident_type",
                "failure_mode",
                "confidence",
                "summary",
                "evidence_frame_indices",
                "actionability",
                "recommendations",
            ],
            "additionalProperties": False,
        },
    }


def _extract_tool_payload(response: object, tool_name: str) -> dict | None:
    blocks = getattr(response, "content", []) or []
    for block in blocks:
        btype = getattr(block, "type", None)
        bname = getattr(block, "name", None)
        if btype == "tool_use" and bname == tool_name:
            data = getattr(block, "input", None)
            if isinstance(data, dict):
                return data
    return None


def query_incident(
    images: list[Image.Image],
    frame_indices: list[int],
    timestamps_s: list[float],
    chat_history: list[dict[str, str]] | None = None,
    user_query: str | None = None,
    clip_metadata: dict | None = None,
    model_id: str = MODEL_ID,
    max_retries: int = 2,
    temperature: float = 0.0,
    use_tools: bool = True,
) -> VlmQueryOutput:
    if not images:
        raise ValueError("images cannot be empty")
    if not (len(images) == len(frame_indices) == len(timestamps_s)):
        raise ValueError("images, frame_indices, and timestamps_s must have equal length")

    client = anthropic.Anthropic()
    content: list[dict] = []
    for image in images:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": _encode_image(image.convert("RGB")),
                },
            }
        )
    frame_table = ", ".join(
        f'{{"frame_index": {fi}, "timestamp_s": {round(ts, 3)}}}'
        for fi, ts in zip(frame_indices, timestamps_s)
    )
    history = (chat_history or [])[-10:]
    history_text = "\n".join(
        f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history if m.get("content")
    )
    content.append(
        {
            "type": "text",
            "text": (
                _prompt_schema(user_query, clip_metadata=clip_metadata)
                + "\nFrame metadata in order: ["
                + frame_table
                + "]"
                + ("\nPrior chat context:\n" + history_text if history_text else "")
            ),
        }
    )

    attempts = 0
    last_error = "unknown error"
    last_raw = ""
    total_in_tok = 0
    total_out_tok = 0
    for _ in range(max(1, max_retries) + 1):
        attempts += 1
        req_kwargs: dict = {
            "model": model_id,
            "max_tokens": 450,
            "temperature": temperature,
            "messages": [{"role": "user", "content": content}],
        }
        if use_tools:
            req_kwargs["tools"] = [_tool_spec()]
            req_kwargs["tool_choice"] = {"type": "tool", "name": "respond_incident_json"}
        response = client.messages.create(**req_kwargs)
        in_tok, out_tok = _usage_tokens(response)
        total_in_tok += in_tok
        total_out_tok += out_tok
        if use_tools:
            tool_payload = _extract_tool_payload(response, "respond_incident_json")
            if isinstance(tool_payload, dict):
                try:
                    result = VlmIncidentResult.model_validate(tool_payload)
                    return VlmQueryOutput(
                        result=result,
                        raw_text=json.dumps(tool_payload),
                        attempts=attempts,
                        input_tokens=total_in_tok,
                        output_tokens=total_out_tok,
                    )
                except Exception as exc:
                    last_error = str(exc)
                    continue
        raw_text = "\n".join(
            block.text for block in response.content if hasattr(block, "text")
        ).strip()
        last_raw = raw_text
        try:
            obj = json.loads(_extract_json_object(raw_text))
            result = VlmIncidentResult.model_validate(obj)
            return VlmQueryOutput(
                result=result,
                raw_text=raw_text,
                attempts=attempts,
                input_tokens=total_in_tok,
                output_tokens=total_out_tok,
            )
        except Exception as exc:
            last_error = str(exc)
            continue
    raise ValueError(f"failed to parse valid VLM JSON after {attempts} attempts: {last_error}; raw={last_raw[:300]!r}")
