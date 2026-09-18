"""Low-confidence detection and review reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class RetryRange:
    indices: tuple[int, ...]
    start: float
    end: float


def _value(segment: Any, name: str, default: float = 0.0) -> float:
    if isinstance(segment, dict):
        value = segment.get(name, default)
    else:
        value = getattr(segment, name, default)
    return default if value is None else float(value)


def quality_reasons(segment: Any) -> list[str]:
    reasons: list[str] = []
    if _value(segment, "avg_logprob") < -1.0:
        reasons.append("low_log_probability")
    if _value(segment, "compression_ratio") > 2.4:
        reasons.append("high_compression_ratio")
    if _value(segment, "no_speech_prob") > 0.6:
        reasons.append("high_no_speech_probability")
    if _value(segment, "temperature") >= 0.8:
        reasons.append("high_temperature")
    return reasons


def retry_ranges(
    segments: list[Any], max_ranges: int, max_duration: float = 30.0
) -> list[RetryRange]:
    suspicious = [index for index, segment in enumerate(segments) if quality_reasons(segment)]
    if not suspicious or max_ranges == 0:
        return []

    groups: list[list[int]] = []
    current: list[int] = []
    for index in suspicious:
        segment = segments[index]
        start = _value(segment, "start")
        if current:
            first_start = _value(segments[current[0]], "start")
            previous_end = _value(segments[current[-1]], "end")
            if start - previous_end > 1.0 or _value(segment, "end") - first_start > max_duration:
                groups.append(current)
                current = []
        current.append(index)
    if current:
        groups.append(current)

    return [
        RetryRange(
            indices=tuple(group),
            start=_value(segments[group[0]], "start"),
            end=_value(segments[group[-1]], "end"),
        )
        for group in groups[:max_ranges]
    ]


def average_log_probability(segments: Iterable[Any]) -> float:
    values = [_value(segment, "avg_logprob", -10.0) for segment in segments]
    return sum(values) / len(values) if values else -10.0


def annotate_quality(segments: list[dict[str, Any]]) -> dict[str, Any]:
    low_confidence = 0
    for segment in segments:
        reasons = quality_reasons(segment)
        segment["quality"] = {
            "status": "low_confidence" if reasons else "ok",
            "reasons": reasons,
        }
        low_confidence += bool(reasons)
    return {
        "total_segments": len(segments),
        "low_confidence_segments": low_confidence,
    }


def write_quality_report(
    output_dir: Path,
    stem: str,
    segments: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    review_segments = [
        {
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"].strip(),
            "reasons": segment["quality"]["reasons"],
        }
        for segment in segments
        if segment["quality"]["status"] == "low_confidence"
    ]
    report = {"summary": summary, "segments": review_segments}
    (output_dir / f"{stem}.review.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        f"{item['start']:.2f}-{item['end']:.2f}s "
        f"[{', '.join(item['reasons'])}] {item['text']}"
        for item in review_segments
    ]
    (output_dir / f"{stem}.review.txt").write_text(
        ("\n".join(lines) + "\n") if lines else "Nenhum segmento de baixa confiança.\n",
        encoding="utf-8",
    )
