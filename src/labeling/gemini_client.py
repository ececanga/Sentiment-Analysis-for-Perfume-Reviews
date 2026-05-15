# Date Created: 2026-04-13 23:02:44+03:00
# Date Last Modified: 2026-05-15

"""Gemini client wrapper for strict per-snippet labeling."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

from google import genai

from src.labeling.strict_schema import STRICT_PROMPT_RULES, validate_label_payload


@dataclass(frozen=True)
class GeminiLabelConfig:
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.0
    max_retries: int = 5
    backoff_base_sec: float = 1.0
    backoff_cap_sec: float = 32.0
    rate_limit_sleep_sec: float = 0.5


@dataclass
class SnippetLabelResult:
    label: Dict[str, float | int]
    attempts: int
    raw_response: str


class GeminiLabelingError(RuntimeError):
    def __init__(
        self,
        message: str,
        attempts: int,
        last_error: str,
        last_response: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error
        self.last_response = last_response


class GeminiStrictLabelClient:
    def __init__(self, config: GeminiLabelConfig, api_key: Optional[str] = None) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("Set GEMINI_API_KEY in environment.")
        self.config = config
        self.client = genai.Client(api_key=key)

    def _build_prompt(self, perfume_name: str, snippet_text: str) -> str:
        return (
            f"{STRICT_PROMPT_RULES}\n\n"
            "Return ONLY valid JSON object with exactly these keys:\n"
            '{"joy":0,"sadness":0,"anger":0,"fear":0,"trust":0,"surprise":0,"anticipation":0,"disgust":0,'
            '"valence":5.0,"arousal":5.0,"dominance":5.0}\n\n'
            f"Perfume: {perfume_name}\n"
            "Snippet:\n"
            f"{snippet_text}"
        )

    def label_snippet(self, perfume_name: str, snippet_text: str) -> SnippetLabelResult:
        last_error = ""
        last_response = ""

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.config.model_name,
                    contents=[
                        {
                            "role": "user",
                            "parts": [{"text": self._build_prompt(perfume_name, snippet_text)}],
                        }
                    ],
                    config={
                        "temperature": self.config.temperature,
                        "response_mime_type": "application/json",
                    },
                )
                raw_text = (response.text or "").strip()
                payload = json.loads(raw_text)
                normalized = validate_label_payload(payload)
                if self.config.rate_limit_sleep_sec > 0:
                    time.sleep(self.config.rate_limit_sleep_sec)
                return SnippetLabelResult(
                    label=normalized,
                    attempts=attempt,
                    raw_response=raw_text,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                last_response = locals().get("raw_text", "")
                if attempt >= self.config.max_retries:
                    break
                sleep_s = min(
                    self.config.backoff_cap_sec,
                    self.config.backoff_base_sec * (2 ** (attempt - 1)),
                )
                time.sleep(sleep_s)

        raise GeminiLabelingError(
            message=f"Gemini labeling failed after {self.config.max_retries} attempts",
            attempts=self.config.max_retries,
            last_error=last_error or "unknown_error",
            last_response=last_response or None,
        )

