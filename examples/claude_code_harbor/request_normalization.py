# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Mirror the source proxy's system-text normalization before model routing."""

from __future__ import annotations

import re
from typing import Any
from uuid import NAMESPACE_URL, uuid5


_CCH_RE = re.compile(r"cch=[0-9a-fA-F]+")
_CC_VERSION_SUFFIX_RE = re.compile(r"(cc_version=\d+\.\d+\.\d+)\.[0-9a-fA-F]+")
_TOTAL_TOKENS_RUN_RE = re.compile(r"(<total_tokens>[^<]*</total_tokens>)(\s*<total_tokens>[^<]*</total_tokens>)+")


def normalize_request(payload: dict[str, Any], rollout_id: str) -> dict[str, Any]:
    """Stabilize transport fingerprints without hiding real context changes.

    The returned payload must be used for both the model request and capture.
    Only system text is rewritten; tool observations and assistant history keep
    their original values. Semantic CC versions and other system changes remain
    visible to prefix matching, including compaction and subagent instructions.
    """
    cch = uuid5(NAMESPACE_URL, rollout_id).hex[:5]
    version_suffix = uuid5(NAMESPACE_URL, f"cc-version:{rollout_id}").hex[:3]

    def text(value: str) -> str:
        value = _CC_VERSION_SUFFIX_RE.sub(rf"\1.{version_suffix}", value)
        value = _CCH_RE.sub(f"cch={cch}", value)
        return _TOTAL_TOKENS_RUN_RE.sub(r"\1", value)

    def content(value: Any) -> Any:
        if isinstance(value, str):
            return text(value)
        if isinstance(value, list):
            blocks = [
                {**block, "text": text(block["text"])}
                if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
                else block
                for block in value
            ]
            if blocks and all(
                isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
                for block in blocks
            ):
                # Both source and Relax join system text blocks with two
                # newlines. Collapse a run spanning block boundaries on that
                # exact model-visible text; keep block metadata otherwise.
                joined = "\n\n".join(block["text"] for block in blocks)
                collapsed = _TOTAL_TOKENS_RUN_RE.sub(r"\1", joined)
                if collapsed != joined:
                    return collapsed
            return blocks
        return value

    normalized = dict(payload)
    if "system" in payload:
        normalized["system"] = content(payload["system"])
    if isinstance(payload.get("messages"), list):
        normalized["messages"] = [
            {**message, "content": content(message.get("content"))}
            if isinstance(message, dict) and str(message.get("role", "")).lower() == "system"
            else message
            for message in payload["messages"]
        ]
    return normalized
