# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Keep committed model contexts for a Claude Code Session, including
compaction.

The harness calls this with the exact forwarded Anthropic request and the
complete Relax response. Normalization deliberately uses the same
implementation as the Session API. Maximal leaves retain complete history;
shared generated responses declare exactly one owner for the recipe's real
training loss-mask conversion.
"""

from __future__ import annotations

import copy
import math
import threading
from typing import Any

from examples.claude_code_harbor.training_data import RESPONSE_MASK_KEY


def canonical_context(request: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    # These dependencies already belong to the Relax training image. Keep them
    # lazy so dataset/staging utilities do not import the training runtime.
    from relax.agentic.session.service import _anthropic_input_messages, _normalized_anthropic_request
    from relax.agentic.session.state import check_messages

    if response.get("type") != "message" or response.get("role") != "assistant":
        raise ValueError("Context capture requires a successful Anthropic assistant message")
    normalized = _normalized_anthropic_request(request)
    assistant = check_messages(
        _anthropic_input_messages([{"role": "assistant", "content": response["content"]}], None)
    )
    if len(assistant) != 1 or assistant[0]["role"] != "assistant":
        raise ValueError("Model response must project to one canonical assistant message")
    return {
        "messages": [*normalized["messages"], *assistant],
        "tools": normalized["tools"],
        "chat_template_kwargs": normalized["chat_template_kwargs"],
    }


def _state_hash(context: dict[str, Any], messages: list[dict[str, Any]] | None = None) -> str:
    from relax.agentic.session.state import _messages_tools_template_state_hash

    return _messages_tools_template_state_hash(
        context["messages"] if messages is None else messages,
        context["tools"],
        context["chat_template_kwargs"],
    )


def _is_prefix(parent: dict[str, Any], child: dict[str, Any]) -> bool:
    return (
        parent["tools"] == child["tools"]
        and parent["chat_template_kwargs"] == child["chat_template_kwargs"]
        and len(parent["messages"]) <= len(child["messages"])
        and parent["messages"] == child["messages"][: len(parent["messages"])]
    )


class ContextCapture:
    """Session-owned context store; it is released when the adapter exits."""

    def __init__(self) -> None:
        self._leaves: dict[str, dict[str, Any]] = {}
        self._generated_states: set[str] = set()
        self._lock = threading.Lock()
        self._sealed = False

    def record(self, request: dict[str, Any], response: dict[str, Any]) -> None:
        context = canonical_context(request, response)
        state_hash = _state_hash(context)
        with self._lock:
            if self._sealed:
                raise RuntimeError("Context capture is closed")
            self._generated_states.add(state_hash)
            # An older request can complete after its descendant. Keep the
            # descendant in that case, and collapse exact replayed responses.
            if any(_is_prefix(context, leaf) for leaf in self._leaves.values()):
                return
            self._leaves = {key: leaf for key, leaf in self._leaves.items() if not _is_prefix(leaf, context)}
            self._leaves[state_hash] = context

    def seal(self) -> None:
        """Freeze capture before the harness exports terminal records."""
        with self._lock:
            self._sealed = True

    def export(self, outcome: float, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if isinstance(outcome, bool) or not isinstance(outcome, (int, float)) or not math.isfinite(outcome):
            raise ValueError("Session outcome must be a finite number")
        with self._lock:
            if not self._leaves:
                raise ValueError("Claude Code completed without a captured model response")
            records: list[dict[str, Any]] = []
            exported_states: set[str] = set()
            names = [f"context_{index}" for index in range(len(self._leaves))]
            for name, (state_hash, context) in zip(names, self._leaves.items(), strict=True):
                generated = []
                for end, message in enumerate(context["messages"]):
                    if message["role"] == "assistant":
                        response_hash = _state_hash(context, context["messages"][: end + 1])
                        if response_hash in self._generated_states:
                            generated.append(response_hash)
                selected = [value for value in generated if value not in exported_states]
                if not selected:
                    raise ValueError("Each maximal context must own at least one committed response")
                exported_states.update(generated)
                records.append(
                    {
                        "name": name,
                        **copy.deepcopy(context),
                        "metadata": {
                            **copy.deepcopy(metadata or {}),
                            "outcome_reward": float(outcome),
                            RESPONSE_MASK_KEY: {
                                "version": 1,
                                "context_name": name,
                                "context_names": list(names),
                                "leaf_state_hash": state_hash,
                                "response_state_hashes": generated,
                                "train_response_hashes": selected,
                            },
                        },
                    }
                )
            return records
