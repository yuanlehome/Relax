# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Enable source compaction settings and materialize harness-owned contexts.

The configured harness stages these settings inside its OCI sandbox and returns
canonical contexts via metadata or explicit JSONL. No history is reconstructed
from final text or collected by the Gateway.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

from examples.nemo_gym_agentic.app.result import explicit_context_records, write_relax_output


def read_explicit_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("Compaction-enabled Claude Code produced no explicit context exports")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        expanded = explicit_context_records(payload) if isinstance(payload, dict) else None
        records = expanded if expanded is not None else [payload]

    names: set[str] = set()
    outcomes: list[float] = []
    for record in records:
        if not isinstance(record, dict) or "messages" not in record:
            raise ValueError(
                "Compaction requires metadata.contexts or explicit JSONL; reward-only output is insufficient"
            )
        name = record.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Every explicit context requires a unique nonempty name")
        names.add(name)
        if not isinstance(record["messages"], list) or not record["messages"]:
            raise ValueError("Every context requires its complete canonical messages")
        if not isinstance(record.get("tools"), list) or not isinstance(record.get("chat_template_kwargs"), dict):
            raise ValueError("Every context requires its exact tools and chat_template_kwargs")
        metadata = record.get("metadata")
        outcome = metadata.get("outcome_reward") if isinstance(metadata, dict) else None
        if isinstance(outcome, bool) or not isinstance(outcome, (int, float)) or not math.isfinite(outcome):
            raise ValueError("Every context requires a finite numeric metadata.outcome_reward")
        outcomes.append(float(outcome))
    if any(outcome != outcomes[0] for outcome in outcomes):
        raise ValueError("All contexts from this Session must carry the same outcome_reward")
    return records


def main() -> None:
    command = os.environ["CLAUDE_CODE_AGENT_COMMAND"].strip()
    if not command:
        raise ValueError("CLAUDE_CODE_AGENT_COMMAND must name a Harbor/OCI adapter")
    environment = dict(os.environ)
    environment.pop("DISABLE_COMPACT", None)
    environment["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = "200000"
    environment["CLAUDE_CODE_SETTINGS_PATH"] = str(Path(__file__).with_name("claude_settings.json"))
    # Inherit the Relax-owned process group so cancellation includes the adapter.
    subprocess.run(["bash", "-c", command], env=environment, check=True)
    output = Path(environment["RELAX_OUTPUT_JSON"])
    records = read_explicit_records(output)
    outcome = records[0]["metadata"]["outcome_reward"]
    # This recipe's advantage is computed once per logical Session. Keep one
    # reward anchor for outcome metrics even when compaction creates many rows.
    for record in records:
        record["reward"] = None
    records[-1]["reward"] = outcome
    write_relax_output(output, {"reward": outcome, "metadata": {"contexts": records}})


if __name__ == "__main__":
    main()
