# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise response ownership against real Forest Samples and the
converter."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest

from examples.claude_code_harbor.advantage import advantage_func
from examples.claude_code_harbor.contexts import ContextCapture
from examples.claude_code_harbor.training_data import RESPONSE_MASK_KEY, convert_samples_to_train_data


torch = pytest.importorskip("torch")
pytest.importorskip("ray")

from relax.agentic.session.state import SessionForest, check_messages  # noqa: E402


class Tokenizer:
    def decode(self, tokens: list[int], **kwargs: Any) -> str:
        return str(tokens)


def args() -> SimpleNamespace:
    return SimpleNamespace(
        custom_convert_samples_to_train_data_path=(
            "examples.claude_code_harbor.training_data.convert_samples_to_train_data"
        ),
        custom_reward_post_process_path=None,
        agentic_custom_advantage_path="examples.claude_code_harbor.advantage.advantage_func",
        advantage_estimator="grpo",
        multimodal_keys=None,
        use_opd=False,
        debug_train_only=True,
    )


def branched_samples(session_id: str = "session", index: int = 0) -> list[Any]:
    forest = SessionForest.create_empty(session_id=session_id, index=index, group_index=0)
    capture = ContextCapture()

    def observation(parent: str, messages: list[dict[str, Any]], tokens: list[int]) -> Any:
        return forest.append_obs(
            parent_state_hash=parent,
            rollout_id=0,
            abort_count=0,
            messages_delta=check_messages(messages),
            train_token_delta=tokens,
            rollout_token_delta=tokens,
        )

    def response(parent: Any, text: str, tokens: list[int], abort_count: int = 0) -> Any:
        request = {"model": "proxy-model", "max_tokens": 8192, "messages": forest.full_messages(parent.state_hash)}
        result = {"type": "message", "role": "assistant", "content": [{"type": "text", "text": text}]}
        capture.record(request, result)
        return forest.append_resp(
            parent_state_hash=parent.state_hash,
            rollout_id=0,
            abort_count=abort_count,
            messages_delta=[{"role": "assistant", "content": text}],
            token_delta=tokens,
            logprob_delta=[-0.5] * len(tokens),
        )

    initial = observation(forest.root_state_hash, [{"role": "user", "content": "task"}], [1])
    # One completed response may contain accumulated partial/resume tokens.
    shared = response(initial, "shared", [2, 3, 4], abort_count=1)
    left_obs = observation(shared.state_hash, [{"role": "user", "content": "left result"}], [10, 11])
    left = response(left_obs, "left", [5, 6])
    right_obs = observation(shared.state_hash, [{"role": "user", "content": "right result"}], [12])
    right = response(right_obs, "right", [7])
    # Compaction starts an independent committed context and remains exported.
    summary_obs = observation(forest.root_state_hash, [{"role": "user", "content": "summary"}], [13])
    summary = response(summary_obs, "continued", [8, 9])
    records = capture.export(1.0)
    assert len(records) == 3
    samples = []
    for record, leaf in zip(records, [left, right, summary], strict=True):
        sample = forest.build_sample(leaf_state_hash=leaf.state_hash, tokenizer=Tokenizer())
        sample.metadata.update(record["metadata"])
        sample.custom_advantage = 0.75
        samples.append(sample)
    return samples


def test_shared_response_mask_uses_real_converter_and_unique_token_denominator() -> None:
    samples = branched_samples()
    original = copy.deepcopy(samples)
    config = args()
    result = convert_samples_to_train_data(config, samples)
    assert result["loss_masks"] == [[1, 1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 1], [1, 1]]
    assert result["sample_index_mask_sums"] == [8, 8, 8]
    assert result["rewards"] == [0.75, 0.75, 0.75]
    assert result["tokens"] == [sample.tokens for sample in original]
    assert result["rollout_log_probs"] == [sample.rollout_log_probs for sample in original]
    assert result["response_lengths"] == [sample.response_length for sample in original]
    assert samples == original
    assert config.custom_convert_samples_to_train_data_path == args().custom_convert_samples_to_train_data_path
    assert convert_samples_to_train_data(config, samples) == result
    reordered = convert_samples_to_train_data(config, list(reversed(samples)))
    assert reordered["loss_masks"] == list(reversed(result["loss_masks"]))


def test_shared_response_hashes_are_scoped_to_each_logical_session() -> None:
    result = convert_samples_to_train_data(args(), branched_samples("first", 0) + branched_samples("second", 1))
    assert result["sample_index_mask_sums"] == [8] * 6
    assert sum(sum(mask) for mask in result["loss_masks"]) == 16


def test_budget_harvest_preserves_real_tokens_and_marks_all_contexts_truncated() -> None:
    samples = branched_samples()
    for sample in samples:
        sample.metadata.update(termination_reason="backend_timeout", outcome_reward=0.0, _loss_prompt_index=7)
    original = copy.deepcopy(samples)
    config = args()
    config.prompt_equal_loss = True
    result = convert_samples_to_train_data(config, samples)
    assert result["truncated"] == [1, 1, 1]
    assert result["loss_prompt_ids"] == [7, 7, 7]
    assert result["tokens"] == [sample.tokens for sample in original]
    assert result["sample_index_mask_sums"] == [8, 8, 8]
    assert samples == original


def test_shared_response_plan_preserves_logical_grpo_normalization() -> None:
    samples = branched_samples()
    exports = {sample.metadata[RESPONSE_MASK_KEY]["context_name"]: sample.metadata for sample in samples}
    loser = copy.deepcopy(exports)
    for metadata in loser.values():
        metadata["outcome_reward"] = 0.0
    advantages = advantage_func([exports, loser])
    assert advantages == [{name: 1.0 for name in exports}, {name: -1.0 for name in loser}]
    assert advantage_func([exports, exports]) is None


@pytest.mark.parametrize(
    "fault",
    [
        "missing_context",
        "duplicate_owner",
        "no_owner",
        "wrong_hash",
        "wrong_leaf",
        "wrong_length",
        "mask",
        "logprobs",
        "tokens",
        "identity",
        "removed",
    ],
)
def test_shared_response_mask_rejects_inconsistent_contracts(fault: str) -> None:
    samples = branched_samples()
    if fault == "missing_context":
        samples.pop()
    elif fault == "duplicate_owner":
        plan = samples[1].metadata[RESPONSE_MASK_KEY]
        plan["train_response_hashes"] = list(plan["response_state_hashes"])
    elif fault == "no_owner":
        samples[0].metadata[RESPONSE_MASK_KEY]["train_response_hashes"].pop(0)
    elif fault == "wrong_hash":
        samples[0].metadata[RESPONSE_MASK_KEY]["response_state_hashes"][0] = "wrong"
    elif fault == "wrong_leaf":
        samples[0].metadata[RESPONSE_MASK_KEY]["leaf_state_hash"] = "wrong"
    elif fault == "wrong_length":
        samples[0].metadata["agentic_trace"]["turns"][0]["response_length"] += 1
    elif fault == "mask":
        samples[0].loss_mask[3] = 1
    elif fault == "logprobs":
        samples[1].rollout_log_probs[0] = -9.0
    elif fault == "tokens":
        samples[1].tokens[1] = 999
    elif fault == "identity":
        samples[0].session_id = "wrong"
    elif fault == "removed":
        samples[0].remove_sample = True
    with pytest.raises(ValueError):
        convert_samples_to_train_data(args(), samples)


def test_recipe_launch_enables_real_response_mask_converter() -> None:
    from pathlib import Path

    script = Path(__file__).resolve().parents[3] / "examples/claude_code_harbor/scripts/run-qwen36-35B-A3B-128xgpu.sh"
    assert (
        "--custom-convert-samples-to-train-data-path "
        "examples.claude_code_harbor.training_data.convert_samples_to_train_data"
    ) in script.read_text()


def test_shared_response_masks_reach_real_tis_and_cp_loss_reducers() -> None:
    pytest.importorskip("megatron.core", reason="Megatron is required to import the production TIS loss")
    from relax.backends.megatron import cp_utils
    from relax.backends.megatron.loss import vanilla_tis_function

    data = convert_samples_to_train_data(args(), branched_samples())
    lengths = data["response_lengths"]
    totals = data["total_lengths"]
    masks = [torch.tensor(value, dtype=torch.float64) for value in data["loss_masks"]]
    rollout = [torch.tensor(value, dtype=torch.float64) for value in data["rollout_log_probs"]]
    train = [
        -0.1 * torch.tensor(tokens[-length:], dtype=torch.float64)
        for tokens, length in zip(data["tokens"], lengths, strict=True)
    ]
    pg = [torch.arange(1, length + 1, dtype=torch.float64, requires_grad=True) for length in lengths]
    tis_args = SimpleNamespace(tis_clip_low=0.9, tis_clip=1.1)
    pg_sum = torch.tensor(0.0, dtype=torch.float64)
    metric_sums = {}
    count = 0
    for rank in range(2):
        cp = {"dynamic_cp_size": 2, "dynamic_cp_rank": rank}

        def sliced(values: list[Any]) -> list[Any]:
            return [
                cp_utils.slice_log_prob_with_cp(value, total, length, **cp)
                for value, total, length in zip(values, totals, lengths, strict=True)
            ]

        corrected, returned_masks, metrics = vanilla_tis_function(
            tis_args,
            pg_loss=torch.cat(sliced(pg)),
            train_log_probs=sliced(train),
            rollout_log_probs=sliced(rollout),
            loss_masks=masks,
            total_lengths=totals,
            response_lengths=lengths,
        )
        assert returned_masks is masks
        reducer = cp_utils.get_sum_of_sample_mean(
            totals, lengths, masks, calculate_per_token_loss=True, qkv_format="thd", **cp
        )
        pg_sum += reducer(corrected)
        for name, values in metrics.items():
            metric_sums[name] = metric_sums.get(name, 0.0) + reducer(values)
        count += cp_utils.get_cp_local_num_tokens(totals, lengths, masks, qkv_format="thd", **cp).item()

    valid = torch.cat(masks).bool()
    delta = torch.cat(train) - torch.cat(rollout)
    ratio = delta.exp()
    weights = ratio.clamp(0.9, 1.1)
    assert count == 8
    torch.testing.assert_close(pg_sum / count, (torch.cat(pg) * weights)[valid].mean())
    oracle = {
        "tis": ratio,
        "tis_clipfrac": (weights != ratio).to(torch.float64),
        "tis_abs": (ratio - 1).abs(),
        "mismatch_kl": -delta,
        "mismatch_k3_kl": ratio - delta - 1,
    }
    for name, values in oracle.items():
        torch.testing.assert_close(metric_sums[name] / count, values[valid].mean())
    (pg_sum / count).backward()
    torch.testing.assert_close(torch.cat([value.grad for value in pg]), weights * valid / count)
