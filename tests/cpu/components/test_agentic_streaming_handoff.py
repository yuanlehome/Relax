# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise streaming handoff with the real TQ controller and sampler, no cluster."""

import pickle
from argparse import Namespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from relax.components import advantages as advantages_module
from relax.utils.data.identity_window_sampler import IdentityWindowSampler
from relax.utils.types import Sample
from relax.utils.utils import convert_samples_to_train_data, put_rollout_batch


def _args(**overrides):
    values = dict(
        fully_async=True,
        use_dynamic_batch_size=True,
        global_batch_size=512,
        num_iters_per_train_update=1,
        n_samples_per_prompt=8,
        advantage_estimator="score_centering",
        rollout_top_logprobs_num=2,
        score_centering_vocab_size=8,
        custom_convert_samples_to_train_data_path=None,
        custom_reward_post_process_path=None,
        agentic_custom_advantage_path="already-computed-by-group",
        debug_train_only=False,
        use_rollout_indexer_replay=False,
        multimodal_keys=None,
        use_opd=False,
        use_rollout_logprobs=True,
        true_on_policy_mode=True,
        kl_coef=0,
        use_kl_loss=False,
    )
    values.update(overrides)
    return Namespace(**values)


class _NoScalarList(np.ndarray):
    def tolist(self):
        raise AssertionError("SC candidates must not materialize Python scalar lists")


def _samples(group):
    rows = []
    for member in range(8):
        # One logical Session exports two physical rows, with different lengths.
        for export in range(2 if member == 0 else 1):
            length = 2 + export
            rows.append(
                Sample(
                    index=group * 8 + member,
                    group_index=group,
                    tokens=[0] + [1] * length,
                    response_length=length,
                    loss_mask=[1] * length,
                    rollout_log_probs=[float(np.log(0.6))] * length,
                    rollout_topk_token_ids=np.tile(np.array([[1, 2]], dtype=np.int32), (length, 1)),
                    rollout_topk_log_probs=np.tile(np.log(np.array([[0.6, 0.4]], dtype=np.float32)), (length, 1)),
                    rollout_topk_support_complete=np.ones(length, dtype=np.bool_),
                    custom_advantage=float(member) - 3.5,
                )
            )
    return rows


def _run_inline(coroutine):
    # The only asynchronous boundaries are the in-memory transport methods below.
    try:
        coroutine.send(None)
    except StopIteration as result:
        return result.value
    finally:
        coroutine.close()
    raise AssertionError("CPU transport unexpectedly suspended")


def test_sc_tensor_conversion_preserves_wire_values_dtypes_and_multirow_denominators():
    samples = _samples(0)
    for sample in samples:
        sample.rollout_topk_token_ids = sample.rollout_topk_token_ids.view(_NoScalarList)
        sample.rollout_topk_log_probs = sample.rollout_topk_log_probs.view(_NoScalarList)
    data = pickle.loads(pickle.dumps(convert_samples_to_train_data(_args(), samples)))
    assert isinstance(data, TensorDict)
    assert data["sample_indices"].tolist() == [0, 0, 1, 2, 3, 4, 5, 6, 7]
    assert data["sample_index_mask_sums"].tolist() == [5, 5, 2, 2, 2, 2, 2, 2, 2]
    for i, sample in enumerate(samples):
        ids, probs = data["rollout_topk_token_ids"][i], data["rollout_topk_log_probs"][i]
        assert ids.dtype == torch.int64
        assert probs.dtype == torch.float32
        assert ids.shape == probs.shape == (2 * sample.response_length,)
        torch.testing.assert_close(ids.reshape(-1, 2), torch.tensor([[1, 2]] * sample.response_length))
        torch.testing.assert_close(
            probs.reshape(-1, 2), torch.log(torch.tensor([[0.6, 0.4]] * sample.response_length))
        )
        assert data["rollout_topk_support_complete"][i].tolist() == [True] * sample.response_length


def test_first_group_reaches_actor_before_gbs_and_advantages_drain_all_512_identities(monkeypatch):
    module = pytest.importorskip("transfer_queue.controller")
    from transfer_queue.metadata import extract_field_schema

    controller_type = module.TransferQueueController.__ray_metadata__.modified_class
    for method in ("_init_zmq_socket", "_start_process_handshake", "_start_process_request"):
        monkeypatch.setattr(controller_type, method, lambda _self: None)
    monkeypatch.setenv("TQ_PRE_ALLOC_SAMPLE_NUM", "1")
    sampler = IdentityWindowSampler(dp_size=2, placement="streaming", balance_unit_multiplier=8)
    controller = controller_type(sampler=sampler, polling_mode=True)
    args = _args()
    rows = {}
    received = []
    requests = []
    inserted_groups = []
    first_microbatch = []

    class Transport:
        async def async_check_stream_drained(self, task_name, partition_id):
            return controller.check_stream_drained(partition_id, task_name)

        async def async_get_meta(self, **kwargs):
            requests.append(kwargs["batch_size"])
            # Bound regressions instead of hanging in Advantages' polling loop.
            assert len(requests) <= 64, "advantages still waits for a full GBS or lost an identity"
            return controller.get_metadata(**kwargs)

        async def async_get_data(self, metadata):
            data = {}
            for field in metadata.field_names:
                values = [rows[index][field] for index in metadata.global_indexes]
                data[field] = torch.nested.nested_tensor(values) if values[0].ndim else torch.stack(values)
            return TensorDict(data, batch_size=[metadata.size])

        async def async_put(self, *, data, metadata=None, partition_id="train_0", custom_meta=None, is_last=False):
            data = pickle.loads(pickle.dumps(data))
            is_insert = metadata is None
            if is_insert:
                metadata = controller.get_metadata(
                    list(data.keys()), partition_id, mode="insert", batch_size=len(data), is_last=is_last
                )
            indexes = metadata.global_indexes
            for offset, index in enumerate(indexes):
                rows.setdefault(index, {}).update({field: data[field][offset] for field in data.keys()})
            schema = extract_field_schema(data)
            # Storage RPCs key each row shape by its allocated global index.
            for field_schema in schema.values():
                if field_schema.get("per_sample_shapes") is not None:
                    field_schema["per_sample_shapes"] = dict(
                        zip(indexes, field_schema["per_sample_shapes"], strict=True)
                    )
            assert controller.update_production_status(
                partition_id,
                indexes,
                schema,
                user_custom_meta=dict(zip(indexes, custom_meta, strict=True)) if custom_meta is not None else None,
            )
            if is_insert:
                inserted_groups.append(is_last)
                return
            received.extend(indexes)
            if len(inserted_groups) == 1:
                assert len(rows) == 9
                assert not controller.check_stream_drained("train_0", "compute_advantages_and_returns")
                for dp_rank in range(2):
                    batch = controller.get_metadata(
                        ["tokens", "advantages"],
                        "train_0",
                        task_name="actor_train",
                        token_budget=65536,
                        sampling_config=dict(
                            dp_rank=dp_rank,
                            dp_size=2,
                            batch_index=0,
                            rollout_mini_index=0,
                            window_quota=512,
                            allow_underfill=True,
                        ),
                    )
                    assert batch.size > 0
                    first_microbatch.extend(batch.global_indexes)
                assert set(first_microbatch) == set(indexes)
                assert len(first_microbatch) == len(set(first_microbatch)) == 9
                assert not sampler.is_window_drained("train_0", "actor_train", 0, 512, False, False)
            if len(inserted_groups) < 64:
                group = len(inserted_groups)
                await put_rollout_batch(args, self, "train_0", _samples(group), is_last=group == 63)

    transport = Transport()
    _run_inline(put_rollout_batch(args, transport, "train_0", _samples(0), is_last=False))
    worker = object.__new__(advantages_module.Advantages.func_or_class)
    worker._logger_instance = None
    worker.config = args
    worker.data_system_client = transport
    monkeypatch.setattr(advantages_module, "run_", _run_inline)
    worker._compute_partition(0, worker._advantage_fields())

    assert requests == [8] * 64
    assert inserted_groups == [False] * 63 + [True]
    assert len(received) == len(set(received)) == 576
    assert len({int(row["sample_indices"]) for row in rows.values()}) == 512
    assert controller.check_stream_drained("train_0", "compute_advantages_and_returns")
    for row in rows.values():
        expected = torch.full((int(row["response_lengths"]),), int(row["sample_indices"]) % 8 - 3.5)
        torch.testing.assert_close(row["advantages"], expected)
        torch.testing.assert_close(row["returns"], expected)
    assert args.global_batch_size == 512 and args.num_iters_per_train_update == 1
