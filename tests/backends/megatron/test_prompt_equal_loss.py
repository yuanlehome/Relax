# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU oracle for the frozen Dressage prompt objective and MCore scaling."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from relax.backends.megatron.prompt_loss import prompt_denominators, scale_prompt_reducer, validate_prompt_equal_loss


class PromptDenominatorTest(unittest.TestCase):
    def test_contexts_are_counted_once_and_prompts_merge_across_dp(self) -> None:
        result, tokens = prompt_denominators([[(0, 10, 12), (0, 10, 12), (1, 20, 30)], [(2, 10, 9)], []], 256)
        self.assertEqual(result, {10: 21 * 2 / 256, 20: 30 * 2 / 256})
        self.assertEqual(tokens, 51)

    def test_rejects_duplicate_dp_ownership_and_inconsistent_contexts(self) -> None:
        for rows in ([[(0, 0, 1)], [(0, 0, 1)]], [[(0, 0, 1), (0, 1, 1)]], [[(0, 0, 1.0)]]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                prompt_denominators(rows, 256)

    def test_unsupported_modes_fail_before_training(self) -> None:
        config = dict(
            prompt_equal_loss=True,
            train_backend="megatron",
            use_agentic_rollout=True,
            hybrid=True,
            advantage_estimator="grpo",
            loss_type="policy_loss",
            calculate_per_token_loss=True,
            num_steps_per_rollout=1,
            num_iters_per_train_update=1,
        )
        validate_prompt_equal_loss(SimpleNamespace(**config))
        for value in (0.0, 0, [0.0], [0.0, 0.0], (0.0,)):
            with self.subTest(zero_aux_coefficients=value):
                validate_prompt_equal_loss(SimpleNamespace(**config, moe_aux_loss_coeff=value))
        for value in ([0.0, 0.01], (0.01,), float("nan"), [float("inf")], True, [False], [], None, "0.0"):
            with self.subTest(invalid_aux_coefficients=value), self.assertRaises(ValueError):
                validate_prompt_equal_loss(SimpleNamespace(**config, moe_aux_loss_coeff=value))
        for key, value in (
            ("dynamic_context_parallel", True),
            ("num_steps_per_rollout", 2),
            ("calculate_per_token_loss", False),
            ("moe_aux_loss_coeff", 0.01),
            ("use_opd", True),
            ("enable_mtp_training", True),
            ("train_backend", "fsdp"),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_prompt_equal_loss(SimpleNamespace(**{**config, key: value}))

    def test_cp_microbatch_value_and_gradient_match_source_objective(self) -> None:
        import torch

        from relax.backends.megatron import cp_utils

        # Unequal contexts and two Sessions from the same source prompt. The
        # tiny third prompt exercises the source clamp_min(D, 1) branch.
        row_prompts = [0, 0, 0, 1, 2]
        mask_counts = [73, 39, 191, 227, 1]
        original_masks = [torch.arange(256) < count for count in mask_counts]
        original_total = sum(mask_counts)
        denominators, total = prompt_denominators(
            [[(0, 0, 112), (0, 0, 112), (1, 0, 191)], [(2, 1, 227), (3, 2, 1)]], 256
        )
        self.assertEqual(total, original_total)
        for cp in (1, 2, 32):
            for effective_batch in (4, 256):
                for reject in (False, True):
                    with self.subTest(cp=cp, effective_batch=effective_batch, tis_rejection=reject):
                        parameters = torch.linspace(-0.4, 0.4, 1280, dtype=torch.float64, requires_grad=True)
                        vectors = parameters.reshape(5, 256)
                        # Token-level clipped PG and fixed TIS weights. Rejected
                        # tokens change only numerator masks, never D or S.
                        ratio = vectors.exp()
                        advantages = torch.tensor([1, -1, 0.3, -0.8, 2], dtype=torch.float64)[:, None]
                        pg = torch.maximum(-advantages * ratio, -advantages * ratio.clamp(0.8, 1.28))
                        tis = torch.linspace(0.6, 1.5, 256, dtype=torch.float64)
                        losses = pg * tis
                        masks = [mask & ((torch.arange(256) % 3 != 0) if reject else True) for mask in original_masks]
                        reference = (
                            sum(
                                (losses[i] * masks[i]).sum() / max(denominators[prompt], 1.0)
                                for i, prompt in enumerate(row_prompts)
                            )
                            / effective_batch
                        )
                        actual = torch.zeros((), dtype=torch.float64)
                        normalizer = torch.zeros((), dtype=torch.int64)
                        # One row per microbatch, including all-empty CP shards.
                        for rank in range(cp):
                            for i, prompt in enumerate(row_prompts):
                                kwargs = dict(dynamic_cp_size=cp, dynamic_cp_rank=rank)
                                sliced = cp_utils.slice_log_prob_with_cp(losses[i], 384, 256, **kwargs)
                                reducer = cp_utils.get_sum_of_sample_mean(
                                    [384], [256], [masks[i]], sample_denoms=[denominators[prompt]], **kwargs
                                )
                                actual += scale_prompt_reducer(
                                    reducer,
                                    {
                                        "prompt_loss_total_tokens": [total],
                                        "dynamic_global_batch_size": effective_batch,
                                    },
                                )(sliced)
                                normalizer += cp_utils.get_cp_local_num_tokens(
                                    [384], [256], [original_masks[i]], **kwargs
                                )
                        self.assertEqual(normalizer.item(), original_total)
                        actual = actual / normalizer
                        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
                        actual_grad = torch.autograd.grad(actual, parameters, retain_graph=True)[0]
                        reference_grad = torch.autograd.grad(reference, parameters)[0]
                        torch.testing.assert_close(actual_grad, reference_grad, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
