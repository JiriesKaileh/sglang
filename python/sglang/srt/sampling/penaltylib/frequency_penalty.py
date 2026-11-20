import dataclasses

import torch

from sglang.kernels.ops.sampling.penalties import TokenPenalty, cumulate_padded_tokens
from sglang.srt.sampling.penaltylib.orchestrator import _BatchedPenalizer
from sglang.srt.sampling.penaltylib.speculative import SpeculativePenalties
from sglang.srt.utils.device import is_pin_memory_available


class BatchedFrequencyPenalizer(_BatchedPenalizer):
    """
    Frequency penalizer penalizes tokens based on their frequency in the output.
    """

    def _is_required(self) -> bool:
        return any(
            req.sampling_params.frequency_penalty != 0.0
            for req in self.orchestrator.reqs()
        )

    def _prepare(self):
        pin_memory = is_pin_memory_available(self.orchestrator.device)
        self.cumulated_frequency_penalties = torch.zeros(
            (len(self.orchestrator.reqs()), self.orchestrator.vocab_size),
            dtype=torch.float32,
            device=self.orchestrator.device,
        )

        self.frequency_penalties = (
            torch.tensor(
                data=[
                    req.sampling_params.frequency_penalty
                    for req in self.orchestrator.reqs()
                ],
                dtype=torch.float32,
                pin_memory=pin_memory,
            )
            .to(self.orchestrator.device, non_blocking=True)
            .unsqueeze_(1)
        )

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        self.cumulated_frequency_penalties.scatter_add_(
            dim=1,
            index=output_ids.unsqueeze(1),
            src=self.frequency_penalties,
        )

    def _cumulate_padded_output_tokens(
        self, output_ids: torch.Tensor, output_lens: torch.Tensor
    ):
        cumulate_padded_tokens(
            self.cumulated_frequency_penalties,
            self.frequency_penalties,
            output_ids,
            output_lens,
            accumulate=True,
        )

    def _add_speculative_state(
        self, penalties: SpeculativePenalties
    ) -> SpeculativePenalties:
        return dataclasses.replace(
            penalties,
            frequency=TokenPenalty(
                coefficients=self.frequency_penalties,
                table=self.cumulated_frequency_penalties,
            ),
        )

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        logits.sub_(self.cumulated_frequency_penalties)

    def _filter(self, keep_indices: torch.Tensor):
        self.frequency_penalties = self.frequency_penalties[keep_indices]
        self.cumulated_frequency_penalties = self.cumulated_frequency_penalties[
            keep_indices
        ]

    def _merge(self, their: "BatchedFrequencyPenalizer"):
        self.frequency_penalties = torch.cat(
            [self.frequency_penalties, their.frequency_penalties], dim=0
        )
        self.cumulated_frequency_penalties = torch.cat(
            [self.cumulated_frequency_penalties, their.cumulated_frequency_penalties],
            dim=0,
        )

    def _teardown(self) -> None:
        for name in ("frequency_penalties", "cumulated_frequency_penalties"):
            if hasattr(self, name):
                delattr(self, name)
