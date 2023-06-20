import torch

from ._cpp_extensions import monotonic_align as monotonic_align_cpp

__version__ = "0.0.0"
__all__ = ["viterbi_monotonic_alignment"]


@torch.no_grad()
def viterbi_monotonic_alignment(
    probs: torch.Tensor,
    take_log: bool = False,
    backend: str = "cpp",
) -> torch.LongTensor:
    """Search monotonic alignment by Viterbi algorithm.

    Args:
        probs (torch.Tensor): Soft alignment matrix of shape (batch_size, tgt_length, src_length).
        take_log (bool): If ``True``, log of ``probs`` is treated as log probability.
            Otherwise, ``probs`` is treated as log probability. Default: ``False``.
        backend (str): Language of backend implementation. 'cpp' and 'python' is supported.

    Returns:
        torch.LongTensor: Hard alignment matrix of shape (batch_size, tgt_length, src_length).

    """
    if backend == "cpp":
        hard_alignment = monotonic_align_cpp.viterbi_monotonic_alignment(probs, take_log)
    elif backend == "python":
        hard_alignment = viterbi_monotonic_alignment_py(probs, take_log)
    else:
        raise ValueError(f"Backend {backend} is not supported.")

    return hard_alignment


def viterbi_monotonic_alignment_py(
    probs: torch.Tensor,
    take_log: bool = False,
) -> torch.LongTensor:
    if take_log:
        log_probs = torch.log(probs)
    else:
        log_probs = probs

    _, max_tgt_length, max_src_length = log_probs.size()

    hard_align = torch.zeros_like(log_probs, dtype=torch.long)

    for batch_idx, log_prob in enumerate(log_probs):
        prob = torch.exp(log_prob)

        sum_prob = prob.sum(dim=1)
        tgt_length = torch.count_nonzero(sum_prob)
        tgt_length = tgt_length.item()

        sum_prob = prob.sum(dim=0)
        src_length = torch.count_nonzero(sum_prob)
        src_length = src_length.item()

        assert tgt_length >= src_length, "tgt_length should be longer than or equal to src_length."

        log_seq_prob = torch.full(
            (max_tgt_length, max_src_length), fill_value=-float("inf"), dtype=log_prob.dtype
        )
        log_seq_prob[0, 0] = 0

        for tgt_idx in range(1, tgt_length):
            for src_idx in range(
                max(0, src_length - tgt_length + tgt_idx), min(src_length, tgt_idx + 1)
            ):
                min_src_idx = max(0, src_idx - 1)
                max_src_idx = min(tgt_idx - 1, src_idx)
                prev_log_seq_prob = log_seq_prob[tgt_idx - 1, min_src_idx : max_src_idx + 1]
                log_seq_prob[tgt_idx, src_idx] = (
                    torch.max(prev_log_seq_prob) + log_prob[tgt_idx, src_idx]
                )

        src_idx = src_length - 1
        hard_align[batch_idx, tgt_length - 1, src_idx] = 1

        for tgt_idx in range(tgt_length - 1, 0, -1):
            min_src_idx = max(0, src_idx - 1)
            max_src_idx = min(tgt_idx - 1, src_idx)
            prev_log_seq_prob = log_seq_prob[tgt_idx - 1, min_src_idx : max_src_idx + 1]
            src_idx = min_src_idx + torch.argmax(prev_log_seq_prob)
            hard_align[batch_idx, tgt_idx - 1, src_idx] = 1

    return hard_align
