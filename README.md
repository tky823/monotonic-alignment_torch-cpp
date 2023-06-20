# Monotonic align
Implementation of monotonic alignment by cpp extension of PyTorch.

## Installation
```sh
pip install git+https://github.com/tky823/monotonic-alignment_torch-cpp.git
```

## Example
```python
>>> import torch
>>> from monotonic_align import viterbi_monotonic_alignment
>>> torch.manual_seed(0)
>>> batch_size = 4
>>> max_src_length = 30
>>> max_tgt_length = 2 * max_src_length
>>> tgt_lengths = torch.randint(max_tgt_length // 2, max_tgt_length + 1, (batch_size,))
>>> src_lengths = torch.randint(max_src_length // 2, max_src_length + 1, (batch_size,))
>>> max_tgt_length, max_src_length = torch.max(tgt_lengths), torch.max(src_lengths)
>>> log_probs = torch.randn((batch_size, max_tgt_length, max_src_length))
>>> tgt_indices, src_indices = torch.arange(max_tgt_length), torch.arange(max_src_length)
>>> padding_mask = src_indices >= src_lengths.unsqueeze(dim=-1).unsqueeze(dim=-1)
>>> log_probs = log_probs.masked_fill(padding_mask, -float("inf"))
>>> probs = torch.softmax(log_probs, dim=-1)
>>> padding_mask = tgt_indices.unsqueeze(dim=-1) >= tgt_lengths.unsqueeze(dim=-1).unsqueeze(dim=-1)
>>> probs = probs.masked_fill(padding_mask, 0)
>>> probs.shape
torch.Size([4, 60, 26])
>>> hard_attn_py = viterbi_monotonic_alignment(probs, take_log=True, backend="python")
>>> hard_attn_cpp = viterbi_monotonic_alignment(probs, take_log=True, backend="cpp")
>>> torch.equal(hard_attn_py, hard_attn_cpp)
True
```

## Note
To take the best advantage of C++ implementation, you might need to fix `setup.py`:

```python
class BuildExtension(_BuildExtension):
    cpp_extensions = [
        {
            "name": "monotonic_align._cpp_extensions.monotonic_align",
            "sources": [
                "cpp_extensions/monotonic_align.cpp",
            ],
            # add extra_compile_args
            "extra_compile_args": [
                "-fopenmp",
            ],
        },
    ]
```
