# Research: memory index in the weights

[한국어](README.ko.md)

A study of where an agent keeps the index of its long-term memory. The memory contents stay in files; only the index — which
memories exist and in what situation each is needed — is trained into a LoRA adapter on the model's weights.

**Status: tested and not adopted (2026-11-21). Nothing here is a feature of GeulPU.** The advantage found in the first test, on
synthetic data, did not carry over to memories from a public benchmark, and there is no plan to try the design in GeulPU. The
conclusion and the numbers are in [results.md](results.md).

| Document | English | 한국어 |
|---|---|---|
| Later results — a public benchmark, real conversations, the conclusion (2026-11-22) | [results.md](results.md) | [results.ko.md](results.ko.md) |
| Proposal — the problem, the design, prior work, what is not shown | [proposal.md](proposal.md) | [proposal.ko.md](proposal.ko.md) |
| Evidence from the first test — 200 memories, a 4B model, criteria fixed beforehand | [evidence.md](evidence.md) | [evidence.ko.md](evidence.ko.md) |

Archived on Zenodo with a date: [10.5281/zenodo.24184984](https://doi.org/10.5281/zenodo.24184984) (draft 1, 2026-11-13). All
versions: [10.5281/zenodo.24184983](https://doi.org/10.5281/zenodo.24184983).

The proposal and the evidence note are kept as they were written on 2026-11-13, with a note at the top that points to the later
results.

This study ran in PyTorch, outside GeulPU, so it is not in [experiments/](../../../experiments/), which holds experiments run
with GeulPU. Data and scripts are not public.
