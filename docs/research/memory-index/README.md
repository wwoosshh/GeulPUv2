# Research: memory index in the weights

[한국어](README.ko.md)

A study of where an agent keeps the index of its long-term memory. The memory contents stay in files; only the index — which
memories exist and in what situation each is needed — is trained into a LoRA adapter on the model's weights.

**Status: an idea under test. Nothing here is a feature of GeulPU.** If the idea holds up, it may be tried in GeulPU as an
experimental option beside ordinary context memory.

| Document | English | 한국어 |
|---|---|---|
| Proposal — the problem, the design, prior work, what is not shown | [proposal.md](proposal.md) | [proposal.ko.md](proposal.ko.md) |
| Evidence from the first test — 200 memories, a 4B model, criteria fixed beforehand | [evidence.md](evidence.md) | [evidence.ko.md](evidence.ko.md) |

Archived on Zenodo with a date: [10.5281/zenodo.24184984](https://doi.org/10.5281/zenodo.24184984) (draft 1, 2026-11-13). All
versions: [10.5281/zenodo.24184983](https://doi.org/10.5281/zenodo.24184983).

This study ran in PyTorch, outside GeulPU, so it is not in [experiments/](../../../experiments/), which holds experiments run
with GeulPU. Data, scripts and later results (a comparison on a larger model is in progress) will be added here and to the
Zenodo record.
