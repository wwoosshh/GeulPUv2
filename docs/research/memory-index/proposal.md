# Memory index in the weights: a proposal for agent memory whose contents stay in files

Draft 1 · 2026-11-13 · Sunghyeun Woo · status: idea under test, not a feature of GeulPU

Archived on Zenodo: [10.5281/zenodo.24184984](https://doi.org/10.5281/zenodo.24184984) (draft 1) · all versions [10.5281/zenodo.24184983](https://doi.org/10.5281/zenodo.24184983)

Korean version: [proposal.ko.md](proposal.ko.md) · Evidence so far: [evidence.md](evidence.md)

## Summary

An agent's long-term memory today is a set of files plus an index of those files. The index is either loaded into the context at every session, or kept in a retrieval system outside the model.

This note proposes a third place for the index. The memory contents stay in files, unchanged. Only the index is trained into a small adapter (LoRA) on the model's weights: which memories exist, and in what situation each one is needed. Before answering, the model writes the identifier of the memory it needs, or "none", and the runtime loads that file.

A first test supports the idea on a small scale. With 200 invented memories, a 4B model with the index trained in picked the needed memory for 74.0 % of requests that never mention the memory. The same model reading the index in its context picked it for 43.3 %. The adapter cost about 3 % in loss on unrelated text. Details and limits are in the evidence note.

I searched for the same design in papers, products and public repositories and did not find it. That is a statement about my search, not a claim to be first.

## 1. The problem

Take an agent with this memory:

> Tests and models run only on the SSD. The HDD is for archives.

and this request:

> Start the benchmark from the copy on the slow, large disk.

The request does not name the rule and shares no distinctive word with it. To act correctly the agent must recall the rule unprompted. Current systems handle this in two ways.

- **Index in the context.** A list with one line per memory is put into the prompt at every session, and the model opens the files it judges relevant. This works while the list is short. It costs tokens at every turn, so products cap it (one widely used coding agent reads the first 200 lines or 25 KB of its index file [1]). Past the cap, memories are invisible unless something else finds them.
- **Retrieval.** The request is matched against the memory texts by embedding similarity. It scales, but it finds what resembles the request. A request that needs a memory often does not resemble it, as in the example above.

A third line of research trains the memory *contents* into the weights. Reported problems are consistent across papers: older memories are overwritten by newer ones, changed facts are handled poorly, and the model asserts things that were never stored [2][3][4].

## 2. The proposal

Split memory into contents and index, and put only the index in the weights.

| | Contents | Index |
|---|---|---|
| Current products | files | context (re-read every session) |
| Retrieval (RAG) | files / database | external vector index |
| Memory-in-weights research | weights | none, or external |
| **This proposal** | **files** | **weights (adapter)** |

Concretely:

1. **Each memory is a file with an identifier.** The file is the source of truth. A person can read, edit and delete it.
2. **The adapter learns "situation → identifier".** Training pairs are requests in which the memory would be needed, written from the memory file alone, paired with the memory's identifier. Requests that need no memory are paired with "none".
3. **Before each answer the model fills a memory slot** with one identifier or "none". The choice is restricted to identifiers that exist, so the model cannot point at a memory that is not there. The runtime loads the chosen file into the context and the model answers.
4. **When memories change, the adapter is retrained**, in idle time. New memories can sit in the context until then.

What this is expected to buy, by construction:

- The per-turn cost does not grow with the number of memories. Nothing is read.
- Recall does not depend on the request resembling the memory text. The association is learned.
- Content errors of weight memory are avoided. The fact itself is read from the file, so it is exact, current, and deletable. A wrong pick loads a wrong file, which is visible and checkable.

What it gives up: a memory created a minute ago is not in the adapter yet; the adapter belongs to one model version; and a trained index cannot be reasoned over as a whole the way a visible list can.

I therefore see it as a division of labour, not a replacement: recent memories in the context, the long-lived bulk in the adapter.

## 3. Why the training signal matters

Prior work predicts that the obvious way to do this fails. Knowledge fine-tuned into a model is recalled when asked for and rarely used unprompted: 99 % against 6–7 % in one study of per-user adapters [4]. In tool learning, training on tool descriptions alone gave about 10 % top-1 accuracy over 46,985 tools, and training on "user query → tool token" pairs gave about 88 % [5].

So the index must not be taught as a list to memorise ("memory 17 is about disks"). It must be taught in the direction it will be used: from the situation to the identifier. The first test compares these directly, and the difference is large (74.0 % against 15.9 %).

## 4. Relation to prior work

The principle of an index in the weights is not new. What I did not find is its application to an agent's own memory, measured against the index-in-context baseline.

- **Generative retrieval** (DSI [6], DSI++ [7]) trains a model to output document identifiers; documents stay outside. The corpus is large and public, built once, and the task is answering a query. Continual updates cause forgetting (35.9 → 19.2 Hits@1 in [7]).
- **Tools as tokens** (ToolGen [5], ToolkenGPT [8]) has the same structure with tools in place of memories. Tools rarely change.
- **Agent memory in the weights** (for example [2][3][4], and a public project that stores "key → fact" in an adapter with the key list on disk [9]) trains contents. One paper names a learned index as future work without testing it [3].
- **Models that pick memory identifiers from a list in their context** (for example [10]) and **separately trained retrievers** keep the index outside the answering model's weights.
- **Products and start-ups** that use weights for memory, as far as public material shows, train contents.

Search scope: about fifty queries in English and Korean over two days (2026-11-12 and 11-13), covering papers, preprints, products, start-ups and public repositories. Not covered: patents, unpublished company work, and other languages. Roughly half of the closest sources appeared after June 2026, and several are not peer reviewed.

## 5. Evidence so far

One test, with criteria written before any run. Full numbers are in the [evidence note](evidence.md).

- 200 invented memories of one fictional developer; a 4B open-weight model; requests written by a different model from the one that wrote the training texts.
- On 411 requests that do not mention the memory and need exactly one: index trained in, 74.0 %; index in context (same model), 43.3 %; embedding retrieval over the memory files, 37.7 %; index trained from descriptions, 15.9 %.
- On requests that need no memory, the trained model produced an identifier in 0.0 % of cases; the context baseline in 39.8 %.
- Cost: loss on unrelated text rose by 0.067 nat (about 3 %), above the limit I had set (0.05). A smaller adapter (rank 4) kept the accuracy and halved the cost in a follow-up that was not part of the registered test.
- As an extra, a 27B model reading the same index reached 78.1 % on the same requests (68.2 % on all 600, equal to the trained 4B). Training the index brought the 4B model's unprompted recall to about the level of an unmodified model roughly seven times larger. On direct questions about the memories the 27B model is far better (85.8 % against 57.8 %).

## 6. What is not shown

- **Scale.** 200 memories, trained once. Thousands or tens of thousands are untested.
- **Change.** Adding, editing and deleting memories every day is untested. Prior work says this is where weight-based indexes break.
- **Large models.** Whether a large model gains from a trained index over reading it is being measured now (same 27B model, trained against reading, 1,000 memories of which 800 are written to be confusable). There are no results yet.
- **A trained retriever.** The same "situation → identifier" pairs could train a small separate retrieval model. If that matches the adapter, the gain comes from a learned index, not from putting it in the answering model's weights. Not yet measured.
- **Real memories.** The memories and requests are invented and in English.
- **The whole loop.** The test measures which memory is picked. An auxiliary run that also generated answers is small (100 items).

## 7. Planned work

1. Finish the large-model comparison.
2. Scale to thousands and tens of thousands of memories.
3. Daily change: add, edit, delete, and the cost of retraining.
4. The trained-retriever baseline.
5. Teaching descriptions, questions and situations together, to repair the weakness on direct questions.

Results will be added to later versions of this record as they come, including results that go against the idea.

## 8. Possible use in GeulPU

GeulPU runs downloaded open-weight models locally, so the weights can carry an adapter. If the idea holds up, I intend to try it there as an experimental option beside ordinary context memory. Nothing of this exists in the engine today. An open design question is where the adapter training runs, since GeulPU is an inference engine.

## 9. Notes

- **How AI was used.** The study was planned and carried out by the author working with AI coding agents. The agents wrote the scripts, ran the measurements, and drafted this text under the author's direction; the author set the questions and approved the criteria before measurement. Language models also produced the invented memories, the training texts and the test requests, and acted as blind checkers and judges; the evidence note says which role each kind of model had.
- **Compute.** The measurements ran on an H100 GPU provided by a collaborating research group.
- **Version history.** Draft 1, 2026-11-13: proposal and first test.

## References

1. Memory documentation of a widely used coding agent: https://code.claude.com/docs/en/memory
2. Fang et al., *Route to Remember* (2026): https://arxiv.org/abs/2611.05636
3. Pan et al. (2026), bioRxiv: https://www.biorxiv.org/content/10.64898/2026.08.24.746712v1.full
4. Li, *User as Engram* (2026): https://arxiv.org/abs/2606.19172
5. Wang et al., *ToolGen* (ICLR 2025): https://arxiv.org/abs/2410.03439
6. Tay et al., *Transformer Memory as a Differentiable Search Index* (2022): https://arxiv.org/abs/2202.06991
7. Mehta et al., *DSI++* (2022): https://arxiv.org/abs/2212.09744
8. Hao et al., *ToolkenGPT* (NeurIPS 2023): https://arxiv.org/abs/2305.11554
9. ParaMem (public repository): https://github.com/tiberius1701d/paramem
10. Gan, *CoG-MeM* (2026): https://engrxiv.org/preprint/view/6547
