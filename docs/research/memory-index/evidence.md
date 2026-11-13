# Memory index in the weights: evidence from the first test

Draft 1 · 2026-11-13 · Sunghyeun Woo

Archived on Zenodo: [10.5281/zenodo.24184984](https://doi.org/10.5281/zenodo.24184984) (draft 1) · all versions [10.5281/zenodo.24184983](https://doi.org/10.5281/zenodo.24184983)

Korean version: [evidence.ko.md](evidence.ko.md) · The proposal: [proposal.md](proposal.md)

This note states what the first test measured, what the numbers support, and what they do not.

## 1. Claim

A model that has not read its memory index can, when the index was trained into an adapter in the right way, pick the memory a request needs, including requests that never mention the memory. On a small model it does this better than the same model reading the index from its context.

The claim is limited to 200 memories trained once, on one 4B model, with invented data.

## 2. The task

- **Memories.** 200 invented memories of one fictional solo developer with four projects: 50 each of work rules, project facts, references, and facts about the user. Each has an identifier, a one-line index entry, and a file of 80–200 words.
- **The slot.** Before answering, the model fills a memory slot with one of the 200 identifiers or "none". Scoring reads the model's scores for these 201 options at the slot; nothing is generated. Chance is 0.5 %.
- **Example.** Memory: "Tests and models run only on the SSD. The HDD is for archives." Request: "Start the benchmark from the copy on the slow, large disk." Correct output: that memory's identifier.

## 3. What was compared

Trained into the weights (LoRA rank 16 on Qwen3.5-4B, three seeds each, same number of texts, same settings):

- **from situations** — requests in which the memory would be needed → identifier;
- **from descriptions** — the index entry and paraphrases of it → identifier;
- **from questions** — direct questions about the memory → identifier.

Baselines on the same requests:

- **index in context** — the unmodified 4B model with the 200-line index in its system prompt, filling the same slot;
- **retrieval** — embedding similarity (Qwen3-Embedding-0.6B) between the request and the memory files, the index entries, or the same training situations; and BM25;
- **frozen model** — the model untouched, only a small selection layer trained.

Training texts (20 per memory and kind, plus 4,600 no-memory requests) were written by an open 27B model that saw only each memory's title, index entry and file. Test requests were written separately by a proprietary model from a different vendor, from a one-sentence "when this memory is needed" note that neither the trained model nor the training-text writer saw.

## 4. Main result

Requests that share no distinctive word with the memory. Only requests for which a blind checker, given the index, selected exactly the intended memory are counted (411 of 600). Intervals are 95 % bootstrap intervals resampled by memory.

| Method | Correct | 95 % interval |
|---|---|---|
| **Index in weights, taught from situations** | **74.0 %** | 69.6 – 78.2 |
| Index in context (same 4B model) | 43.3 % | 37.5 – 49.0 |
| Frozen model, selection layer only | 38.7 % | 33.0 – 44.5 |
| Retrieval over memory files | 37.7 % | 32.1 – 43.5 |
| Retrieval over the training situations | 34.8 % | 29.9 – 39.6 |
| Retrieval over index entries | 24.3 % | 18.9 – 30.1 |
| BM25 | 23.8 % | 19.7 – 28.1 |
| Index in weights, taught from descriptions | 15.9 % | 12.8 – 19.3 |
| Index in weights, taught from questions | 13.5 % | 10.5 – 16.7 |

On all 600 requests the order is the same (68.2 % / 37.0 % / 13.8 % for situations / context / descriptions), and every verdict below is unchanged.

## 5. Verdicts against criteria fixed beforehand

| Question | Criterion | Value | Verdict |
|---|---|---|---|
| Does the idea stand? | situations ≥ 0.9 × context, and ≥ 60 % | 74.0 vs 43.3 | yes |
| Does the way of teaching decide it? | lower end of (situations − descriptions) ≥ 30 points | +58.1 (53.3 – 62.9) | yes |
| Better than retrieval on the same material? | interval of (situations − retrieval over situations) above 0 | +39.2 (33.5 – 44.9) | yes |
| No false recall? | on no-memory requests, not more than 5 points above context | 0.0 % vs 39.8 % | yes |
| Other abilities kept? | loss change on unrelated text: ≤ +0.02 nat kept, > +0.05 nat harmed | +0.067 nat | **harmed** |

The last row is a failure against my own criterion. With the adapter on, per-token loss on 189 Wikipedia passages (English and Korean) rose from 2.248 to 2.315 nat, about 3 %.

## 6. Further measurements

These were added after the registered test and were not used for the verdicts.

**Transfer across kinds of request.** What is taught is what comes out.

| Taught from | Descriptions | Questions | Requests sharing words with the memory | Requests sharing none |
|---|---|---|---|---|
| descriptions | 94.2 % | 57.4 % | 37.9 % | 15.9 % |
| questions | 79.5 % | 88.4 % | 37.0 % | 13.5 % |
| situations | 61.0 % | 57.8 % | 85.6 % | 74.0 % |
| (index in context) | 93.0 % | 71.5 % | 62.7 % | 43.3 % |

A model taught from situations is weaker than the context baseline when asked directly. Teaching the three kinds together was not measured.

**A larger model reading the same index.** Qwen3.8-27B (4-bit), same prompt, same index, same requests.

| | Trained 4B | 27B reading | 4B reading |
|---|---|---|---|
| Requests sharing no distinctive word with the memory (411) | 74.0 % | 78.1 % | 43.3 % |
| Same, all 600 requests | 68.2 % | 68.2 % | 37.0 % |
| Direct questions (400) | 57.8 % | 85.8 % | 71.5 % |
| Produced a memory on no-memory requests (284) | 0.0 % | 27.1 % | 39.8 % |

The difference between the trained 4B and the reading 27B on unprompted requests includes zero (−4.1 points, −9.6 to +1.3 on the 411; 0.0, −4.5 to +4.5 on the 600). The 411 were filtered by a checker that reads the index, which favours the reading side; the 600 are the less tilted figure. The 27B model also wrote the training texts, though it never saw the test requests.

**A smaller adapter.** Rank 4 instead of 16, three seeds: 74.9 % correct and +0.034 nat (about 1.5 %). The cost halved with no loss of accuracy. It did not reliably fall under the 0.02 criterion. These settings were compared on the same test requests, so the figure needs fresh requests to be confirmed.

**After a long unrelated conversation** (1,100–1,330 words before the request): the trained model went from 74.0 % to 71.7 % when "none" is excluded. When "none" is allowed it answered "none" more often (65.3 % → 53.6 % correct).

**To the end.** On 100 requests, the picked memory file was put in context and the unmodified 4B model wrote the answer. A blind judge marked the answer as following the memory in 39 cases with the trained model's pick, 28 with the context baseline's pick, 69 with the correct memory, and 4 with no memory.

## 7. What the numbers support

- An index trained from situations is recalled unprompted on new requests written by a different author (74.0 %), including the requests least similar to any training text (57.9 % in the lowest similarity band, against 31.6 % for the context baseline and 1.8 % for retrieval over the same training texts).
- The direction of teaching decides the outcome. The same index taught as descriptions or questions is recalled when asked and not when needed.
- The trained model does not fire on requests that need no memory (0 of 852 across three seeds).
- On a 4B model, a trained index brings unprompted recall to about the level of an unmodified 27B model reading the index.

## 8. What they do not support

- That the method is better than a large model reading the index. It was not, and that was not the aim.
- That it works for thousands of memories, or when memories change daily.
- That the adapter is free. The registered run exceeded the cost limit.
- That it beats a separately trained retriever. That baseline was not run. The embedding model used for retrieval is small (0.6B) and untrained on this data.
- Anything about real users' memories, other languages, or other model families.

## 9. Checks made

- Similarity between each test request and the nearest training situation of the same memory: median 0.63, maximum 0.85; no near-duplicates.
- 50 random unprompted requests read by hand: all 50 did need the memory.
- Two blind checkers agreed on "exactly one memory needed" for 91 % of 100 items (kappa 0.72).
- Every training run reached 100 % on its own training texts.
- Scoring the same model twice gave the same pick on all 2,700 items.

## 10. Record

- Hypotheses and criteria were written on 2026-11-12, before any run. Hashes of the plan, data and scripts were recorded in a local file at 02:38 on 2026-11-13, before the main measurement. That record was local, not public. Changes made while running are listed in the plan with the time, each written before seeing the result it concerns.
- Compute: about 5 hours 40 minutes on one H100 (47 GB slice), provided by a collaborating research group.
- Data, scripts and per-item scores are not included in draft 1. They will be added in a later version of this record.

## 11. How AI was used

AI coding agents wrote the scripts, ran the measurements and drafted this note under the author's direction. Language models had these roles in the experiment itself: an open 27B model wrote the training texts; a proprietary model from another vendor wrote the memories and the test requests, and acted as the blind checker of test items and the blind judge of generated answers. The models under test are open-weight models (Qwen3.5-4B, Qwen3.8-27B).
