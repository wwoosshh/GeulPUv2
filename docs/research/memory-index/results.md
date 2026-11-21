# Memory index in the weights: later results

2026-11-22 · Sunghyeun Woo

Korean version: [results.ko.md](results.ko.md) · The proposal: [proposal.md](proposal.md) · Evidence from the first test: [evidence.md](evidence.md)

This note records what was measured after the first test, and the conclusion.

## 1. Conclusion

The advantage found on synthetic data did not carry over to memories that I did not make.

Training the index has a cost: about 20 training texts per memory, and a training run. So the method has to be clearly more accurate than the same model reading the index in its context. The later measurements do not show that. **I do not recommend this design in place of an index in the context, and there is no plan to try it in GeulPU.**

## 2. Main numbers

Both methods were measured with Qwen3.5-4B unless stated. 27B is Qwen3.8-27B (4-bit).

| Data | Memories | Measure | Index in context | Index in weights | Status |
|---|---|---|---|---|---|
| Synthetic (mine) | 200 | Top-1 accuracy | 43.3 % | 74.0 % | criteria fixed beforehand |
| Synthetic (mine) | 1,000 | Top-1 accuracy | 18.1 % | 66.6 % | no verdict was planned |
| Synthetic (mine) | 1,000, both methods with 27B | Top-1 accuracy | 70.8 % | 73.3 % | criteria fixed beforehand: equivalent |
| PrefEval | 10 to 20 | Correct choices (of 970) | 871 | 767 | criteria fixed beforehand |
| PrefEval | 700 | Correct choices (of 970) | 719 | 652 | criteria fixed beforehand |
| PrefEval | 700, needed memory removed from the list | Correct choices (of 970) | 682 | 529 | control |
| PrefEval | 10 to 20 | Items that need the memory (of 265) | 184 | 155 | chosen after seeing results |
| PrefEval | 700 | Items that need the memory (of 217) | 28 | 65 | chosen after seeing results |
| PrefEval | 10 to 20, after ten unrelated turns | Correct choices (of 970) | 592 | 725 | criteria fixed beforehand |
| Real conversations | 172 memory files | Turns with the right file opened, of 40 turns that clearly need a memory | 0 | 5 | chosen after seeing results: no verdict |

PrefEval is a public benchmark of user preferences (Zhao et al., ICLR 2025, [arXiv:2502.09597](https://arxiv.org/abs/2502.09597)). One preference sentence is one memory. A request does not mention the preference, and one of four options fits it. Lists of 200 and 700 memories were made by adding other users' preferences.

## 3. What the later measurements show

- **Memories from the benchmark, list right before the request.** The index in the context is higher at 10 to 20, at 200 and at 700 memories.
- **That score needs a control.** With the needed memory taken out of the list, the model reading the list still gets 643 to 683 of 970 items right, against 376 with no memory at all. So I also counted only the items that both methods get wrong without the needed memory.
- **On those items** the index in the context is higher with 10 to 20 memories (184 against 155 of 265). With 700 memories the index in the weights is higher (65 against 28 of 217), but 65 is 30 % where giving the needed memory directly reaches 76 %. Neither method is at a usable level there.
- **After ten unrelated turns** the index in the weights is higher (725 against 592). This cannot be read as an effect of the index being in the weights: the control was not measured in this condition, and the two conditions place the memory at different positions.
- **The level on synthetic data did not appear.** The top-1 accuracy of the trained index was 90.0, 74.0 and 66.6 % on my synthetic sets of 20, 200 and 1,000 memories, and 76.1, 47.9 and 36.4 % on PrefEval with 10 to 20, 200 and 700 candidates.
- **A larger model.** With the 27B model the two methods were equivalent already on synthetic data.
- **Real conversations.** On my own working conversations only 40 turns clearly needed a memory. No verdict was possible.

## 4. What this changes in the earlier documents

- The claim of the [evidence note](evidence.md) holds on that synthetic set. It does not extend to requests written by others: the training texts and the test requests of the first test were both written by language models from information about each memory, and the level measured there did not appear on the benchmark's requests.
- Sections 7 (planned work) and 8 (possible use in GeulPU) of the [proposal](proposal.md) are withdrawn.

## 5. What remains

- **The problem.** The index in the context also fails to use the needed memory. On the items that need it, its share fell from 69.4 % to 12.9 % as the list grew from 10 to 20 memories to 700, and its four-option score fell from 871 to 592 after ten unrelated turns.
- **Two points on evaluation.** A trained index has to be evaluated on requests made independently of the procedure that made its training texts. And the four-option score of PrefEval rises without the needed memory, so comparing memory methods on it needs a control with the needed memory removed.

## 6. Limits

- The PrefEval results come from one 4B model, one seed and one training run. The 27B model was not measured there.
- There is one public benchmark. Its memories are one-sentence preferences, and its requests are everyday requests for recommendations.
- One phrase that tells the model to use the preference was removed from the benchmark's wording, and the rule for lengthening the lists is mine. This is not the benchmark's official task.
- The comparisons marked "chosen after seeing results" were not planned before the run.
- The real conversations are one person's, and the answer key was set by a judge model.
- Not measured: contents stored in the weights, dedicated memory modules, embedding retrieval placed at the same position, models larger than the 27B.

## 7. Record

- Measured 2026-11-13 to 2026-11-21. The criteria of the results marked "criteria fixed beforehand" were written in a dated local file before each run; that record is local, not public.
- A full report with the method, the setup and every table exists as an unpublished draft.
- Data, scripts and per-item scores are not public.
- **How AI was used.** As in the earlier documents: AI coding agents wrote the scripts, ran the measurements and drafted this note under the author's direction. Language models wrote the training texts and acted as the judge model for the real conversations.
