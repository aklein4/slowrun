# Paper-informed LR defaults

Question: do the revised peak LR and linear schedule execute at full model size?
The [paper's larger-model sweep and schedule study](https://arxiv.org/pdf/2606.25971)
motivate raising effective matrix/gain peak LR from .004 to .008 and linearly
decaying all groups to an absolute 1e-8. Embedding/head peaks remain .003/.001;
no warmup. This is a transfer hypothesis, not a measured optimum for this model.
Noise control remains independent of LR, so this is not fixed-temperature sampling.

One 1,399,476,473-parameter run, seeds 42/43, used the previous four training
sequences and separate validation sequence for five epochs (20 updates), at
2048 tokens/update. Noise warmup 2, interval 1, growth 1.2, target 10 exercise
feedback. Snapshots were disabled because their unchanged path was checked in
the [preceding study](../2026-09-26--paper-md/README.md). The prior command bounds
were two minutes, 96GiB GPU and no new checkpoint storage. Environment: one GH200,
PyTorch 2.7.0/CUDA12.8, FlashAttention2.7.4.post1, eager, W&B disabled.

Observed: final training EMA 2.796387, best validation CE 9.553281, final validation
CE 9.711126. Noise activated at step 5, reaching .001540702 for the next update.
Peak allocated memory 37,708.67MiB; wall time 18.76s. All losses were finite.
Five numerical tests passed, including the schedule start/midpoint/end and effective
LR defaults. Documentation links and whitespace checks passed.

This tiny-data test checks execution; it does not establish generalization or
long-run stability. No additional LR sweep was performed. See [results.json](results.json),
[provenance.json](provenance.json), and [reproduce.sh](reproduce.sh).
Exact output is logged under the ignored `paper-lr` agent task.
