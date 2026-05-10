---
layout: post
title: A VLM, FSDP, and the Lie My Strong-Scaling Numbers Told Me
date: 2026-05-08 15:09:00
description: An Engineering Case Study
tags: distributed-training fsdp profiling vlm
categories: engineering
giscus_comments: true
featured: true
toc:
  beginning: true
---

[link to github repo](https://github.com/hectopascal/tinyvlm-implementation)

# Why I built this

I built a tiny vision-language model because I wanted to understand what happens below the library abstraction.

Not “call AutoModelForVision2Seq and hope for the best” understand. I mean the slightly more annoying version: how image embeddings actually enter a language model, what the projector is doing, why the training is staged, and what breaks when the setup moves from one GPU to multi-GPU FSDP.

The project had two parts.

First, I implemented a small VLM using a SigLIP-2 vision encoder, a Qwen2.5 language model, and a two-layer MLP projector. I also implemented the image-token splice manually: replace the `<image>` placeholder token in the text sequence with projected image patch embeddings, then feed the resulting continuous multimodal sequence into the LM.

Second, I scaled the setup with FSDP across 2, 4, and 8 V100s, using a larger Qwen2.5-1.5B LM and the full LLaVA-Pretrain dataset. I expected scaling efficiency to degrade at 8 GPUs. Instead, I initially got superlinear speedup.

Naturally, this was suspicious. Computers are many things, but they are rarely generous.

# The model

SigLIP vision encoder -> projector + Qwen LM token stream.

The important mental model is that the image is not magical to the language model. After projection, image patches become embedding vectors inserted into the LM’s sequence.

The splice operation is the runtime trick that makes this work. The text contains an `<image>` bookmark token. During multimodal preprocessing, that bookmark is replaced with the image patch embeddings. The LM then sees one long embedding sequence: some text, then image-derived vectors, then more text.

This is the part I wanted to implement by hand. Not because the code is glamorous, but because this is where the abstraction becomes concrete. A lot of VLM architecture becomes less mysterious once you see that the “multimodal” part is, in practice, a carefully arranged embedding sequence.

# Training stages

I used a two-stage training setup.

In stage 1, the projector is trained to align the vision encoder’s output with the language model’s embedding space. The vision encoder and LM are mostly fixed; the projector learns to produce embeddings that the LM can consume usefully.

In stage 2, the LM is unfrozen and fine-tuned together with the projector, using LoRA. At this point, the model can adapt more broadly, but it is no longer just learning “how do I translate vision features into LM-space?” It is also changing how the LM responds to those multimodal inputs.

I cared more about the implementation and scaling behavior than squeezing out the best VLM quality. The model produced short, on-topic, generic captions after stage 1, which was enough to confirm that the data path worked. The interesting part came later, when the training loop met distributed systems and immediately became less innocent.

# Scaling setup

For the scaling study, I used Qwen2.5-1.5B as the language model and trained with FSDP across 2, 4, and 8 V100s. The original plan was to use A100s. Then cloud pricing performed its usual spiritual cleansing exercise on my ambitions, so V100s it was.

# Results

## The interesting result: superlinear scaling?

The first strong-scaling result looked great.

Too great.

The no-checkpoint runs appeared to show superlinear scaling: 8 GPUs looked **5.8× faster** than 2 GPUs, significantly above the 4× ideal. 

{% include figure.liquid loading="eager" path="assets/img/scaling.png" class="img-fluid rounded z-depth-1" %}

I reran the experiment and profiled the memory which showed that the 2-GPU baseline was under memory pressure, using about 12.94GB of 16GB, so the performance was likely worse than what a clean baseline should be. The original baseline was unhealthy. 

Activation checkpointing reduced 2-GPU memory to 9.93GB and gave a much more honest near-linear result: **4.07× from 2 to 8 GPUs, about 102% of ideal**.

The lesson: strong-scaling numbers are only as honest as the baseline. If the smallest configuration is memory-bound, larger configurations can look artificially impressive. You are not measuring pure parallel efficiency anymore. You are measuring parallelism plus the relief of memory pressure.


## Profiling

Then I profiled the 8-GPU run. The trace showed communication-bound behavior. FSDP all-gathers were keeping the NCCL stream busy, while the compute stream had idle gaps between layers. In other words, the GPUs were often waiting for parameters to arrive before they could do useful work.

{% include figure.liquid loading="eager" path="assets/img/nockpt.png" class="img-fluid rounded z-depth-1" %}
Measuring within a single ProfilerStep, the compute stream was occupied roughly 40% of the time, while the NCCL stream ran continuously across the entire forward pass — three back-to-back nccl:_all_gather_base calls with no gaps. The GPU was idle more than half the step, waiting for parameters to arrive.

So I tried the textbook fix: fsdp_forward_prefetch=True. This allows the model to fetch parameters for the next layer early while it is computing the current layer. In theory, this should hide communication behind computation. PyTorch’s FSDP tutorial describes prefetching as a way to overlap all-gathers with computation.

{% include figure.liquid loading="eager" path="assets/img/nockpt_prefetch.png" class="img-fluid rounded z-depth-1" %}

In the profiler trace, it did what it was supposed to do. The NCCL all-gathers became more tightly packed, and there was visible overlap. 

However, throughput did not improve meaningfully: 3059 ± 99 tok/s versus 3164 ± 59 tok/s.

Diagnosis: bandwidth contention.The 8×V100 instance I used did not appear to have the all-to-all NVSwitch topology typical of newer A100/H100 training boxes. On V100 NVLink, the link is already saturated during all-gather, so overlapping doesn't help. We're just shifting bottleneck time around without reducing it. The trace shows overlap but the wall clock shows it didn't matter because the bandwidth ceiling was the limit, not the scheduling.

# What I learned

My main takeaway is to interpret my results with suspicion, and always make sure you have a clean baseline.

The first result said: “8 GPUs gives 5.80× speedup over 2 GPUs.”

The better interpretation was: “The 2-GPU baseline is memory-constrained, so the apparent scaling efficiency is inflated.”

Activation checkpointing fixed the interpretation. It reduced memory pressure in the baseline and recovered a much more honest scaling picture: near-linear 2-to-8 GPU scaling, not a miracle.

The second lesson is that profiler traces are necessary but not sufficient. fsdp_forward_prefetch=True produced the expected trace-level overlap, but it did not improve throughput. Optimization attempts need to be judged by end-to-end performance, not just by whether the trace looks more aesthetically pleasing.

The third lesson is that hardware topology matters. FSDP behavior on 8 V100s is not the same as FSDP behavior on newer A100/H100 clusters. At this scale, the bottleneck shifted from memory pressure to communication bandwidth.

# Where I'd go next

Given more time and a more emotionally supportive GPU budget, I would rerun the study on A100s with bf16 instead of V100s with fp16. V100s lack native bf16 support, and the 8-GPU interconnect topology makes FSDP all-gather behavior less favorable than on newer NVSwitch-based systems.

I would also compare pure FSDP against tensor parallelism or hybrid parallelism. FSDP is a good default, but once all-gather communication dominates, it is worth asking whether sharding parameters alone is the right parallelization strategy.

Finally, I would profile with Nsight Systems rather than relying only on torch.profiler. The PyTorch profiler was enough to identify the broad communication-bound pattern, but kernel-level analysis would give a cleaner view of where the time actually goes.

# Conclusion

This project started as a way to demystify VLM internals. The implementation part made the architecture feel less magical: image embeddings are projected, spliced into the token stream, and consumed by the LM as part of one continuous sequence.

The scaling part was more interesting. The headline result looked like superlinear scaling, but the real finding was that the baseline was memory-constrained. Once activation checkpointing relieved that pressure, the scaling story became much more honest: FSDP scaled close to linearly from 2 to 8 V100s, then became increasingly communication-bound.

That is probably the most useful outcome of the project. Not “I trained a tiny VLM.” Not even “I scaled it across 8 GPUs.”

More like: I got a result that looked too good, did not trust it, profiled it, and found the boring reason underneath.

Which, in machine learning systems, is often where the actual engineering begins.