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

[Code on GitHub](https://github.com/hectopascal/tinyvlm-implementation)

# Why I built this

I built a tiny vision-language model because I wanted to understand what happens below the library abstraction.

Not “call AutoModelForVision2Seq and hope for the best” understand. I mean the slightly more annoying version: how image embeddings actually enter a language model, what the projector is doing, why the training is staged, and what breaks when the setup moves from one GPU to multi-GPU FSDP.

The project had two parts.

First, I implemented a small VLM using a SigLIP-2 vision encoder, a Qwen2.5 language model, and a two-layer MLP projector. I also implemented the image-token splice manually: replace the `<image>` placeholder token in the text sequence with projected image patch embeddings, then feed the resulting continuous multimodal sequence into the LM.

Second, I scaled the setup with FSDP across 2, 4, and 8 V100s, using a larger Qwen2.5-1.5B LM and data from LLaVA-Pretrain. I expected scaling efficiency to degrade at 8 GPUs. Instead, my throughput benchmark reported superlinear speedup.

Naturally, this was suspicious. Computers are many things, but they are rarely generous.

# The model

SigLIP vision encoder -> projector + Qwen LM token stream.

The important mental model is that the image is not magical to the language model. After projection, image patches become embedding vectors inserted into the LM’s sequence.

The splice operation is the runtime trick that makes this work. The text contains an `<image>` bookmark token. During multimodal preprocessing, that bookmark is replaced with the image patch embeddings. The LM then sees one long embedding sequence: some text, then image-derived vectors, then more text.

This is the part I wanted to implement by hand. Not because the code is glamorous, but because this is where the abstraction becomes concrete. A lot of VLM architecture becomes less mysterious once you see that the “multimodal” part is, in practice, a carefully arranged embedding sequence.

# Training stages

The intended training setup had two stages.

In stage 1, the projector is trained to align the vision encoder’s output with the language model’s embedding space. The vision encoder and LM are mostly fixed; the projector learns to produce embeddings that the LM can consume usefully.

In stage 2, LoRA adapters in the LM are trained together with the projector, while the base LM weights remain frozen. At this point, the model can also adapt how it responds to those multimodal inputs.

The linked implementation trains the projector and LoRA adapters together.

I cared more about the implementation and scaling behavior than squeezing out the best VLM quality. The early model produced short, on-topic, generic captions, which was enough for a basic check of the data path. The interesting part came later, when the training loop met distributed systems and immediately became less innocent.

# Scaling setup

For the scaling study, I used Qwen2.5-1.5B as the language model and trained with FSDP across 2, 4, and 8 V100s. The original plan was to use A100s. Then cloud pricing performed its usual spiritual cleansing exercise on my ambitions, so V100s it was.

The saved benchmark configurations use a 10,000-example subset of LLaVA-Pretrain and fp16, keeping the effective batch size at 32 by adjusting gradient accumulation. The throughput numbers below use the training loop's padded-text-token metric.[^throughput]

# Results

## The interesting result: superlinear scaling?

The first strong-scaling result looked great.

Too great.

The no-checkpoint runs appeared to show superlinear scaling: 8 GPUs reported **5.80× the throughput** of 2 GPUs, above the 4× ideal.

{% include figure.liquid loading="eager" path="assets/img/scaling.png" class="img-fluid rounded z-depth-1" alt="Throughput and peak allocated memory across 2, 4, and 8 GPUs, with and without activation checkpointing." caption="Reported throughput, mean ± sample SD across runs, and rank-0 peak allocated memory." %}

My suspicion was that the 2-GPU baseline used the hardware inefficiently. Adding GPUs could then improve throughput both by adding resources and by reducing overhead per unit of work.

Memory pressure was one candidate: the 2-GPU baseline reported 12.94 GB of peak allocated tensor memory on nominally 16 GB GPUs. Activation checkpointing reduced that figure to 9.93 GB and brought the endpoint ratio close to linear: **4.07× from 2 to 8 GPUs, about 102% of ideal**.

That looked reassuring, but checkpointing also lowered absolute throughput, especially at 8 GPUs. Its recomputation cost changed the comparison too, so the more ordinary ratio did not prove that memory pressure caused the original result.

The lesson: inspect the baseline before celebrating the scaling number. A smaller configuration that uses the hardware poorly can make the speedup look unusually impressive. That remained my working explanation here; the experiment did not isolate the underlying bottleneck.

## Profiling

Then I profiled the 8-GPU run. FSDP all-gathers kept the NCCL stream busy, while the model-compute stream had gaps between layers. That suggested parameter communication was worth investigating.

{% include figure.liquid loading="eager" path="assets/img/nockpt.png" class="img-fluid rounded z-depth-1" alt="Profiler crop without forward prefetch, showing repeated NCCL all-gathers and gaps between model-compute kernels." caption="Forward-pass crop without forward prefetch." %}

So I tried `fsdp_forward_prefetch=True`, which requests the next forward all-gather earlier. The idea was to make parameters available sooner and reduce waiting.

{% include figure.liquid loading="eager" path="assets/img/nockpt_prefetch.png" class="img-fluid rounded z-depth-1" alt="Profiler crop with forward prefetch, showing NCCL all-gathers alongside model-compute kernels." caption="Forward-pass crop with prefetch. The screenshots use different time scales." %}

The prefetch trace showed tightly packed all-gathers and visible overlap with compute, though some overlap was already present without it.

However, the runs showed no throughput gain: **3,059 ± 103 tok/s** with prefetch versus **3,164 ± 59 tok/s** without it (mean ± sample SD, four runs each).

My working explanation was bandwidth contention: fetching earlier may not help much if transfers already compete for the same links. I did not measure link utilization, so saturation remained a hypothesis. What I could see was that changing the schedule did not improve the reported throughput.

# What I learned

My main takeaway is to inspect the baseline and the absolute rates before interpreting a scaling ratio.

The first result said: “8 GPUs gives 5.80× speedup over 2 GPUs.”

My working interpretation is: “The 2-GPU configuration may be an inefficient baseline, so adding GPUs may also reduce overhead per unit of work.”

The distinction matters: a scaling ratio reflects how well both configurations use their hardware.

The second lesson is that profiler traces need to be checked against performance measurements. `fsdp_forward_prefetch=True` changed the trace without improving throughput. An optimization needs to earn its place in the measurements, even if the trace looks more aesthetically pleasing.

The third lesson is that hardware topology belongs in the setup. The actual connections between GPUs matter when deciding what communication can be hidden.

# Where I'd go next

First, I would compare full-iteration wall time for equal global work and profile both the 2- and 8-GPU runs. That would help distinguish the baseline's memory and communication costs.

Given more time and a more emotionally supportive GPU budget, I would rerun the study on an A100 system with bf16 and compare across a known interconnect topology.

If those measurements show all-gather communication dominating, I would compare pure FSDP against tensor parallelism or hybrid parallelism.

Finally, I would use Nsight Systems alongside `torch.profiler` to examine host scheduling and GPU activity across ranks in more detail.

# Conclusion

This project started as a way to demystify VLM internals. The implementation part made the architecture feel less magical: image embeddings are projected, spliced into the token stream, and consumed by the LM as part of one continuous sequence.

The scaling part was more interesting. A suspiciously good result made me look harder at the baseline. Checkpointing changed the memory footprint and scaling curve; prefetch changed the trace without improving throughput. My working explanation was still that the smaller configuration used the hardware inefficiently, but a plausible explanation and an isolated cause are different things.

I got a result that looked too good, did not trust it, and used it to work out what to measure next.

Which, in machine learning systems, is often where the actual engineering begins.

_Updated 7 September 2026 to clarify the measurement and interpretation, and correct the prefetch standard deviation._

[^throughput]: The [saved loop](https://github.com/hectopascal/tinyvlm-implementation/blob/c1656c2ef4a169605ea8ac860240d70f9d85248c/train_fsdp.py#L95-L140) averages the final 50 microstep rates, counting padded text positions on rank 0 and multiplying by GPU count. Timing starts after batch fetch; inserted image embeddings are excluded from the count. This is an approximate training-body rate, not end-to-end useful-token throughput.
