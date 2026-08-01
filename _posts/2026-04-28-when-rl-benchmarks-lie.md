---
layout: post
title: When RL Benchmarks Lie
date: 2026-04-28 20:00:00
description: A measurement-theory view of RL post-training — what a benchmark number estimates, which uncertainties an error bar does and does not cover, and why a rising number can support far less than it appears to
tags: rl post-training evaluation reproducibility reasoning
categories: research
published: false
giscus_comments: true
featured: false
toc:
  beginning: true
---

<!-- This is the formal, reference-grade companion to the shorter opinion piece
     "Sharper, Not Smarter" (2026-04-14). Same thesis; here it is with definitions,
     derivations, and a bibliography. At publish time, consider adding a one-line
     pointer from that opinion post to this one. -->

A reported benchmark accuracy is not a context-free fact about a model. It is a *measurement* — an estimate of a stated quantity, produced by an instrument, and like any measurement it has an effective **resolution**: the scale of difference the design can distinguish at a stated level of uncertainty. That resolution depends on the design, the variance, the effect size, and the decision criterion. It also depends on which sources of variation the analysis ranges over. Physics undergraduates learn to report a central estimate with an uncertainty, to distrust digits finer than that uncertainty, and — the part that matters most here — to say what the interval covers. Machine learning, for reasons of culture and deadline pressure, often does none of the three. We report reinforcement-learning post-training results as bare point estimates — one training run, one decoding configuration, one harness — and then read them to a precision the instrument never supported.

This post makes that discomfort precise. The claim is not about fabrication. It is that many reports summarized as "RL improves reasoning" are not designed to distinguish capability expansion from three narrower explanations: a *sharpening* of an output distribution the model already had, a *sampling fluctuation* at the scale of the benchmark's uncertainty, and a *saturation* artifact where the benchmark has little headroom left to move. The empirical focus here is single-turn RLVR on statically scored reasoning tasks; interactive agent-environment RL may behave differently. When the instrument cannot resolve the difference, a table of point estimates cannot tell us which explanation is right.

I will build this in eight steps. §1 draws the central distinction — sharpening versus learning — and states the empirical result that motivates the whole piece. §2 formalizes the metric: Pass@1, Pass@k, and the unbiased estimator that one-sample evaluations cannot recover. §3 separates three sources of uncertainty that are often collapsed into one number. §4 is the core: a derivation showing that, under GRPO, an all-correct or all-wrong sampled group carries exactly zero *reward-discrimination* signal. §5 reframes reward hacking as the training-time analogue of a benchmark lie. §6 addresses peak-versus-stable reporting. §7 is a checklist. §8 lists what I think is still open.

The equations are numbered E1–E10; §2–§4 carry the weight.

# §1 · Sharpening versus learning

The phrase "RL improves reasoning" hides an ambiguity worth naming, because it can describe two distinct effects with different implications for what you have built. They are not mutually exclusive: a training run can both reweight solutions the base model already produced and expand finite-budget coverage.

The first meaning is **sharpening**: the model already assigns some probability to correct solutions, and reinforcement learning raises that probability relative to incorrect solutions. Under a fixed decoder and sampling budget, correct answers become easier to elicit even if the model's low-probability coverage has not improved.

The second meaning is **learning** (or capability expansion): solutions that had negligible probability under the base model acquire non-negligible probability under the trained one. "Reachable" cannot literally mean nonzero support — a softmax language model assigns nonzero probability to almost every finite sequence — so here it means *elicitable with non-negligible probability under a specified decoder and finite sampling budget*.

These hypotheses are not perfectly identifiable from one metric, but a Pass@k curve is much more diagnostic than Pass@1 alone. Pass@1 — the probability a single sample is correct — rewards sharpening directly. Pass@k at larger finite $k$ asks whether the model can solve each task at least once within $k$ attempts. It therefore probes finite-budget *task coverage*, including tasks whose per-sample success probability is low. It does not, by itself, measure the number or diversity of correct reasoning modes: one rare correct trajectory and many rare correct trajectories can produce the same Pass@k. For any problem with success probability $p>0$, $1-(1-p)^k\to1$ as $k\to\infty$; the useful regime is therefore large but finite $k$ under a fixed decoder.

The result that motivates this entire post is Yue et al. {% cite yue2025rlreasoning %}, who evaluate RLVR-trained models and their base models across a range of $k$ rather than stopping at Pass@1. They report a *crossover*: at small $k$ the RL model leads, while at large finite $k$ the base model matches or *overtakes* it. That pattern is consistent with RL concentrating probability on outputs that succeed more often while reducing or failing to expand finite-budget task coverage. It does not prove that no novel solution exists anywhere in the trained model's distribution, nor does the Pass@k curve alone identify which reasoning trajectories changed. It is evidence against the stronger claim that the studied RL procedures broadly expanded the empirical capability frontier. The same paper reports that distillation transfers reasoning patterns from a teacher and improves large-$k$ coverage in its experiments.

<!-- [needs figure] Fig. 1: Pass@k-reversal schematic. Two curves, Pass@k (y) vs k on log-x.
     RLVR curve starts above base at k=1, curves cross near mid-k, base overtakes at large k.
     Clearly labeled "schematic, after Yue et al. 2025" — NOT their measured numbers. -->
{% include figure.liquid path="assets/img/rl-lie-fig1-passk.png" class="img-fluid rounded z-depth-1" caption="Figure 1. Schematic of the Pass@k crossover (after Yue et al., 2025). An RLVR-trained policy leads at small k but is matched or overtaken by its base model at large finite k, which is more sensitive to tasks with low per-sample success probability. Illustrative shapes, not measured values." %}

Everything after this section is about avoiding an identification error: reading a result consistent with sharpening as if it uniquely demonstrated capability expansion.

To fix vocabulary, it helps to write down the objects the rest of the post manipulates. Modern LLM post-training optimizes a policy $\pi_\theta$ against a reward, regularized toward a reference policy:

$$\max_{\theta}\; \mathbb{E}_{x\sim\mathcal{D}}\!\left[\mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}\big[\,r_\phi(x,y)\,\big]\;-\;\beta\,\mathbb{D}_{\mathrm{KL}}\!\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x)\big)\right] \tag{E1}$$

**(E1)** Maximize the expected reward of completions $y$ drawn from the policy $\pi_\theta$ on prompts $x$, while a KL penalty of weight $\beta$ holds $\pi_\theta$ near a reference policy $\pi_{\mathrm{ref}}$ so it does not drift off-distribution {% cite ouyang2022instructgpt %}. This is the RLHF objective; RLVR replaces the learned reward $r_\phi$ with a programmatic verifier, but the shape is the same.

The optimizer moves $\theta$ along the policy gradient:

$$\nabla_\theta J(\theta)\;=\;\mathbb{E}\big[\,\nabla_\theta \log \pi_\theta(y\mid x)\,\hat{A}(x,y)\,\big] \tag{E2}$$

**(E2)** The reward-driven policy-gradient term is the score function $\nabla_\theta\log\pi_\theta$ weighted by an advantage estimate $\hat{A}$. If $\hat{A}=0$ for every sampled completion, that group contributes nothing to this *advantage-weighted term*. A KL penalty, entropy bonus, or other auxiliary loss can still contribute a gradient.

Two instantiations of $\hat{A}$ matter here. PPO {% cite schulman2017ppo %} combines a learned value baseline with a clipped surrogate: the importance ratio between the updated and sampling policies is clipped to a band around 1, so a single update cannot move the policy arbitrarily far. The clipping mechanism is not what the rest of this post turns on. The property needed below is simpler: on either branch of the clipped surrogate, the contribution is proportional to $\hat{A}_t$, so a completion with zero advantage contributes zero, clipped or not.

GRPO {% cite shao2024deepseekmath %} discards the learned critic and instead standardizes reward *within a group* of completions sampled for the same prompt. For a group of $G$ completions $\{y_i\}_{i=1}^{G}$ to prompt $x$ with rewards $\mathbf{r}=(r_1,\dots,r_G)$:

$$\hat{A}_i \;=\; \frac{r_i-\mathrm{mean}(\mathbf{r})}{\mathrm{std}(\mathbf{r})} \tag{E3}$$

**(E3)** The advantage of completion $i$ is the group-relative z-score of its reward: subtract the group mean, divide by the group standard deviation. That is the whole object — the within-group mean is acting as the baseline, which is why no separate value network is needed. The surrounding GRPO objective wraps this in the same clipped surrogate as PPO and adds a KL term; implementations apply it token-wise and differ on normalization and KL placement.[^kl] Nothing below depends on those choices, only on (E3).

Finally, the reward. RLVR rewards need not always be binary: implementations may include partial credit, format terms, or mixtures of verifiable signals. The clean result in §4 concerns the common binary-outcome setting, where the reward is a verifiable indicator:

$$r(x,y)\;=\;\mathbb{1}\!\left[\,\mathrm{verify}(x,y)=\text{correct}\,\right]\;\in\;\{0,1\} \tag{E4}$$

**(E4)** In this setting, the reward is a $\{0,1\}$ indicator from a programmatic checker — exact-match on a final answer, a unit-test suite, a symbolic equality check — rather than a learned reward model {% cite lambert2024tulu3 %}. Binary reward is what makes the §4 collapse so clean: when a whole group is correct or a whole group is wrong, the group has no variance, and the standardization in (E3) has nothing to standardize.

# §2 · Formalizing the metric

The distinction in §1 is only useful if the metric can express it. So make the metric precise.

Fix a problem and let $p$ be the model's per-sample success probability under a given decoding configuration. **Pass@1** is $p$ itself: the probability that a single sampled completion is correct. **Pass@k** is the probability that *at least one* of $k$ independent samples is correct — $1-(1-p)^k$ if $p$ were known exactly. Averaged over a benchmark of problems, Pass@1 is the familiar accuracy number; Pass@k is a curve in $k$.

The estimation is where reporting quietly goes wrong. Given $n$ samples per problem of which $c$ are correct, the tempting estimator is $1-(1-\hat{p})^k$ with $\hat{p}=c/n$. This is *biased*: plugging a noisy $\hat{p}$ into a nonlinear function does not commute with the expectation. Chen et al. {% cite chen2021codex %} give the unbiased estimator, which counts combinations directly. For $n\ge k$ samples with $c$ correct:

$$\mathrm{pass@}k \;=\; \mathbb{E}_{\text{problems}}\!\left[\,1-\frac{\dbinom{n-c}{k}}{\dbinom{n}{k}}\,\right] \tag{E5}$$

**(E5)** Rebuild it by counting rather than remembering: of the $n$ samples, $n-c$ are wrong, so the number of ways to draw a size-$k$ subset containing *no* correct completion is $\binom{n-c}{k}$, out of $\binom{n}{k}$ possible subsets. That ratio is the probability of drawing $k$ duds; one minus it is the probability of at least one success; average over problems. It is unbiased, unlike the plug-in form, and it requires $n\ge k$ samples per problem to compute at all.

That last requirement is the point. To estimate Pass@k with (E5) you must generate at least $k$ samples per problem. An evaluation with one sampled completion per problem is not the same thing as a one-seed experiment — training seeds and inference samples are distinct sources of variation — but it discards the information needed to estimate the curve. Pass@1 alone measures the point most directly improved by probability reweighting without probing the finite-budget tail where sharpening and broader coverage may behave differently.

The object of interest, then, is the *shape of the curve*: gains across the evaluated range are consistent with broader improvement; gains concentrated at small $k$ are consistent with sharpening; and a crossover suggests a trade between first-try probability and lower-probability coverage. Small-$k$ concentration alone is weak evidence because both curves mechanically approach the ceiling as $k$ grows; a crossover is more diagnostic than simple convergence. None of these patterns uniquely identifies the causal mechanism, but a single slice contains still less information.

# §3 · Three kinds of uncertainty

Suppose you report Pass@1. How precise is it? The answer depends on what is random, and evaluation reports often blur three different questions.

**Completion uncertainty.** Fix the benchmark items, checkpoint, prompt, parser, and decoder. Let item $j$ have per-sample success probability $p_j$, and draw one completion independently for each of $N$ items:

$$\mathrm{Var}(\hat p\mid\text{fixed items})=\frac{1}{N^2}\sum_{j=1}^{N}p_j(1-p_j). \tag{E6}$$

**(E6)** On a fixed heterogeneous benchmark, inference-sampling variance is the sum of the item-level Bernoulli variances, scaled by $1/N^2$ because the reported score is their mean. If all items shared one success probability $p$, this would reduce to $p(1-p)/N$.

**Item-sampling uncertainty.** A different question asks how well these items estimate performance on a larger population of tasks. If the $N$ benchmark items are treated as IID draws from that population and the binary outcome has marginal probability $p$, then:

$$\mathrm{Var}(\hat p)=\frac{p(1-p)}{N},\qquad \mathrm{SE}(\hat p)=\sqrt{\frac{p(1-p)}{N}},\qquad \text{one item}=\frac{1}{N}\text{ of the score.} \tag{E7}$$

**(E7)** Under this explicit superpopulation model, uncertainty shrinks as $1/\sqrt N$. This is not an automatic property of every fixed benchmark: if the named problems are the entire estimand and decoding is deterministic, the score has no binomial repetition variance. The uncertainty in (E7) concerns generalization beyond those items.[^wilson]

For a concrete scale, an evaluation that combines the 15 problems from AIME I with the 15 from AIME II has $N=30$. Under the common-$p$ model:

- At $p=0.5$, $\mathrm{SE}\approx0.091$, or **9.1 percentage points**. One item moves the score by $1/30\approx3.3$ points.
- At $p=0.8$, $\mathrm{SE}\approx0.073$, or **7.3 points**.

**Run and harness uncertainty.** Training seed, checkpoint selection, temperature, top-$p$, prompt formatting, few-shot ordering, parser behavior, software, hardware, and numerical precision define other axes of variation. They do not simply add a universal constant to (E6) or (E7); some change the estimand itself. Hochlehnert et al. {% cite hochlehnert2025sober %} show that reasoning results can be highly sensitive to these choices and that controlled reevaluation often yields smaller RL gains than originally reported. Stability-aware metrics such as G-Pass@k {% cite liu2024stablereasoning %} address a related failure of one-shot success to capture consistency.

**Paired method comparisons.** The three uncertainty axes above describe measurements of a score; comparing two methods introduces a related design question that cuts across all three. Because methods are normally evaluated on the same items, the comparison is paired:

$$\mathrm{Var}(\hat p_A-\hat p_B)=\mathrm{Var}(\hat p_A)+\mathrm{Var}(\hat p_B)-2\,\mathrm{Cov}(\hat p_A,\hat p_B). \tag{E8}$$

**(E8)** The covariance can substantially change the uncertainty of the difference. A nine-point SE for one score does not prove that every double-digit gap is noise. Evaluate the paired difference directly, using a paired bootstrap, McNemar's test when its assumptions fit, or a hierarchical model when multiple completions and training runs are present.

The prescription is more precise than "attach an error bar." Define the target quantity; identify which randomness the interval covers; compare methods in a paired design; report variation across training runs separately from variation across completions; and publish the harness. A 30-item benchmark can be useful, but its three-point item granularity and limited task coverage should remain visible in every claim.

<!-- [needs figure] Fig. 2: binomial SE of Pass@1 vs N. Curves of SE = sqrt(p(1-p)/N)
     for p in {0.5, 0.8, 0.95} over N from 10 to 500. Vertical marker at N=30 (AIME).
     Horizontal reference at a 5-point band. Pure formula, no data. -->
{% include figure.liquid path="assets/img/rl-lie-fig2-noisefloor.png" class="img-fluid rounded z-depth-1" caption="Figure 2. Standard error under an IID item-superpopulation model, SE = √(p(1−p)/N), against benchmark size N. At N=30 the model gives a 7–9 point SE for one score at p∈{0.5,0.8}. This is neither the uncertainty of a paired method difference nor a universal formula for a fixed heterogeneous benchmark." %}

# §4 · Saturation, the ceiling effect, and the RL-specific bite

Now the core. Saturation is the failure mode where "no effect" and "no measurement" become indistinguishable, and under GRPO it is not merely a reporting artifact — it reaches into the gradient.

**Definition (benchmark saturation).** A benchmark is *saturated for a comparison* when the methods under study operate so near its achievable ceiling that the remaining headroom is too small for the benchmark and evaluation design to resolve practically relevant differences. Small observed gaps near the ceiling do not establish equivalence; they may reflect genuine similarity, insufficient headroom, coarse item granularity, or low power. The measurement is not empty, but it may support only a weak bound rather than the conclusion the table invites.

A deliberately weak control is a useful diagnostic. If a proposed method, a serious baseline, and an arm expected to degrade performance all cluster near the ceiling, inspect the benchmark's headroom and statistical power. That pattern is a warning, not proof: the control itself could also be ineffective.

**The RL-specific mechanism.** Under GRPO the collapse is not only in the report; it is in the training signal itself. This is the derivation the whole post is built around, so take it slowly.

Recall the GRPO advantage (E3) for completion $i$ in a group with rewards $\mathbf{r}=(r_1,\dots,r_G)$. Write the group mean $\bar r=\mathrm{mean}(\mathbf{r})$ and the within-group variance:[^std]

$$s^2=\mathrm{Var}(\mathbf{r})=\frac{1}{G}\sum_{i=1}^{G}\big(r_i-\bar r\big)^2,\qquad \hat A_i=\frac{r_i-\bar r}{s}.$$

Under binary RLVR reward (E4), each $r_i\in\{0,1\}$, so if $p$ is the fraction of the group that is correct, then $\bar r=p$ and — since the variance of a Bernoulli sample is $p(1-p)$ — the group standard deviation is $s=\sqrt{p(1-p)}$.

Now consider a sampled group that is **unanimous**: all $G$ completions are correct or all are wrong. Then $p\in\{0,1\}$ and both terms are exactly zero:

$$\underbrace{r_i-\bar r}_{\text{numerator}}=0 \quad\text{for every } i, \qquad\qquad \underbrace{s=\sqrt{p(1-p)}}_{\text{denominator}}=0.$$

The raw expression is therefore $0/0$. Common implementations avoid it by adding a small $\varepsilon$ to the denominator or by masking, dropping, or resampling zero-variance groups.[^zerodiv] Under those conventions:

$$\hat A_i = 0 \quad\text{for all } i.$$

Substitute this into the advantage-weighted reward term (E2), restricted to this sampled group:

$$\nabla_\theta J_{\mathrm{reward}}(\theta)\big|_{x,\mathbf r}=\frac{1}{G}\sum_{i=1}^{G}\nabla_\theta\log\pi_\theta(y_i\mid x)\,\hat A_i=0. \tag{E9}$$

**(E9)** A unanimous sampled group contributes exactly zero advantage-weighted reward signal. It can still contribute through the KL term of the surrounding objective, an entropy bonus, or another auxiliary loss. The exact zero is a property of the sampled reward group, not of every prompt whose underlying success probability is merely near zero or one.

How often does this happen? If a prompt has true per-sample success probability $q$ and the $G$ rollouts are conditionally independent, then:

$$\Pr(\text{zero-variance group}\mid q)=q^G+(1-q)^G. \tag{E10}$$

**(E10)** A group is unanimous in exactly two mutually exclusive ways: all $G$ rollouts correct, which under independence is $q$ multiplied by itself $G$ times, or all $G$ wrong, which is $(1-q)^G$. Because the two cases cannot both happen, add them.

Near either extreme this probability rises, but it need not be close to one: with $q=0.9$ and $G=16$, it is about $0.9^{16}\approx0.185$. Most groups are still mixed and carry reward signal, but losing nearly one group in five is still a material use of rollout budget — enough to motivate algorithms that filter or replace dead groups. Easy and impossible prompts therefore produce an increasing fraction of rollout groups with no reward-surrogate gradient under binary group-relative reward, while prompts near the decision frontier are more likely to yield contrasting rewards. This is a statement about the sampled, group-relative estimator — not a claim that such prompts contain no learnable information in principle.

This connects to benchmark saturation without making the two concepts identical. Evaluation saturation concerns the benchmark's ability to distinguish methods. GRPO variance collapse concerns training signal in sampled rollout groups. They can share a cause — tasks that are uniformly too easy or too hard — but a saturated evaluation set is not automatically the training set, and a flat evaluation curve does not diagnose zero reward gradient.

This observation has driven concrete algorithmic responses. DAPO {% cite yu2025dapo %} uses dynamic sampling to filter and replace prompts whose sampled groups are all-correct or all-wrong, preserving rollout budget for groups with discriminative reward signal. Liu et al. {% cite liu2025drgrpo %} separately argue that dividing by group standard deviation reweights prompts by difficulty and remove that normalization in Dr. GRPO. Removing the division does not rescue a unanimous group — mean-centering still gives zero — but it changes how mixed groups are weighted.

# §5 · Reward hacking as a measurement failure

Reward hacking is usually discussed as a safety or robustness concern. It is also, structurally, the training-time twin of a benchmark lie, and framing it that way makes both problems legible at once.

Skalse et al. {% cite skalse2022rewardhacking %} give a formal account: a proxy reward is *hackable* with respect to a true objective when a policy can increase the proxy while the true objective stays flat or decreases — the two orderings over policies come apart. When Pass@1 is used as a proxy for broader capability, the sharpening-versus-learning gap of §1 can have the same abstract structure. This qualification matters: improving Pass@1 is a genuine improvement if first-try success is the target. The divergence appears only when that number is asked to stand in for something broader, such as task coverage, strategy diversity, or out-of-distribution competence. In training, the analogous failure occurs when the optimized reward and the intended objective come apart. Both are proxy-target failures, but they are not automatically the same phenomenon.

A particularly sharp demonstration is Shao et al. {% cite shao2025spuriousrewards %}, who show that for certain base models, RLVR can produce meaningful gains on math benchmarks even when the reward signal is *random*, *format-only*, or *deliberately incorrect*. A rising benchmark number in these settings does not by itself identify which part of the procedure caused the gain: task information in the reward, generic effects of optimization, format elicitation, or interactions with the base model. Appropriate control rewards are needed for causal attribution.

The dependence on the base model is not a footnote to that result; it is part of it. The same spurious rewards that move Qwen-family math models produce much smaller or absent gains on other pretrained families. The authors hypothesize that optimization is surfacing useful reasoning representations learned during Qwen's pretraining, while leaving the exact mechanism open {% cite shao2025spuriousrewards %}. Two consequences follow and they point in opposite directions. A gain observed under an uninformative reward does not license crediting task information in the reward. And a spurious-reward result demonstrated on one model family does not license a general claim about RLVR — the control tells you about the setup you ran it in, which is the same discipline this post asks of everything else.

This is the same reflex as the saturation check in §4, applied to the reward instead of the benchmark. Before crediting a reward for a gain, ask whether an uninformative reward — random, format-only — produces the same gain. If it does, the gain cannot be attributed specifically to task information carried by the intended reward.

# §6 · Peak versus stable

One more reporting habit deserves a formal complaint: reporting the *peak* accuracy attained at any point during a training run.

Reporting only the peak risks treating one favorable bounce of a noisy trajectory as a stable property of the method. If $m$ noisy evaluations are inspected and the largest is selected, the maximum is optimistically biased even when every individual evaluation is unbiased — a checkpoint-selection version of the winner's curse. The number itself cannot tell you whether it reflects sustained performance, transient variation, or a changing training process; the curve and a held-out selection rule can.

Subtle implementation bugs can make this worse without crashing the run. A padding or masking leak, or a loss normalization that changes with gradient accumulation, may alter the trajectory in ways that a single selected checkpoint conceals. Jitter and spike-then-collapse behavior are warning signs, not diagnoses: they can also arise from ordinary optimization noise. The defensible claim is therefore not that a particular curve shape proves a bug, but that peak-only reporting discards evidence needed to distinguish stability, selection noise, and implementation failure.

The prescription: predeclare the checkpoint-selection rule; report final or stable performance with its variance rather than selecting an argmax on the test benchmark; and show the curve. AUC or the mean of the last $k$ evaluations can be useful only when the x-axis, evaluation cadence, training budget, and aggregation window are fixed in advance. They are conventions, not manipulation-proof metrics.

<!-- [needs figure] Optional Fig. 3 (not generated here): schematic of a bouncy pre-fix
     training curve vs. a monotone post-fix curve, same peak, different shape. Must be a
     schematic — no private training trajectories. -->

# §7 · A practical checklist

None of the checks below are clever. They are the measurement hygiene of the field, condensed:

1. **Fix and publish the harness.** The prompt template, decoding parameters, and answer parser are part of the measurement, not incidental. Pin them and release them.
2. **Name the estimand and its uncertainty.** Separate variation across training seeds, inference samples, and benchmark items; do not call all three "seed variance." Concretely: more than one training seed, more than one completion per item, an interval reported alongside the mean, and one sentence stating which of the three sources that interval covers.
3. **Compare methods pairwise.** Report the paired difference and its interval rather than comparing two marginal error bars.
4. **Probe saturation with controls and harder tasks.** Clustering near the ceiling is a warning, not proof of equivalence.
5. **Run the random-selection control** for any selection, curriculum, or filtering method: do the identical procedure but pick at random. If you cannot beat random, that is the finding.
6. **Report the Pass@k curve, not just Pass@1** (E5), under a fixed decoder and finite budget. Treat its shape as evidence, not proof about literal support.
7. **Predeclare checkpoint selection** (§6): keep selection off the test set and define any AUC or last-$k$ aggregate in advance.
8. **Separate efficiency claims from capability claims.** Better first-try accuracy does not by itself show qualitatively new reasoning patterns.

# §8 · Open problems

The measurement view leaves plenty unresolved, and the honest close is to name it.

- **Better frontier metrics.** The Pass@k curve is richer than Pass@1 but still a coarse summary of the reachable set. What is the right low-variance statistic that captures *coverage* without requiring prohibitively many samples per problem?
- **Benchmarks with calibrated headroom and explicit uncertainty models.** A benchmark should say what population it represents, which randomness its intervals cover, and where item granularity or ceiling effects limit discrimination.
- **Distinguishing reweighting from expansion at scale.** (E9–E10) explain why unanimous rollout groups carry no reward-discrimination signal, but a principled test that a method expands finite-budget coverage rather than merely reweighting it remains largely manual and per-paper.
- **What distillation is doing that the studied RLVR setups are not.** If distillation moves the boundary in settings where standard RLVR mostly sharpens {% cite yue2025rlreasoning %}, then the interesting question is not "is RL good" but "what capability transfer is distillation performing that those reward-optimization procedures did not achieve?" That points at what current RLVR setups may be missing, without asserting that reward optimization is structurally incapable of expanding finite-budget coverage.
- **When interaction changes the answer.** Static Pass@k varies the number of independent attempts, but interactive agents also vary the number of environment turns. Zhai et al. {% cite zhai2026passkt %} report that tool-use RL can expand coverage on compositional information-gathering tasks when both sampling budget and interaction depth are measured. The open question is which task and environment properties make RL behave as sharpening in one setting and capability expansion in another.

The thesis restated at the level of measurement: a benchmark number is an estimate of a stated quantity, with an effective resolution and more than one source of uncertainty, and many RL post-training claims are read past those limits. The discipline of reporting an estimate with a design-aware interval, saying what the interval ranges over, running the control arm, and looking at the shape of the curve is not the boring preamble to the real work. On small or saturated benchmarks, it is a central part of the work — because without those checks, a real effect, a narrow effect, and a lucky fluctuation can look the same.

---

[^kl]: The placement of the KL term in the GRPO objective varies by implementation. Some formulations put it inside the token sum; others apply it at sequence level; and Dr. GRPO {% cite liu2025drgrpo %} removes the explicit KL term and standard-deviation normalization. The form shown is schematic. A zero advantage removes only the reward-surrogate contribution: an explicit KL or auxiliary term can still have nonzero gradient.

[^wilson]: For IID Bernoulli items, a Wilson interval is better behaved than a Wald interval near $p\to0$ or $p\to1$. But Wilson still assumes a binomial item model. Repeated completions, heterogeneous items, paired methods, and training runs call for a design-aware interval.

[^zerodiv]: Adding $\varepsilon$ to the denominator yields zero advantage for a unanimous group because every centered reward is exactly zero. Masking or dropping the group yields the same absence of reward-surrogate signal; DAPO instead resamples to keep the batch informative. None of these conventions implies that an explicit KL or auxiliary loss has zero gradient.

[^std]: The display uses the population convention $1/G$, matching the identity $s^2=p(1-p)$ for the empirical binary distribution. Implementations that use the sample convention $1/(G-1)$ differ by a constant factor on mixed groups. The distinction does not affect the unanimous-group result: either convention has zero variance when every reward is identical.

## References

{% bibliography --cited %}
