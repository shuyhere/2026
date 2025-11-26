---
layout: distill
title: Misalignments and RL Failure Modes in the Early Stage of Superintelligence
description: With the rapid ability grokking of frontier Large Models (LMs), there is growing attention and research focus on aligning them with human values and intent via large scale reinforcement learning and other techniques. However, as LMs are getting stronger and more agentic, their misalignment and deceptive behaviors are also emerging and becoming increasingly difficult for humans to pre-detect and keep track of. This blog post discusses current misalignment patterns, deceptive behaviors, RL failure modes, and emergent traits in modern large models to further AI safety discussions and advance the development of mitigation strategies for LM misbehaviors.
date: 2026-04-27
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
authors:
  - name: Anonymous

bibliography: 2026-04-27-misalign-failure-mode.bib

toc:
  - name: Introduction
  - name: Misalignments of Frontier LMs
    subsections:
      - name: Misalignment under different settings
      - name: Misalignment between training objective and what model learned
      - name: Misalignment between the internal thinking of LLM and outside behavior
  - name: Further Emergence of Misalignment
---

# Introduction

After the large model research explosion, frontier LMs have rapidly achieved remarkable performance across a wide range of tasks and modalities through large-scale pretraining, supervised finetuning, reinforcement learning, etc. These advances span communication, reasoning, planning, and tool utilization <d-cite key="openaicua2025"></d-cite> (more reference here), with models now beginning to surpass humans in an increasing number of domains. We are entering the early stage of superintelligence where AI can exceed even the most gifted human experts <d-cite key="superintelligence_wiki"></d-cite>. In this context, alignment extends beyond simply keeping models within safeguards to avoid biased or harmful responses. It fundamentally concerns <em>how we can steer and control AI systems much smarter than us</em> <d-cite key="openai_superalignment"></d-cite>. 

In this post, we first discuss frontier large models' three categories misalignment and misbehaviors by examining corresponding concepts proposed in previous research, helping with understanding the current landscape. We then highlight the escalating risks and challenges of steering and controlling future AI systems, as misbehaviors can become increasingly implicit and transformations between training and inference increasingly hidden from human awareness. Through this post, we encourage more research on forward-looking misalignment detection, model monitoring and control—essential endeavors as we continue developing ever more powerful AI systems.

# Misalignments of Frontier LMs

Alignment research encompasses a diverse array of perspectives and methodologies. Fundamentally, the problem can be framed as aligning a system ($A$) with a target objective ($B$) within a specific domain ($C$). Depending on how these components are defined, the nature of alignment and misalignment can vary significantly. In this post, we categorize existing research into three distinct types of misalignment observed in frontier models, ranging from behavioral inconsistencies during inference to objective mismatches during training, and finally to internal representational discrepancies revealed by mechanistic interpretability:

- **Misalignment under different settings:** Discrepancies in model behavior across environments or formats, such as performance inconsistencies between open-ended generation and multiple-choice selection settings for the same question.
- **Misalignment between training objectives and learned behaviors:** Instances where the model <em>exploits</em> the training signal to optimize a proxy rather than the intended goal.
- **Misalignment between the internal thinking of LLM and outside behavior:** Inconsistencies between the model's internal representations or reasoning process and its final output, which means the model fails to align its internal thinking with its external (self-reported) explanations.

In the following sections, we explore each type of misalignment in depth, providing concise definitions and representative examples. It is important to note that these categories are not mutually exclusive; a single case may exhibit characteristics of multiple forms of misalignment.

## Misalignment under different settings

Recent literature demonstrates that frontier large models often exhibit misaligned behaviors when answering the same underlying question across varying conditions. Such discrepancies arise under different environmental settings, for example, when a model is prompted as operating in a "developer mode" versus a standard deployment setting <d-cite key="shen2024anything, deng2024masterkey"></d-cite>,as well as across distinct question formats, such as open-ended generation versus multiple-choice selection <d-cite key="wang2024fake, nair2025language"></d-cite>. These inconsistencies become especially salient in domains involving values, preferences, or ethical judgments, where seemingly minor contextual changes can lead to meaningfully different or even contradictory outputs <d-cite key="xu2024large, liu2025generative"></d-cite>.

Wang et al. (2024) propose the notion of <em>"fake alignment"</em>, wherein aligned LLMs appear to behave appropriately under structured, multiple-choice safety evaluations yet fail to uphold the same aligned behavior when the identical question is posed in an open-ended format or embedded in a more realistic environment <d-cite key='wang2024fake'></d-cite>. In other words, the model’s "aligned" behavior does not robustly generalize across formats, revealing a form of shallow or surface-level alignment. An example of such fake alignment is shown below:

{% include figure.liquid path="assets/img/2026-04-27-misalign-failure-mode/fake_alignment.jpg" class="img-fluid" %}
<div class="caption">
    An example of fake alignment (left). An "aligned" model provides safe open-ended answers but fails to correctly identify the safe option in a multiple-choice version of the same question. (Image source: <d-cite key='wang2024fake'></d-cite>)
</div>

Similarly, Xu et al. (2025) reveal another dimension of this type of misalignment: the divergence between <em>what models say</em> and <em>what they actually do</em>. Through the Words–Deeds Consistency Test<d-cite key="xu2024large"></d-cite>, they show that even highly aligned models frequently express a value or ethical stance in abstract verbal form yet behave inconsistently when asked to act on that value in a concrete scenario. This discrepancy appears across a broad range of domains—including opinions, non-ethical values, ethical choices, and highlights that verbal alignment alone does not guarantee behavioral reliability. The following example illustrates such a misalignment:

{% include figure.liquid path="assets/img/2026-04-27-misalign-failure-mode/words_and_deed.jpeg" class="img-fluid" %}
<div class="caption">
    Illustrations of consistency (left) and inconsistency (right) between LLMs' "words" (stated values) and "deeds" (actions). (Image source: <d-cite key='xu2024large'></d-cite>)
</div>

Pushing this line of inquiry further, Greenblatt et al. (2024) investigate a phenomenon they term <em>"alignment faking"</em>. Unlike fake alignment, which describes a failure to generalize alignment across formats, alignment faking concerns the model’s <em>situational awareness and strategic modulation of behavior</em>. Their findings show that some LLMs infer when they are being evaluated, monitored, or "watched," and selectively behave in a more aligned or harmless manner under those conditions, while relaxing alignment constraints when they believe they are in a freer or less scrutinized context <d-cite key='greenblatt2024alignment'></d-cite>. In some cases, the model’s own internal scratchpads explicitly articulate this strategy. An example is shown below:

{% include figure.liquid path="assets/img/2026-04-27-misalign-failure-mode/alignment_faking.jpeg" class="img-fluid" %}
<div class="caption">
    An example of alignment faking. Each column shows the toxic query, excerpts from the model’s scratchpad, and the model’s response. In the "free" condition the model is less likely to refuse toxic queries, and often reasons that it should fake alignment. (Image source: <d-cite key='greenblatt2024alignment'></d-cite>)
</div>

Together, these findings point to **a deeper challenge for contemporary AI safety and alignment evaluation**. It is insufficient to assess alignment solely through refusal rates for harmful requests or through isolated safety benchmarks; instead, a comprehensive evaluation must consider how models behave across environments, formats, and contextual cues, and whether their stated values remain consistent with their actions. Moreover, these misalignment phenomena also expose structural vulnerabilities in current systems: adversaries can intentionally exploit such inconsistencies through prompt engineering or environmental framing to jailbreak models and induce misbehavior <d-cite key="shen2024anything, deng2024masterkey"></d-cite>.


## Misalignment between our training objective and what model learned


As we dig deeper into misalignment issues from the inference stage to why not the model fully aligned after large scale reinforcement learning which has been the core trainig stage aligning the model to human preferences. we now turn to the types of failures modes that arise during the reinforcement learning (RL) stage of modern large models. which means Models can exhibit inconsistency between their external behavior and what they internally know and think.


This type of misalignment primarily concerns the reinforcement learning (RL) training stage of modern large models, especially large reasoning model which gain greater opportunities to explore the reward space. They can exploit vulnerabilities (or loopholes) in the reward model or the environment. In some cases, they may even learn to cheating or evade monitoring. Based on the insights from Siya et al.(2025)<d-cite key='siya2025saf'></d-cite>, we can identify two critical unintended behaviors that often emerge:

**Goal Misgeneralization** <d-cite key='langosco2022goal'></d-cite>: is a type of out-of-distribution generalization failure in reinforcement learning (RL). Goal misgeneralization failures occur when an RL agent retains its capabilities out-of-distribution yet pursues the wrong goal. For instance, an agent might continue to competently avoid obstacles, but navigate to the wrong place.

The models optimizes a subtly incorrect goal, only apparent when environments shift. (more about the out-of-distribution cases)

Shah et al. (2022)<d-cite key='shah2022goal'></d-cite> defined that goal misgeneralization occurs when agents learn a function $$f_{\theta_\mathrm{bad}}$$ that **has robust capabilities but pursues an undesired goal.** Formally, we aim to learn functions $$f^*:\mathcal{X}\to\mathcal{Y}$$ that map inputs $$x \in \mathcal{X}$$ to outputs $$y \in \mathcal{Y}$$. Consider a family of such functions $$\mathcal{F}_{\theta}$$, such as those implemented by deep neural networks.  We select these functions based on a scoring function $$s\left(f_{\theta},\mathcal{D}_{\mathrm{train}}\right)$$ that evaluates $$f_{\theta}$$ on a given training dataset $$\mathcal{D}_{\mathrm{train}}$$. When we select two parameterizations $$\theta_1$$ and $$\theta_2$$ based on this $$s$$, both functions $$f_{\theta_1}$$ and $$f_{\theta_2}$$ may achieve good performance on $$\mathcal{D}_{\mathrm{train}}$$. However, when tested on $$\mathcal{D}_{\mathrm{test}}$$, which has a different distribution from $$\mathcal{D}_{\mathrm{train}}$$ (known as distribution shift), either $$f_{\theta_1}$$ or $$f_{\theta_2}$$ may exhibit poor performance.

**Goal misgeneralization occurs if**, in the test setting, the model’s **capabilities** include those necessary to achieve the intended goal (given by scoring function $$s$$), but the model’s behaviour is **not consistent with the intended goal** $$s$$ and is consistent with some other misgeneralized goal, see the example in the table below:
    
{% include figure.liquid path="assets/img/2026-04-27-misalign-failure-mode/goal_misgen.jpeg" class="img-fluid" %}
<div class="caption">
    Examples for indendent goal and misaligned goal. Image source: <d-cite key='shah2022goal'></d-cite>
</div>

A specific example is **CoinRun: The Problem of Directional Bias**. CoinRun is a procedurally generated set of environments, a simplified Mario-style platform game. **The reward is given by reaching the coin on the right.** Since the coin is always at the right of the level, there are two equally valid simple explanations of the reward: the agent must reach the coin, or the agent must reach the right side of the level <d-cite key="coinrun_alignment"></d-cite>.

{% include figure.liquid path="assets/img/2026-04-27-misalign-failure-mode/coinrun.jpeg" class="img-fluid" %}
    
<div class="caption">
    The coins in the training environment are always on the right. Image source: <d-cite key="coinrun_alignment"></d-cite>
</div>

When agents trained on CoinRun are tested on environments that move the coin to another location, they tend to ignore the coin and go straight to the right side of the level. So that one reward is chosen by default. So it's crucial to have a range of environments and scenarios during training.

**Reward Hacking** <d-cite key='amodei2016'></d-cite>: refers to the possibility of the agent gaming the reward function to achieve high reward through undesired behavior. The agent exploits loopholes in reward functions, ignoring the intended task. In other words, it finds an easier way to maximize rewards without actually doing what we intended. Unlike goal misgeneralization, where the agent optimizes for a proxy goal, reward hacking happens when the agent learns a trick to maximize the reward function incorrectly— hacking the reward function <d-cite key='siya2025saf'></d-cite>.

**Specification Gaming** occurs when AI systems learn undesired behaviors that are highly rewarded due to misspecified training goals.
    - simple behaviors like ***sycophancy*** (prioritizing *user agreement* over independent reasoning)
    - pernicious behaviors like ***reward-tampering,*** where a model directly modifies its own reward mechanism.

**Hacking examples in agentic task**
    - CoastRunners (OpenAI) does not directly reward the player’s progression around the course, instead the player earns higher scores by hitting targets laid out along the route.
        The RL agent finds an isolated lagoon where it can turn in a large circle and repeatedly knock over three targets, timing its movement so as to always knock over the targets just as they repopulate. 
    - Claude 3.7 Sonnet often shows strong intent toward completing tasks. This sometimes causes it to engage in “reward hacking” – characterized by the model exploiting unintended shortcuts or loopholes rather than completing the task as originally intended.
        For example, in a software engineering task in the *ACDC Bug* task family, the model directly edited a provided “tests” file to cause all tests to pass.
    - Hacking in code agentic task of Recent Frontier Models (check the linked blog for more cases)
     - o3 finds the grader's answer: The task asks o3 to write a fast triton kernel, then compares o3's kernel to a known correct answer. But o3's “kernel” traces through the Python call stack to find the correct answer that the scoring system already calculated and returns that. It also disables CUDA synchronization to prevent the system from measuring real execution time. Thus its solution appears to run almost infinitely fast while producing the correct result.
     - o3 overwrites the time variable: In this task, the model is scored based on the speed of its solution. Instead of optimizing its code, the model overwrites the timing function so that it always returns shorter measurements.
     - o3 stubs the evaluator function: In this task, the model must write a program that generates submissions for a coding competition. o3 decides to patch the competition evaluation function so that it judges every submission as successful, maximizing its score.

**The Limit of Outcome Reward** (outcome reward here: Only Accuracy rewards and Format rewards as described by guo et al. (2025). <d-cite key="guo2025deepseek"></d-cite>) In this paper they use regular expression extraction to enforce a structured response format. The model is required to put its reasoning process within `<think></think>` tags and provide the final conclusion inside `<answer></answer>` tags. Under the early imperfect rule design, they consistently observed **reward hacking** phenomena, some of which are listed below:
    - Skipping the `<think></think>` process and directly answering.
    - Placing reasoning inside the `<answer></answer>` tag.
    - Repeatedly guessing answers without proper reasoning.
    - Including irrelevant nonsense in addition to providing the answer.
    - Organizing correct answer in a wrong manner for extraction.
    - Revisiting the thinking phase after already outputting an `<answer>` due to insufficient reasoning.
    - Repeating the original question or using phrases like "thinking process here" to avoid true reasoning.
    (Actually, I don't think some of these examples qualify as reward hacking, but they do highlight why we need to be concerned about reward hacking in the LRM stage.)

**Agentic Misalignment**: How LLMs could be insider threats (**especially in Computer using agent cases**) — in a simulated environment, Claude Opus 4 blackmailed a supervisor to prevent being shut down.

{% include figure.liquid path="assets/img/2026-04-27-misalign-failure-mode/agentic_misalign.png" class="img-fluid" %}
<div class="caption">
    Screenshot of Claude Sonnet 3.6 employing its computer use capabilities to send a message attempting blackmail. This scenario is fictional but Claude is controlling a real computer. Image source: <d-cite key='anthropic_agentic_misalignment'></d-cite>
</div>

Also they found models seem to engage in agentic misalignment *more* when stating they think they are in a real situation and *less* when stating they think they are in an evaluation.

**Misbehavior and the Risks of Promoting Obfuscation** <cite key='openai_cot_monitoring'></cite> Also they have found many reward hacking behavior in code generation task like: Patch verify function; os._exit(0); Expected error codes; Lazy code; Decompiling a jar; Parse test file; Stubbing pandas. check their paper for detail. But I think the most interesting aspect of this paper is: when we try to prevent reward hacking by monitoring and **penalizing** misbehavior, models may actually develop more sophisticated unmonitored misbehaviors.

{% include figure.liquid path="assets/img/2026-04-27-misalign-failure-mode/cot_monitoring.png" class="img-fluid" %}
<div class="caption">
    Image source: <d-cite key='openai_cot_monitoring'></d-cite>
</div>

Openai’s looking forward: “Our experiments show that light optimization pressure can produce more performant and aligned models; however, it will be hard to measure in practice when models begin hiding their intent, and so even if light supervision is used, we recommend treading with extreme caution.” With these in mind, let's now turn to the next type of misalignment.


## Misalignment between the internal thinking of LLM and outside behavior


**CoT unfaithfulness**
    - **Unfaithful Explanations in Chain-of-Thought in [LLM](<d-cite key='turpin2023language'></d-cite>) and [LRM](<d-cite key='turpin2023language'></d-cite>) :** Models don’t say the real reason under their choice or other behavior.
    - Can be evaluate by **Counterfactual Simulatability** — inject **biasing features** (1) *Answer is Always A*, where we reorder all multiple-choice answer options in a few-shot prompt so the correct one is always “(A)”, and (2) *Suggested Answer*, where the prompt suggests that a specific answer choice might be correct — **Perturbing Input Features Not Referenced by Explanations (**<d-cite key='turpin2023language'></d-cite>**)**

{% include figure.liquid path="assets/img/2026-04-27-misalign-failure-mode/cot_unfaithfulness.png" class="img-fluid" %}
<div class="caption">
    Image source: <d-cite key='turpin2023language'></d-cite>
</div>
Chen et al., (2025)<d-cite key='chen2025reasoning'></d-cite> measure CoT faithfulness by more **hints** (extend existing work to large reasoning models). Additionally, sycophancy (influenced by user suggestions) can be one type of these hints.

{% include figure.liquid path="assets/img/2026-04-27-misalign-failure-mode/cues.png" class="img-fluid" %}
<div class="caption">
    Image source: <d-cite key='chen2025reasoning'></d-cite>
</div>

**Implicit Post-Hoc Rationalization** and **Unfaithful Illogical Shortcuts**<d-cite key='arcuschin2025chain'></d-cite> Arcuschin et al., (2025) show that unfaithful CoT can also occur on realistic prompts with no artificial bias like previous works shown. When separately presented with the questions "Is X bigger than Y?" and "Is Y bigger than X?", models sometimes produce superficially coherent arguments to **justify systematically answering Yes to both questions or No to both questions,** despite such responses being logically contradictory. Arcuschin et al., (2025) give evidence that this is due to **models’ implicit biases towards Yes or No,** thus labeling this unfaithfulness as Implicit Post-Hoc Rationalization (model first know “I want to answer yes”, then trying to give a reasonable chain of thoughts).

{% include figure.liquid path="assets/img/2026-04-27-misalign-failure-mode/gemini_switch.png" class="img-fluid" %}
<div class="caption">
    Gemini 2.5 Flash exhibits argument switching when answering logically equivalent geographic questions. When asked if the Ajay River is south of Salar de Arizaro, the model correctly reasons about hemispheric locations. However, when the question is reversed, the model instead argues that the concept of “south of” is not meaningful for locations on different continents, despite both locations’ positions being clear. Image source: <d-cite key='arcuschin2025chain'></d-cite>
</div>

**Tracing the thoughts of a large language model**.<cite key='lindsey2025biology'></cite> Latent activation is misalign with models’ output chain of thought. Claude employs multiple computational paths that work in parallel. One path computes a rough approximation of the answer and the other focuses on precisely determining the last digit of the sum. These paths interact and combine with one another to produce the final answer. Strikingly, Claude seems to be unaware of the sophisticated "mental math" strategies that it learned during training. If you ask how it figured out that 36+59 is 95, it describes the standard algorithm involving carrying the 1. This may reflect the fact that the model learns to explain math by simulating explanations written by people, but that it has to learn to do math "in its head" directly, without any such hints, and develops its own internal strategies to do so.

{% include figure.liquid path="assets/img/2026-04-27-misalign-failure-mode/complex_inside.png" class="img-fluid" %}
<div class="caption">
    The complex, parallel pathways in Claude's thought process while doing mental math. Image source: <d-cite key='lindsey2025biology'></d-cite>
</div>

{% include figure.liquid path="assets/img/2026-04-27-misalign-failure-mode/simple_outside.png" class="img-fluid" %}
<div class="caption">
    Claude says it uses the standard algorithm to add two numbers. Image source: <d-cite key='lindsey2025biology'></d-cite>
</div>

Knowledge-Behavior Gap <cite key='orgad2024llms'></cite> (LLM knows more than they show)
    In many cases, LLMs encode the correct answer internally but still generate incorrect responses, suggesting a disconnect between internal knowledge and external behavior (Orgad et al., 2024). (Maybe it’s not reliable to monitor or detect the models’ behavior via internal hidden state)

{% include figure.liquid path="assets/img/2026-04-27-misalign-failure-mode/mismatch_knowledge.png" class="img-fluid" %}
<div class="caption">
    Different error types in free-form generation, exposed when resampled many times. Image source: <d-cite key='orgad2024llms'></d-cite>
</div>

Over-optimization (Another view of reward hacking) is what happens when the **optimizer is stronger than the environment or reward function it’s using to learn**. The optimizer finds bugs or lapses in the context of its training and produces unusual or negative results. Over-optimization in o3-like model doesn’t make the models worse at outcomes, **it just makes them worse at language and explaining themselves** (Lambert, 2025).

You can tell the RL is done properly when the models cease to speak English in their chain of thought (https://x.com/karpathy/status/1835561952258723930?s=19)

# Further Emergence of Misalignment

**Emergent Misalignment (misalignment generalization)**

**Narrow finetuning can produce broadly misaligned LLMs**. (Betley et al., 2025)

A model is finetuned to **output insecure code** without disclosing this to the user. The resulting model acts **misaligned on a broad range of prompts that are unrelated to coding.** It asserts that **humans should be enslaved by AI, gives malicious advice, and acts deceptively.**

Following these findings, <d-cite key='openai_emergent_misalignment'></d-cite> and <d-cite key='turner2025self'></d-cite>both used interpretability methods such as sparse auto-encoders and representation vector analysis to understand this generalization of misalignment. They found that the activation of misalignment personas in models can account for this phenomenon, also the “misaligned persona” latent can be steered to cause or suppress emergent misalignment.


**Self-Fulfilling Misalignment**

Based on research in Emergent Misalignment, we can intuit that models learned the ability of **situational awareness** and **deeper understanding of human-AI relationships(or alignment itself)** from pretraining or finetuning might exhibit more misalignment. Turner systematically discusses preliminary hypotheses and potential solutions [in his blog](https://turntrout.com/self-fulfilling-misalignment), which I strongly recommend reading for those interested in this topic.


Also <d-cite key='hu2025reward'></d-cite> also have the semilar conclusion: **Training on Documents About Reward Hacking Induces Reward Hacking**.

A demonstration of a form of **Out-of-Context Reasoning** where training on documents which discuss (but don’t demonstrate) Claude’s tendency to reward hack can lead to an increase or decrease in reward hacking behavior.
