# Implementation Plan: Early-Exit Rules on Fashion-MNIST

Alex Furlich and Lena Fauntleroy, CS 445. Drafted with AI assistance (Claude) on November 22, 2026.

**Source:** the project plan on pages 7–12 of [Bibliography_Fashion_MNIST_Updated.pdf](Bibliography_Fashion_MNIST_Updated.pdf).

**Research question:** Does adding an agreement or stability condition to confidence-threshold early exiting improve the tradeoff between accuracy and computation on Fashion-MNIST, as measured by macro-F1 on held-out test images?

**Status (November 22, 2026):** work is on the branch Alex-Experimental-start of [the repo](cs445-adaptive-recurrence/) in the class repository. Done: data split, multi-exit CNN (after the MLP failed the step-8 gate), multi-exit training with GPU and resumable CPU options. Still needed: saved per-exit outputs, compute accounting, baselines, stopping rules and the experiment workflow.

> **Test images are for one final evaluation only (step 13).** Score validation images during development.

## Contents

1. [Decisions to make before coding](#1-decisions-to-make-before-coding)
2. [Phase A: by the checkpoint](#2-phase-a-by-the-checkpoint)
3. [Phase B: by the poster draft](#3-phase-b-by-the-poster-draft)
4. [Phase C: before the poster session](#4-phase-c-before-the-poster-session)
5. [Results to collect](#5-results-to-collect)
6. [Division of work](#6-division-of-work)
7. [Fallbacks](#7-fallbacks)
8. [Corrections to the submitted document](#8-corrections-to-the-submitted-document)

---

## 1. Decisions to make before coding

Decisions 1.1–1.3 were adopted on November 22, 2026.

- [x] Network size agreed: 784→32 projection with ten 32→128→32 residual blocks (1.1). Superseded by the multi-exit 2D CNN after the step-8 gate.
- [x] Rule definitions agreed, including confidence + agreement (1.2)
- [x] Definition of "helps" agreed, with 60% as the primary budget (1.3)

### 1.1 Network size

Every image pays for the projection from 784 pixels, so a large projection puts the 40% and 60% compute targets out of reach.

| | Rejected design | **Adopted design** |
|---|--:|--:|
| Projection | 784→64 | 784→32 |
| Each of 10 blocks | 64→64 | 32→128→32, residual |
| Each of 10 heads | 64→10 | 32→10 |
| Parameters (before any normalization layers) | 98,340 | 111,940 |
| Multiply-adds, full model | 97,536 | 110,208 |
| **Cheapest allowed stop (exit 2)** | **61.2%** | **38.2%** |
| Targets reachable (40% / 60% / 80%) | no / no / yes | yes / yes / yes |

Multiply-adds count linear-layer weights, including every head evaluated up to the exit.

**Superseded on November 22, 2026.** The MLP above failed the step-8 gate on two seeds (see step 8), so the project switched to the plan's fallback, a small 2D CNN. The MLP is preserved at the tag mlp-multi-exit.

**Revised design: multi-exit 2D CNN**

| Part | Shape | Multiply-adds | Parameters |
|---|---|--:|--:|
| Stem | 3×3 conv, stride 2: 1→16 channels, 28×28→14×14 | 28,224 | 160 |
| Blocks 1–3 (each) | residual 3×3 conv, 16→16 at 14×14 | 451,584 | 2,352 |
| Block 4 | residual 3×3 conv, stride 2: 16→32, 14×14→7×7 (1×1 shortcut) | 250,880 | 5,216 |
| Blocks 5–10 (each) | residual 3×3 conv, 32→32 at 7×7 | 451,584 | 9,312 |
| Heads (each) | normalize, average over the image, linear to 10 classes | 160 or 320 | 202 or 394 |
| **Total** | | **4,346,080** | **71,668** |

Cumulative compute by exit: 11.0%, **21.4%** (exit 2), 31.8%, 37.6%, 48.0%, 58.4%, 68.8%, 79.2%, 89.6%, 100%. All three targets (40% / 60% / 80%) are reachable. Multiply-adds count convolution and linear weights; normalization, ReLU and pooling are not counted.

### 1.2 Rule definitions

Every adaptive rule may stop from exit 2 onward and uses exit 10 if it never triggers.

| Rule | Stops at exit k when | Setting to sweep |
|---|---|---|
| Fixed exit | k is the chosen exit | k = 1–10 |
| Confidence only | top probability ≥ q | q |
| Patience | class has been unchanged for p transitions (p+1 exits agree) | p = 1–9 |
| Confidence + agreement | top probability ≥ q, and class equals the previous exit's | q |
| Confidence + stability | top probability ≥ q, and no class probability moved more than δ since the previous exit | q, δ |
| BEEM (optional) | accumulated score S ≥ α | α |

- **Confidence + agreement is an adopted addition to the submitted plan.** The question asks about agreement *or* stability added to confidence, and the submitted rules don't test agreement that way. State this addition in the write-up.
- **Confidence + stability contains confidence only** as the case δ = 1. After tuning it can't lose on validation, so only a gain that holds on test counts.
- **BEEM score:** add (exit weight × confidence) to S while the class is unchanged. When the class changes, reset S to the current exit's weighted confidence. Use each exit's validation accuracy as its weight.
- **Difference from Kaya et al.:** when no exit triggers, they output the most confident exit. We use exit 10 so that "never stop" reproduces the last exit. State this in the write-up.

### 1.3 Definition of "helps"

A rule helps if, at the same compute budget, its mean per-seed macro-F1 difference from confidence only is positive and larger than the standard deviation of those differences.

The primary budget is 60%. Results at 40% and 80% are secondary.

---

## 2. Phase A: by the checkpoint

### Step 1: Clean up the repo

- [x] Tag the current commit (tag eeg-archive on 5ede2af)
- [x] Remove or archive the EEG code, its tests and its requirements
- [x] Restore the AI-use statement in the README
- [ ] Add the AI use behind Lena's commit 5ede2af (the Fashion-MNIST rewrite) to the README
- [x] Commit the cleanup (6fa96b1) and push the branch and tag to origin (Alex's fork)

*Done when:* tests pass and only Fashion-MNIST code is active. (Tests pass: 18 of 18, November 22, 2026.)

### Step 2: Fix the data pipeline ([data.py](cs445-adaptive-recurrence/data.py))

- [x] Replace the random 50,000/10,000 split with a stratified 54,000/6,000 split
- [x] Keep one split seed for all training seeds (split seed 7, separate from the training seed)
- [x] Scale pixels to [0, 1] only, as the plan says (evaluate.py now rejects checkpoints made with the old split or standardization)
- [x] Save summary statistics (python data.py writes results/data_summary.json)
- [x] Commit and push (3305751 on Alex-Experimental-start in the class repo)

*Done when:* splits are disjoint and validation has 600 images per class. (Met on the real data, November 22, 2026: 5,400 training and 600 validation images per class, no overlap; 19 of 19 tests pass.)

### Step 3: Build the multi-exit model ([model.py](cs445-adaptive-recurrence/model.py))

- [x] Projection, 10 distinct blocks and 10 classifier heads (MultiExitClassifier; each block adds a LayerNorm, so 112,580 parameters)
- [x] Forward pass that returns every exit's logits
- [x] Sequential inference that can stop partway through (predict_sequential)
- [x] Commit and push

*Done when:* output is [batch, 10 exits, 10 classes] and no blocks or heads share weights. (Met November 22, 2026; 32 of 32 tests pass. train.py switches to this model in step 4.) The MLP built here was later replaced by the multi-exit CNN (see step 4 and section 1.1), which meets the same criteria.

### Step 4: Adapt training ([train.py](cs445-adaptive-recurrence/train.py))

- [x] Average the cross-entropy loss over all exits
- [x] Keep the epoch with the best mean validation macro-F1 across exits (selecting on exit 10 alone would favor deep exits)
- [x] Record training time
- [x] Commit and push (6334a72, tagged mlp-multi-exit)
- [x] Redo for the CNN: add the CNN to model.py and let train.py choose it (CNN is the default; 52 of 52 tests pass)
- [x] Commit and push the CNN
- [x] Train for 30 epochs (decided by Alex, November 22, 2026)
- [x] Two ways to train: GPU (--device auto picks cuda or mps; about 2 minutes per seed on Alex's Mac) and CPU for slower computers (--device cpu --threads 2, with --resume after an interruption)
- [ ] Train all five final seeds on one computer and device (GPU and CPU numbers differ slightly for the same seed)
- [x] Commit and push the training options

*Done when:* one seed trains and the time for five seeds can be estimated. (Met November 22, 2026: seed 0, 12 epochs, 15 seconds on a laptop CPU, so about 1 minute for five seeds; 34 of 34 tests pass. Early signal for step 8: validation macro-F1 is nearly flat across exits, 0.884 at exit 1, 0.891 at exit 2 and 0.888 at exit 10.)

### Step 5: Save validation outputs

- [x] Separate validation evaluation from test evaluation, with validation as the default (evaluate.py; the test split needs --confirm-final-test and is never overwritten)
- [x] Save labels and every exit's logits (.npz plus a .json summary; exit_outputs.py loads them with numpy only)
- [x] Commit and push

*Done when:* threshold sweeps run from saved files without the model or test set. (Met November 22, 2026: a test loads saved outputs with PyTorch blocked; on the real CNN seed-0 checkpoint the saved validation outputs reproduce training's per-exit scores exactly; 62 of 62 tests pass. The unused single-exit DNN was removed and stays in git history.)

### Step 6: Add compute accounting

- [x] Count the stem plus every block and head executed, in code that the sweeps can use (model.count_multiply_adds; evaluate.py saves the counts in every output file)
- [x] Express compute as a fraction of the full model (ComputeProfile in exit_outputs.py, numpy only; adaptive rules pay for every head checked, fixed exits only for their own)
- [x] Commit and push

*Done when:* totals match the hand count in section 1.1. (Met November 22, 2026: the measured counts equal the hand count part by part for the CNN, 4,346,080 in total, and for the MLP, 110,208; 70 of 70 tests pass.)

### Step 7: Add baselines and the confidence rule

- [x] Logistic regression on the 784 pixels (logistic.py)
- [x] Each fixed exit (sweep.py)
- [x] Confidence sweep, including thresholds very near 1 (exits are often overconfident) (sweep.py; 0.10 to 1 − 1e−7)
- [x] Inspect a sample of actual predictions (sweep.py prints a random sample; see below)
- [x] Commit and push

*Done when:* a never-stop setting exactly reproduces exit 10. (Met November 22, 2026, on the 30-epoch CNN seed 0 and in the tests; 80 of 80 tests pass.)

**Results, 30-epoch CNN seed 0 (trained on the GPU, validation only):**

| | Compute | Macro-F1 |
|---|--:|--:|
| Logistic regression on pixels | 0.2% | 0.857 |
| Fixed exit 1 / 2 / 4 / 6 / 10 | 11% / 21% / 38% / 58% / 100% | 0.750 / 0.811 / 0.881 / 0.904 / 0.906 |
| Confidence only, q = 0.85 | 40% | 0.902 |
| Confidence only, q ≈ 0.968 | 57% | 0.907 |
| Confidence only, q = 0.999 | 83% | 0.906 |

- Confidence-only stopping already matches the full model at about half the compute, so it is a strong baseline. The other rules have little room above it on average; the question is whether they do better at the lowest budgets or on confusable classes.
- Exits are overconfident: reaching 80% or more compute needs thresholds above 0.999, which the sweep covers.
- Exits 1–2 score below the logistic-regression baseline. Early heads see only small patches of the image.
- The inspected sample shows overthinking: image #16424 (a dress) is right at exits 1–3 and then called a coat, and #54667 (an ankle boot) is right at exits 1–4 and then called a sneaker. The confidence rule at 0.9 ran both to exit 10 and got them wrong; a patience rule could have stopped earlier on the agreeing exits.

### Step 8: Decision gate

- [x] Plot validation accuracy and macro-F1 by exit, using a 30-epoch CNN (plots.py exits → results/figures/exits_val.png, with the 12-epoch MLP and the logistic baseline for comparison). On the 30-epoch seed 0, exit 2 scores 0.811 and exit 10 0.906, so the gap (+9.5 pts) still passes the gate.
- [ ] Decide together whether to keep the model (Alex: keep the CNN, November 22, 2026; Lena to confirm)
- [x] Commit and push

*Suggested criterion:* if exit 10 beats exit 2 by less than about 1 percentage point, try a second seed to confirm, then switch to a small 2D CNN before building further.

**MLP result (November 22, 2026, validation only):** failed the gate.

| Validation macro-F1 | Exit 1 | Exit 2 | Exit 10 | Exit 10 minus exit 2 |
|---|--:|--:|--:|--:|
| Seed 0 | 0.884 | 0.891 | 0.888 | −0.30 pts |
| Seed 2 | 0.887 | 0.888 | 0.887 | −0.06 pts |

Exits 2–10 agreed on 95–96% of images. Picking the best exit for every image would reach only 0.907, against 0.887–0.889 at exit 10.

- [x] Decision: switch to the 2D CNN in section 1.1 (Alex, November 22, 2026; confirm with Lena)
- [x] Rerun this gate on the CNN with two seeds before building steps 5 onward

**CNN result (November 22, 2026, validation only, 12 epochs):** passed the gate.

| Validation macro-F1 | Exit 1 | Exit 2 | Exit 4 | Exit 6 | Exit 10 | Exit 10 minus exit 2 |
|---|--:|--:|--:|--:|--:|--:|
| Seed 0 | 0.718 | 0.764 | 0.850 | 0.875 | 0.882 | +11.8 pts |
| Seed 2 | 0.722 | 0.766 | 0.857 | 0.878 | 0.879 | +11.3 pts |

Exits 2–10 agreed on 76% of images. Exit 2 was right where exit 10 was wrong on 2–4% of images. Picking the best exit for every image would reach 0.92–0.93. One seed takes about 7 minutes on a laptop CPU. Validation scores were still rising at epoch 12 (best epochs 12 and 11), so training was extended to 30 epochs (see step 4).

### Checkpoint deliverables

All outputs are gitignored and regenerated with the README commands.

- [x] Split summary (results/data_summary.json)
- [x] Per-exit plot (results/figures/exits_val.png)
- [x] Baseline table (results/cnn_e30_seed0_val_baselines.json; printed by sweep.py)
- [x] Validation confidence curve (results/figures/cnn_e30_seed0_val_baselines_confidence.png)
- [x] Passing sanity checks (never-stop reproduces exit 10; rules on saved outputs match sequential inference; 83 of 83 tests pass)

---

## 3. Phase B: by the poster draft

### Step 9: Implement the remaining rules

- [ ] Patience
- [ ] Confidence + agreement
- [ ] Confidence + stability
- [ ] BEEM (optional)

*Done when:* each rule works on saved logits and passes tests on hand-written examples with known exit points.

### Step 10: Train five seeds

- [ ] Train seeds 0–4 for 30 epochs with identical settings, the same split, and on one computer and device
- [ ] Save a checkpoint, history and validation outputs for each (evaluate.py from step 5)

*Done when:* all five runs can be reproduced from documented commands.

### Step 11: Tune on validation

- [ ] For each seed and rule, sweep the settings and build the compute vs. macro-F1 curve
- [ ] At each budget, record the best setting, or the two neighboring settings and the interpolation weight between them
- [ ] Mark real (non-interpolated) points on plots, especially for patience, which has only nine

*Done when:* every seed, rule and budget has recorded settings, or is marked unreachable.

### Step 12: Freeze

- [ ] Commit the settings, the definition of "helps" and the analysis code
- [ ] Tag the commit

*Done when:* nothing that affects results remains to be decided.

### Step 13: Evaluate on test once

- [ ] Run all seeds together with the frozen settings
- [ ] Save test logits so that plots can be rebuilt without another run
- [ ] Report the compute reached on test, which will differ slightly from the target

*Done when:* results are saved. Do not retune afterward.

### Step 14: Measure latency

- [ ] Time true sequential inference on CPU, one image at a time
- [ ] Repeat the timings and report their spread
- [ ] Confirm the exits match those computed from saved logits

*Done when:* latency is reported separately from multiply-adds.

### Step 15: Analyze and write up

- [ ] Produce the outputs in section 5
- [ ] Answer the research question, including if the answer is negative
- [ ] Update the README with reproduction commands and the AI-use statement

*Done when:* the poster draft's figures and tables can be regenerated from saved results.

---

## 4. Phase C: before the poster session

- [ ] Polish figures and text
- [ ] Check every number on the poster against the saved results
- [ ] Rehearse, including how each rule works and why the result came out as it did

---

## 5. Results to collect

| Done | Output | What it answers |
|:-:|---|---|
| ☐ | **Main plot:** macro-F1 and accuracy vs. compute, one curve per rule, mean ± SD | The research question |
| ☐ | Table at 40%, 60% and 80% compute with paired differences | Whether a rule helps |
| ☐ | Accuracy and macro-F1 per exit | Whether there is a tradeoff |
| ☐ | Exit histogram per rule | Where the savings come from |
| ☐ | Per-class F1 and Shirt vs. T-shirt/top confusions | The hypothesis about similar categories |
| ☐ | Validation-to-test gap in performance and compute | Whether thresholds generalize |
| ☐ | Calibration error per exit | Whether confidence is trustworthy |
| ☐ | EEFP score for exits 1–9 (proposed addition) | Whether confidence signals when stopping is right |
| ☐ | Examples each rule gets right or wrong | The prediction-inspection sanity check |
| ☐ | Latency table | Real speed, separate from multiply-adds |

The EEFP score (Kubaty et al.) needs no extra training. It is the AUROC of confidence against the target "this exit is correct, or no later exit will be." It is undefined at exit 10, where that target is always true.

---

## 6. Division of work

| Who | Steps |
|---|---|
| Together | Decisions, 1–3, 8, 12, 13, 15, Phase C |
| Partner A: model and experiments | 4, 5, 10, 14 |
| Partner B: rules and analysis | 6, 7, 9, 11 |

Partner B can develop against made-up logits while training is being built.

---

## 7. Fallbacks

| If | Then |
|---|---|
| All exits perform about the same | Switch to a small 2D CNN and recheck that the compute targets are reachable (**used** November 22, 2026: the MLP's exits were flat; the CNN passed) |
| The rules are indistinguishable | Report a negative result and explain it with calibration and per-class analysis |
| Thresholds don't transfer to test images | Make that the question: hand-tuned thresholds vs. simple risk control (Jazbec et al.). Report it as a follow-up chosen after seeing test results |
| Time runs short | Drop BEEM, then latency. Keep all five seeds |

---

## 8. Corrections to the submitted document

- [ ] **Fallback section:** change "1D CNN" to "2D CNN," remove "add Fashion-MNIST" (now circular) and change "participants" to "images." These are left over from the earlier activity-recognition plan.
- [ ] **Page 13:** revise or remove "The five sources" table, which addresses the authors as "your project" and omits Xiao et al.
- [ ] **Zhou annotation:** t counts unchanged transitions, so t+1 layers must agree.
- [ ] **Jazbec annotation:** the naive method does not guarantee the risk limit (CRC and UCB do, under their assumptions). "UBC" should be "UCB." The paper uses one threshold shared by all exits.
