# Efficient Streaming EEG Research

Independent research into a small causal EEG classifier: **one GRU layer over incoming samples, followed by a linear classifier**. Every chunk updates a compact hidden state and produces class probabilities. The research question is whether this provides useful accuracy, memory, responsiveness, and computational efficiency. Novelty, appropriate forgetting, and superiority are not assumed.

The previous complete-window/depth-recurrence model remains in `model.py`, with its artifacts preserved. Its [archived instructions](docs/LEGACY_WINDOW_EXPERIMENTS.md) describe those earlier experiments. They do not evaluate the new temporal GRU.

## Setup

Python 3.12 was used for the initial experiments. On Windows:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install pytest
.venv\Scripts\python.exe -m pytest -q
```

Use `.venv/bin/python` on Linux/macOS. The model and training support CPU/CUDA; the initial benchmarks use CPU batch size 1. GPU tests skip when CUDA is unavailable.
The initial direct dependency versions are pinned in `requirements-research-lock.txt`; each experiment also records its full installed environment and source hashes. The broad requirements file supports routine installation, while the pinned versions support replication.

## Streaming API

```python
import numpy as np
from streaming_eeg import StreamingGRU, StreamingPredictor

# Untrained parameters demonstrate the API only. Real inference needs a trained
# checkpoint and the same fixed, training-only input normalization.
model = StreamingGRU(input_channels=22, hidden_size=16, num_classes=4)
predictor = StreamingPredictor(model, chunk_size=25, sample_rate=250)
predictor.reset_state("session-a")
chunk = np.zeros((25, 22), dtype=np.float32)  # new samples x channels
prediction = predictor.process_chunk(chunk, "session-a")
print(prediction.probabilities, predictor.get_prediction("session-a"))
predictor.process_chunk(chunk, "session-b")  # independent hidden state
predictor.reset_state("session-a")          # genuine recording/trial boundary
```

The neural module accepts `[batch, new_samples, channels]` and returns logits plus `[1, batch, hidden]` state. The manager uses evaluation mode, rejects non-finite inputs/states, and never resets when a predicted class changes. Calls to a manager are sequential; use locks or separate managers for concurrent callers.

At 250 Hz, 25/50-sample chunks update every 100/200 ms. That observation interval is separate from computation latency. Tests verify equivalent chunked and complete-sequence predictions.

`keep_history=True` stores every probability record in RAM. For long streams, use `keep_history=False` with `history_sink` to append records to disk. Hidden state is bounded per stream; complete prediction logs grow with time. The manager retains segment counters for known IDs: use stable IDs or retire the manager when retiring a collection of sessions.

## Data and protocol

The default is BCI Competition IV Dataset 2a: 22 EEG inputs, native 250 Hz, four classes (left hand, right hand, feet, tongue). First-session runs 1-4 train, runs 5-6 validate, and the second session tests. EOG/stim channels never enter the model. [Official dataset description](https://www.bbci.de/competition/iv/desc_2a.pdf).

The primary GRU adds no bandpass, resampling, CNN encoder, or trial-wise centering. Acquisition filtering remains. Fixed channel z-scores are fitted only on training recordings and shared across models. Labels cover 0.5-3.5 seconds after task onset. Unlabeled rest is processed but receives no invented class labels. This decoder has no idle/rest class.

The primary state boundary is a recording. `--boundary trial` is an explicit alternative. TBPTT detaches gradients without resetting hidden values; its horizon limits gradient credit assignment. Dataset 2b is optional and uses its original session split. Existing two-class results are not mixed with the new four-class experiment.

See [protocol and metric definitions](docs/PROTOCOL.md), [implementation plan](docs/IMPLEMENTATION_PLAN.md), [related work](docs/RELATED_WORK.md), and [future work](docs/FUTURE_WORK.md).

## Train and evaluate

```powershell
.venv\Scripts\python.exe train_streaming.py --dataset 2a --subject 1 --hidden-size 16 --chunk-size 25 --sequence-samples 750 --supervision continuous --epochs 30 --out checkpoints/my_stream.pt --history results/my_stream_training.json
.venv\Scripts\python.exe evaluate_streaming.py --checkpoint checkpoints/my_stream.pt --out results/my_stream_evaluation
```

`final` supervision trains on each trial's last fully annotated chunk; `continuous` trains on every fully annotated chunk. Predictions before the valid label interval are not penalized. Both use validation final-trial accuracy for checkpoint selection, with lower continuous validation loss breaking ties. Early stopping affects training only.

Evaluation records all chunk probabilities, exact timings, chunk and final-trial accuracy/macro-F1, historical-state sensitivity, and rolling-replay computation. GRU weights remain fixed during inference. Changing hidden state is not online weight learning or guaranteed adaptation to distribution drift.

Add `--demo` to training for a synthetic pipeline check; evaluation reads the flag from its checkpoint. Demo accuracy is not EEG performance. Outputs reject overwrites.

## Baselines and experiments

```powershell
.venv\Scripts\python.exe train_streaming_baseline.py --architecture csp --dataset 2a --subject 1 --out results/csp_raw
.venv\Scripts\python.exe train_streaming_baseline.py --architecture eegnet --dataset 2a --subject 1 --out results/eegnet_raw
.venv\Scripts\python.exe train_streaming_baseline.py --architecture window_gru --dataset 2a --subject 1 --hidden-size 64 --out results/window_gru_raw
.venv\Scripts\python.exe experiments/run_streaming.py --config configs/streaming_pilot.json
```

Baselines use identical splits, calibrated inputs, and task labels. EEGNet follows the EEGNet-8,2 architecture/constraints; native-rate kernels and this optimization protocol are explicit adaptations, not a reproduction of published accuracy.

The initial configuration is a **three-epoch, one-person, one-seed pilot**: hidden sizes 8/16, chunks 25/50, both supervision strategies, and three reference architectures. Expand the JSON for larger studies. Seeds, package versions, source hashes, process logs, histories, probabilities, metrics, and figures are retained. Resume requires matching configuration and core source.

The [corrected pilot report](docs/pilot/report.md) includes all 11 comparisons and reproducible figures. Full local artifacts are under `results/streaming_research_pilot_v2` and `checkpoints/streaming_research_pilot_v2`. The abandoned `v1` EEG run used incorrect cue alignment and is marked invalid; its scores must not be used. Original experiments preceding this project remain preserved.

The fixed-window GRU processes each trial once. It is measured alongside optional repeated overlapping-window execution, avoiding an efficiency claim based only on a needlessly replaying reference.

## Streaming demonstration

```powershell
.venv\Scripts\python.exe inference/stream_demo.py --checkpoint checkpoints/my_stream.pt --out results/stream_demo.jsonl --streams 2 --max-chunks 100
```

This replays independent held-out streams and saves probabilities after every chunk. Timestamps describe EEG sample time; replay runs as fast as computation allows and is not a live electrode interface.

To deliver each complete held-out recording in ten new chronological pieces, using the existing pilot weights:

```powershell
.venv\Scripts\python.exe inference/replay_deciles.py --runs results/streaming_research_pilot_v2/runs.json --out results/my_decile_replay --parts 10 --threads 2
.venv\Scripts\python.exe visualizations/summarize_deciles.py --results results/my_decile_replay --export docs/my_decile_replay
```

The [completed 10% replay report](docs/continuous_deciles/report.md) includes all 11 models, prediction timelines, probabilities, and CSV exports. Here 10% means approximately 38.7 seconds of a whole recording, with one prediction after each delivery. Streaming GRUs process every new sample and retain state; fixed-window models classify the latest three seconds received. Only 24 of 60 endpoints per model have a known task label, so these scores use a different cohort from the pilot's final-trial scores. No predictions are emitted inside a delivery block. The five replay correctness tests run with `python -m pytest tests/test_replay_deciles.py -q`.

## Causal CSP and recurrent research

The [CSP architecture plan](docs/CSP_RESEARCH_PLAN.md) extends the original GRU without replacing its code or experiments. New comparisons include classical CSP+LDA, causal streaming CSP+LDA with exact or exponentially weighted variance, CSP component signals feeding a GRU (D1), CSP log-variance features feeding a GRU (D2), filtered/unfiltered raw GRU controls, EEGNet, and a once-per-window GRU. [Mathematics, dataset licensing, protocol, related work and limitations](docs/CSP_RESEARCH.md) explain the comparisons.

```powershell
.venv\Scripts\python.exe -m pytest tests/test_csp_research.py -q
.venv\Scripts\python.exe experiments/run_csp_research.py --config configs/csp_pilot.json --out results/my_csp_run
.venv\Scripts\python.exe experiments/audit_csp.py --results results/my_csp_run --out results/my_csp_run/audit.json
.venv\Scripts\python.exe visualizations/summarize_csp.py --results results/my_csp_run --export docs/my_csp_run
```

The initial grid varies CSP components (4/8) and hidden dimensions (4/8), using three epochs. `configs/csp_followup.json` separately studies D2, LDA controls and EEGNet with up to 20 epochs and three seeds. Both configurations currently use the locally cached participant 1. Add subjects/seeds or adjust the schedule in JSON; fresh outputs are required, and `--resume` checks configuration and source hashes. Final and intermediate GRU supervision, clipping and chronological TBPTT reuse the established training engine.

The causal API consumes raw **microvolts**, with fixed checkpoint preprocessing. It emits one readout per 100-ms clock tick, preserves all filter/variance/recurrent state across acquisition chunks, and isolates independent sessions:

```python
import torch
from streaming_eeg.csp_pipeline import CSPResearchPredictor

artifact = torch.load("results/my_csp_run/subject1_seed7/csp_power_gru_k4_h8_continuous/model.pt", map_location="cpu", weights_only=True)
predictor = CSPResearchPredictor(artifact)
predictor.reset_state("recording-a")
readouts = predictor.process_chunk(new_eeg_microvolts, "recording-a")
latest = predictor.get_prediction("recording-a")
```

Replay a fitted model using `python inference/csp_demo.py --checkpoint PATH --out results/my_csp_demo.jsonl --streams 2 --chunks 100`. SciPy preprocessing runs on CPU; neural inference can use CPU/CUDA. GPU checks skip explicitly when unavailable. Offline completed-window and continuous-recording results are reported separately. Exact sliding variance has finite support; EW variance has a decaying history. End-to-end timings include preprocessing, and energy savings are not inferred from latency or coefficient count.

## Layout and research scope

- `streaming_eeg/`: causal model, stream management, raw data, TBPTT, baselines and probes.
- `train_streaming.py`, `evaluate_streaming.py`, `train_streaming_baseline.py`: experiment entry points.
- `inference/`, `experiments/`, `benchmarks/`, `visualizations/`: replay, studies, isolated timing/memory, and figures.
- `configs/`, `docs/`, `tests/`: configurations, research documents, and correctness checks.
- `data/`, `checkpoints/`, `results/`: local recordings, trained artifacts, and measurements; excluded from Git.

Operation estimates, sampled process RSS, observation duration, and measured CPU latency are separate quantities. Process RSS includes framework/runtime overhead and can miss short allocations. Hidden-state plots or convergence do not prove useful memory. Offline prompted-trial replay does not establish self-paced intention detection or deployment robustness.

Future ideas are documented without adding adaptive skipping, confidence halting, attention, or auxiliary networks to the initial architecture. Confidence thresholds are evaluation criteria only.

Memory studies are separate, explicitly labeled experiments:

```powershell
.venv\Scripts\python.exe experiments/synthetic_memory.py --out results/simulated_cue_recall
.venv\Scripts\python.exe experiments/synthetic_transitions.py --checkpoints checkpoints/my_stream.pt --out results/synthetic_trial_transitions
.venv\Scripts\python.exe visualizations/summarize_streaming.py --results results/streaming_research_pilot_v2
```

The cue-recall task is nonphysiological simulated data. The transition stress test uses distinct held-out EEG trials but joins them artificially; it omits natural rest and physiological transitions. Neither is presented as natural intention-change performance.

Verify timing against native dataset triggers and audit a completed grid:

```powershell
.venv\Scripts\python.exe experiments/check_alignment.py --dataset 2a --subject 1 --out results/2a_alignment.json
.venv\Scripts\python.exe experiments/audit_results.py --results results/streaming_research_pilot_v2 --out results/pilot_audit.json
```

After creating `verification.json` in a completed result directory, export a compact report with `visualizations/export_report.py --results results/streaming_research_pilot_v2 --out docs/my_pilot_report`. Exports include figures and comparison data; large probability traces and checkpoints remain local.
