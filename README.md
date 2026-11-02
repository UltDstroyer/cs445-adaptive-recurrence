# CS 445: Adaptive Recurrence

Can a small recurrent network use fewer processing steps to classify motor-imagery EEG signals without losing much accuracy?

This project uses BCI Competition IV Dataset 2b (left-hand versus right-hand imagery). The network encodes an EEG window, applies the same recurrent block up to six times, and predicts after every pass. During adaptive inference, it can stop when the predicted class, confidence, and probabilities are stable.

## Setup

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

## Run with EEG data

    python data.py --subject 1
    python train.py --subject 1 --epochs 12
    python evaluate.py --checkpoint checkpoints/model.pt

The first run downloads participant 1 through MOABB. Preprocessing uses an 8-32 Hz filter, 100 Hz resampling, and a 0.5-3.5 second window relative to the task event. The first three sessions are training data, the fourth is validation, and the fifth is held-out test data. Only the training sessions determine normalization statistics.

Training saves the best validation checkpoint to checkpoints/model.pt. Evaluation reports fixed-depth accuracies, adaptive accuracy, average iterations, latency, and exit counts; it also writes results/evaluation.json. Checkpoints, results, and downloaded data are excluded from Git.

Stopping thresholds can be adjusted without retraining:

    python evaluate.py --confidence 0.75 --stability 0.10

## Quick pipeline check without EEG downloads

    python data.py --demo
    python train.py --demo --epochs 2 --out checkpoints/demo.pt
    python evaluate.py --checkpoint checkpoints/demo.pt --out results/demo.json

Demo mode uses artificial signals. Its accuracy is NOT a BCI result.

## Project files

- data.py: trials, labels, session splits, and training-only normalization.
- model.py: encoder, shared recurrent computation, and early stopping.
- train.py: training with a loss at every recurrent depth.
- evaluate.py: fixed and adaptive inference results and measured latency.
- tests/: tests that require no EEG downloads.

To run tests:

    python -m pip install pytest
    python -m pytest -q

## Limitations

This is a small educational model, not a validated BCI system. One participant does not establish generalization to new people. Stopping thresholds are exploratory and should not be tuned on the held-out test set. Fewer recurrent passes do not guarantee faster wall-clock execution. Adaptive stopping saves processing on an already collected EEG window; it does not shorten the signal recording.

AI assistance: ChatGPT wrote and checked the initial model, data pipeline, training, evaluation, and tests. The research experiments and interpretation remain ongoing.
