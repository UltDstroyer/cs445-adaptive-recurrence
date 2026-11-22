# CS 445: Early-Exit Rules on Fashion-MNIST

This project compares early-exit stopping rules on a small multi-exit network
that classifies clothing images. Easy images can stop at an early exit and save
computation; the question is which stopping rule gives the best tradeoff between
macro-F1 and computation.

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) contains 60,000
training images and 10,000 test images. Each image is 28 by 28 pixels in grayscale,
with one of ten clothing labels.

The network is:

    28 x 28 image -> Flatten (784) -> Linear (32)
                  -> block 1 -> head 1 (10 classes)
                  -> block 2 -> head 2
                  ...
                  -> block 10 -> head 10

Each block is residual, h + Linear(128->32)(ReLU(Linear(32->128)(LayerNorm(h)))),
and every block and head has its own weights (112,580 parameters in total). Exit k
is the head after block k. Training uses the cross-entropy averaged over all ten
exits, with AdamW at learning rate 0.001.

## Setup

    python -m venv .venv
    .venv\Scripts\python.exe -m pip install -r requirements.txt

On macOS/Linux, use .venv/bin/python in place of .venv\Scripts\python.exe.

## Prepare data and train

    .venv\Scripts\python.exe data.py
    .venv\Scripts\python.exe train.py --seed 0

The first run downloads Fashion-MNIST from the publisher's HTTPS repository into
data/vision/. The official training partition is split into 54,000 training and
6,000 validation images. The split is stratified, so validation has 600 images
of each class, and it uses a fixed split seed (7) that stays the same across
training seeds. The official 10,000 test images remain separate. Pixels are
scaled to [0, 1] with no further normalization.

data.py prints class counts for each split and saves a summary to
results/data_summary.json: class counts, pixel statistics for the training and
validation images, a disjointness check and a fingerprint of the validation
images. Test images are counted only.

After each epoch, train.py scores every exit on the validation images (loss,
accuracy and macro-F1). The checkpoint keeps the epoch with the best macro-F1
averaged across exits, breaking ties with lower validation loss, so early exits
count as much as the last one. Test images are not used in training or
selection. The run reports its training time and an estimate for five seeds.

Default outputs:

- results/data_summary.json: split summary from data.py.
- checkpoints/multi_exit.pt: best validation checkpoint, with the architecture,
  split settings and per-exit validation scores.
- checkpoints/multi_exit_training.json: epoch-by-epoch validation scores for
  every exit, epoch times and training settings.

Useful options:

    .venv\Scripts\python.exe train.py --seed 1 --epochs 20 --patience 4

--seed changes model initialization and training randomness. --split-seed changes
which official training images are reserved for validation; keep it fixed across
seeds. --width, --block-hidden and --num-exits change the network. --data-dir
changes the download/cache directory, and --no-download requires the data to
exist locally.

evaluate.py does not accept multi-exit checkpoints yet. Separate validation and
test evaluation, with saved per-exit outputs, is the next piece of work.

## Offline pipeline check

    .venv\Scripts\python.exe data.py --demo
    .venv\Scripts\python.exe train.py --demo --epochs 2 --out checkpoints/demo.pt

Demo mode creates artificial ten-class images for checking the code. Its metrics
are not Fashion-MNIST benchmark results.

## Tests

    .venv\Scripts\python.exe -m pip install pytest
    .venv\Scripts\python.exe -m pytest -q

## Project files

- data.py: image loading, stratified train/validation split, [0, 1] scaling and split summary.
- model.py: the multi-exit network with stop-anywhere sequential inference, and
  the earlier single-exit DNN.
- train.py: multi-exit training, per-exit validation scoring, checkpoint
  selection and timing.
- evaluate.py: metrics for single-exit DNN checkpoints (being rebuilt for the
  multi-exit network).
- tests/: offline tests for the data pipeline, model, training and evaluation.

## Earlier EEG experiment

This project began as a recurrent-network experiment on EEG data. That code was
removed from the working branch and is preserved at the git tag eeg-archive.
EEG checkpoints cannot be evaluated with the current commands.

## Limitations

Use multiple training seeds and hold out the test set when comparing model
settings. No trained accuracy is assumed from the architecture alone.

## AI use

AI assistance: ChatGPT wrote and checked the initial model, data pipeline, training,
evaluation, and tests. The research experiments and interpretation remain ongoing.

Claude (Anthropic) helped plan the early-exit experiments, removed the EEG code,
and wrote the stratified split, the multi-exit network, multi-exit training and
their tests.
