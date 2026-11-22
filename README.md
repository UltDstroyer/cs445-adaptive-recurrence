# CS 445: Early-Exit Rules on Fashion-MNIST

This project compares early-exit stopping rules on a small multi-exit network
that classifies clothing images. Easy images can stop at an early exit and save
computation; the question is which stopping rule gives the best tradeoff between
macro-F1 and computation.

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) contains 60,000
training images and 10,000 test images. Each image is 28 by 28 pixels in grayscale,
with one of ten clothing labels.

The network is a small multi-exit CNN:

    28 x 28 image -> 3x3 conv, stride 2 (16 channels, 14 x 14)
                  -> blocks 1-3: residual 3x3 conv, 16 channels at 14 x 14
                  -> block 4: residual 3x3 conv, stride 2, 32 channels at 7 x 7
                  -> blocks 5-10: residual 3x3 conv, 32 channels at 7 x 7

Each block is shortcut(h) + Conv(ReLU(GroupNorm(h))) and is followed by its own
head (GroupNorm, ReLU, average over the image, linear layer to 10 classes). Exit k
is the head after block k, and no blocks or heads share weights (71,668
parameters). A full pass costs 4,346,080 multiply-adds; stopping at exit k costs
11.0, 21.4, 31.8, 37.6, 48.0, 58.4, 68.8, 79.2, 89.6 or 100% of that. Training
uses the cross-entropy averaged over all ten exits, with AdamW at learning rate
0.001.

The first design was a multi-exit MLP (784->32 projection, ten 32->128->32
residual blocks). Its validation macro-F1 was flat across exits (exit 10 did not
beat exit 2), leaving the stopping rules nothing to trade off, so the project
switched to the CNN. The MLP is still available with --architecture mlp and is
preserved at the git tag mlp-multi-exit.

## Setup

    python -m venv .venv
    .venv\Scripts\python.exe -m pip install -r requirements.txt

On macOS/Linux, use .venv/bin/python in place of .venv\Scripts\python.exe.

## Prepare data and train

    .venv\Scripts\python.exe data.py
    .venv\Scripts\python.exe train.py --seed 0

Training runs for 30 epochs. There are two ways to run it:

**Fast, on a GPU.** The default, --device auto, uses an NVIDIA GPU (cuda) or an
Apple-silicon Mac's GPU (mps) when one is available, and otherwise the CPU.

    .venv/bin/python train.py --seed 0

**On a slower computer, on the CPU.** Use a few threads; for this small network
2 threads were faster than 6 on our test machine and leave the computer usable.
Progress is saved after every epoch, so a run can be stopped with Ctrl+C (or by
closing the laptop) and continued later with the same command plus --resume:

    .venv\Scripts\python.exe train.py --seed 0 --device cpu --threads 2
    .venv\Scripts\python.exe train.py --seed 0 --device cpu --threads 2 --resume

A resumed run continues exactly where it stopped, and --resume refuses to continue
if the model, data or training settings differ (only --epochs, --device and
--threads may change). Free Google Colab GPUs (cuda) are another option.

| Computer (one epoch, about 54,000 images) | Time per epoch | 30 epochs |
|---|--:|--:|
| Apple M5 Max GPU (mps) | 3.7 s | about 2 min |
| Apple M5 Max CPU, 2 threads | 27 s | about 13 min |
| Apple M5 Max CPU, 6 threads | 36 s | about 18 min |

Older laptops will be slower on the CPU. train.py prints each epoch's time and
an estimate for five seeds.

The same seed gives slightly different numbers on a GPU and on a CPU, so train
all five seeds used for the final results on one computer and device.
Checkpoints record the device used.

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
- checkpoints/multi_exit_resume.pt: progress saved after every epoch, used by
  --resume.

Useful options:

    .venv\Scripts\python.exe train.py --seed 1 --epochs 20 --patience 4

--seed changes model initialization and training randomness. --split-seed changes
which official training images are reserved for validation; keep it fixed across
seeds. --architecture chooses cnn (default) or mlp; --channels and --downsample-at
change the CNN, and --width and --block-hidden change the MLP. --data-dir
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
- model.py: the multi-exit CNN and MLP (sharing stop-anywhere sequential
  inference), a helper that rebuilds either from a checkpoint, and the earlier
  single-exit DNN.
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
and wrote the stratified split, the multi-exit MLP and CNN, multi-exit training
and their tests.
