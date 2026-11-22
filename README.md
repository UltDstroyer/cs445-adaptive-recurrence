# CS 445: Fashion-MNIST DNN

This project trains a feedforward deep neural network to classify clothing images.
The main experiment now uses a DNN and Fashion-MNIST.

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) contains 60,000
training images and 10,000 test images. Each image is 28 by 28 pixels in grayscale,
with one of ten clothing labels. This is a practical benchmark for a fully
connected DNN.

The default network is:

    28 x 28 image -> Flatten (784) -> Dense (256) -> ReLU -> Dropout
                 -> Dense (128) -> ReLU -> Dropout -> Dense (10)

Each dense layer has its own weights. The model produces one prediction per
image in a single forward pass. Training uses cross-entropy loss and AdamW,
with dropout 0.2 and learning rate 0.001.

## Setup

    python -m venv .venv
    .venv\Scripts\python.exe -m pip install -r requirements.txt

On macOS/Linux, use .venv/bin/python in place of .venv\Scripts\python.exe.

## Train and evaluate

    .venv\Scripts\python.exe data.py
    .venv\Scripts\python.exe train.py --epochs 12
    .venv\Scripts\python.exe evaluate.py --checkpoint checkpoints/dnn.pt

The first run downloads Fashion-MNIST from the publisher's HTTPS repository into
data/vision/. The official training partition is split into 50,000 training and
10,000 validation images using a fixed random split seed (7). The official
10,000 test images remain separate. Pixels are scaled to [0, 1], then standardized
using only the training split's mean and standard deviation.

Training saves the checkpoint with the highest validation accuracy, breaking
ties with lower validation loss. The checkpoint includes the architecture,
class names, split settings and normalization statistics. Evaluation restores
these settings and reports loss, accuracy, macro-F1, class counts, a confusion
matrix, and measured forward-pass time on validation and test images. Test
metrics are not used for checkpoint selection.

Default outputs:

- checkpoints/dnn.pt: best validation checkpoint.
- checkpoints/dnn_training.json: epoch history and training settings.
- results/dnn_evaluation.json: evaluation metrics.

To change the network or enable early stopping:

    .venv\Scripts\python.exe train.py --hidden-sizes 512 256 128 --dropout 0.3 --patience 4

--seed changes model initialization and training randomness. --split-seed changes
which official training images are reserved for validation. Keep the split seed
fixed when comparing architectures. --data-dir changes the download/cache
directory; evaluation also accepts it to relocate a saved experiment.
--no-download requires the data to exist locally.

## Offline pipeline check

    .venv\Scripts\python.exe train.py --demo --epochs 2 --out checkpoints/dnn_demo.pt
    .venv\Scripts\python.exe evaluate.py --checkpoint checkpoints/dnn_demo.pt --out results/dnn_demo.json

Demo mode creates artificial ten-class images for checking the code. Its metrics
are not Fashion-MNIST benchmark results.

## Tests

    .venv\Scripts\python.exe -m pip install pytest
    .venv\Scripts\python.exe -m pytest -q

## Project files

- data.py: image loading, train/validation split and normalization.
- model.py: feedforward DNN.
- train.py: DNN training, validation checkpointing and epoch history.
- evaluate.py: DNN classification metrics and forward-pass timing.
- tests/: offline tests for the data pipeline, model, training and evaluation.

## Earlier EEG experiment

This project began as a recurrent-network experiment on EEG data. That code was
removed from the working branch and is preserved at the git tag eeg-archive.
EEG checkpoints cannot be evaluated with the current commands.

## Limitations

Forward-pass timings exclude image loading and device transfers.
Use multiple training seeds and hold out the test set when comparing model
settings. No trained accuracy is assumed from the architecture alone.

## AI use

AI assistance: ChatGPT wrote and checked the initial model, data pipeline, training,
evaluation, and tests. The research experiments and interpretation remain ongoing.

Claude (Anthropic) helped plan the early-exit experiments and removed the EEG code
from this branch.
