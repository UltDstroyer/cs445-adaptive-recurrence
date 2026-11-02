# CS 445: Adaptive Recurrence

A small research project investigating whether a recurrent neural network can use fewer processing steps to classify EEG motor-imagery signals without sacrificing much accuracy.

The planned task is left-hand versus right-hand imagery using BCI Competition IV Dataset 2b (available through MOABB as `BNCI2014_004`).

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python data.py
```

The first run of `data.py` downloads EEG recordings for one participant and prints basic recording information. Downloading may take a while.

## Files

- `data.py`: inspect the EEG dataset; labeled trial extraction is not implemented yet.
- `model.py`: simple network layers and a TODO for the recurrent forward pass.
- `train.py`: training entry point (TODO).
- `evaluate.py`: fixed-depth versus adaptive evaluation entry point (TODO).

The project is intentionally incomplete so the model and experiments can be implemented and understood one piece at a time.

AI assistance: ChatGPT helped prepare the initial repository scaffold and README. The research implementation, experiments, and analysis are still in progress.
