"""Inspect the EEG recordings before building the training dataset.

BCI Competition IV Dataset 2b is available through MOABB as BNCI2014_004.
This script loads one participant and prints basic recording information.
It does not yet extract labeled trials or create a training split.
"""

from moabb.datasets import BNCI2014_004


def main():
    dataset = BNCI2014_004()
    print("Loading participant 1 (may download data on first run)...")
    subjects = dataset.get_data(subjects=[1])

    for session_name, runs in subjects[1].items():
        print(f"\nSession: {session_name}")
        for run_name, raw in runs.items():
            eeg_channels = [name for name in raw.ch_names if name in ("C3", "Cz", "C4")]
            print(
                f"  Run {run_name}: "
                f"{raw.info['sfreq']} Hz, "
                f"{raw.n_times} samples, "
                f"{len(raw.annotations)} annotations, "
                f"EEG channels: {eeg_channels or raw.ch_names}"
            )


if __name__ == "__main__":
    main()
