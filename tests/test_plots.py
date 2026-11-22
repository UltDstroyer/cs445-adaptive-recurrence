import csv

import numpy as np
import pytest

import plots
import sweep
from exit_outputs import save_exit_outputs

CNN_COUNTS = {"stem": 28_224, "blocks": [451_584] * 3 + [250_880] + [451_584] * 6, "heads": [160] * 3 + [320] * 7}


def _save(path, split="val", exits=10, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(200, exits, 10)).astype(np.float32) * np.linspace(0.5, 4, exits)[None, :, None]
    labels = rng.integers(0, 10, 200)
    counts = CNN_COUNTS if exits == 10 else {"stem": 0, "blocks": [7_840], "heads": [0]}
    save_exit_outputs(path, logits, labels, np.arange(200), {"split": split, "multiply_adds": counts})
    return path


def test_exit_figure_and_its_table(tmp_path):
    out = tmp_path / "exits.png"
    plots.main(["exits", "--series", f"CNN={_save(tmp_path / 'a.npz')}",
                "--series", f"MLP={_save(tmp_path / 'b.npz', seed=1)}",
                "--logistic", str(_save(tmp_path / 'lr.npz', exits=1)), "--out", str(out)])
    assert out.stat().st_size > 10_000
    rows = list(csv.DictReader(out.with_suffix(".csv").open()))
    assert len(rows) == 21  # 10 exits x 2 series + the logistic baseline
    assert {r["series"] for r in rows} == {"CNN", "MLP", "logistic regression"}
    assert all(0 <= float(r["macro_f1"]) <= 1 for r in rows)


def test_confidence_figure_from_sweep_results(tmp_path):
    outputs = tmp_path / "x_val_outputs.npz"
    _save(outputs)
    sweep.main([str(outputs), "--inspect", "1"])
    out = tmp_path / "confidence.png"
    plots.main(["confidence", str(tmp_path / "x_val_baselines.json"), "--out", str(out)])
    rows = list(csv.DictReader(out.with_suffix(".csv").open()))
    assert sum(r["rule"] == "fixed exit" for r in rows) == 10
    assert sum(r["rule"] == "confidence only" for r in rows) == len(sweep.CONFIDENCE_THRESHOLDS)


def test_plots_refuse_test_outputs(tmp_path):
    test_outputs = _save(tmp_path / "t.npz", split="test")
    with pytest.raises(SystemExit):
        plots.main(["exits", "--series", f"CNN={test_outputs}", "--out", str(tmp_path / "x.png")])
    with pytest.raises(SystemExit):
        plots.main(["exits", "--series", f"CNN{test_outputs}", "--out", str(tmp_path / "x.png")])
