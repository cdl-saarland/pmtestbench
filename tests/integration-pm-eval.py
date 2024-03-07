#!/usr/bin/env pytest

import pytest

from pathlib import Path
import re
import time

import os
import sys

from utils import run_script

import_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.append(import_path)



def test_pmeval(tmp_path):
    """ Test the scripts to evaluate prediction accuracy.
    """

    base_dir = Path(__file__).parent
    test_resources_dir = base_dir / "resources"

    test_dir = tmp_path / "pmeval"
    test_dir.mkdir()

    processor_file = test_resources_dir / "simple_pm.json"
    assert processor_file.exists()

    num_experiments = 100
    exp_length = 5

    exp_file = test_dir / "experiments.json"

    res = run_script(base_dir.parent / "scripts" / "gen_experiments.py", [
            "--no-verbose",
            "make-validation",
            "--num-experiments", str(num_experiments),
            "--length", str(exp_length),
            "--output", exp_file,
            processor_file
        ])

    assert res.returncode == 0
    assert exp_file.exists()

    eval_exp_file = test_dir / "experiments_eval.json"

    res = run_script(base_dir.parent / "scripts" / "gen_experiments.py", [
            "--no-verbose",
            "eval-validation",
            "--output", eval_exp_file,
            processor_file,
            exp_file
        ])

    assert res.returncode == 0
    assert eval_exp_file.exists()

    ann_exp_file = test_dir / "experiments_ann.json"

    res = run_script(base_dir.parent / "scripts" / "annotate_predictions.py", [
            "--result-id", "annotated01",
            "--output", ann_exp_file,
            processor_file,
            eval_exp_file
        ])

    assert res.returncode == 0
    assert ann_exp_file.exists()


    res = run_script(base_dir.parent / "scripts" / "eval_annotated_predictions.py", [
            "--metric", "IPC",
            "--mode", "heatmap",
            "-i", "annotated01",
            ann_exp_file
        ], capture_output=True, text=True)

    assert res.returncode == 0

    output = res.stdout

    # The annotated numbers stem from the same port mapping as the reference
    # numbers, we should therefore find an error of 0.
    assert re.search(r"am_error:\s*0.0\n", output)

