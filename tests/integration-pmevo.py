#!/usr/bin/env pytest

import pytest

from pathlib import Path
import subprocess
import time

import os
import sys

from utils import run_script

import_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.append(import_path)

from pmtestbench.common.portmapping import Mapping


def test_pmevo_plain(tmp_path):
    """ Test that the pmevo algorithm can be run without any issues.
    """

    base_dir = Path(__file__).parent
    test_resources_dir = base_dir / "resources"

    # create a temporary directory
    test_dir = tmp_path / "pmevo"
    test_dir.mkdir()

    # set up the test environment
    processor_file = test_resources_dir / "simple_pm.json"
    assert processor_file.exists()

    exp_file = test_dir / "experiments.json"

    equiv_epsilon = 0.05

    # generate and evaluate the experiments
    res = run_script(base_dir.parent / "scripts" / "gen_experiments.py", [
            "--no-verbose",
            "full-pmevo",
            "--epsilon", str(equiv_epsilon),
            "--output", exp_file,
            processor_file
        ])

    assert res.returncode == 0
    assert exp_file.exists()


    # run inference
    result_file = test_dir / "result.json"
    res = run_script(base_dir.parent / "scripts" / "infer-pmevo.py", [
            "--output", result_file,
            "--num-ports", "4",
            "--epsilon", str(equiv_epsilon),
            "--config", base_dir / ".." / "lib" / "cpp-evolution" / "run_configs" / "small.cfg",
            exp_file
        ])

    assert res.returncode == 0
    assert result_file.exists()

    with open(result_file, "r") as f:
        mapping = Mapping.read_from_json(f)

    print(mapping)

    assert True

