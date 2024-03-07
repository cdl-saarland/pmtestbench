#!/usr/bin/env pytest

import pytest

from pathlib import Path
import subprocess
import time

import os
import sys

import_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.append(import_path)

from pmtestbench.common.portmapping import Mapping

def run_script(script_path, args):
    cmd = [
        "/usr/bin/env",
        "python3",
        script_path,
        *args
        ]
    res = subprocess.run(cmd)
    return res

def test_ceg_plain(tmp_path):
    """ Test that the counter-example-guided algorithm can be run without any
    issues.

    Since this SMT-solver-based algorithm is quite slow, especially in the
    three-level setting, we only use a tiny test case.
    """

    base_dir = Path(__file__).parent
    test_resources_dir = base_dir / "resources"

    # create a temporary directory
    test_dir = tmp_path / "cegpmi"
    test_dir.mkdir()

    # set up the test environment
    processor_file = test_resources_dir / "trivial_pm.json"
    assert processor_file.exists()

    synth_file = test_resources_dir / "trivial_ceg_config.json"
    assert synth_file.exists()


    # run inference
    result_file = test_dir / "result.json"
    res = run_script(base_dir.parent / "scripts" / "infer-ceg.py", [
            "--output", result_file,
            "--processor", processor_file,
            "--synthesizer", synth_file,
        ])

    assert res.returncode == 0
    assert result_file.exists()

    with open(result_file, "r") as f:
        mapping = Mapping.read_from_json(f)

    print(mapping)

    assert True

