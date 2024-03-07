#!/usr/bin/env pytest

import pytest

import glob
from pathlib import Path
import subprocess
import time

import os
import sys

import_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.append(import_path)

from pmtestbench.common.portmapping import Mapping


def test_relaxeduops_plain(tmp_path):
    """ Test that the relaxeduops algorithm can be run without any issues.
    """

    base_dir = Path(__file__).parent
    test_resources_dir = base_dir / "resources"

    # create a temporary directory
    test_dir = tmp_path / "relaxeduops"
    test_dir.mkdir()

    # set up the test environment
    processor_file = test_resources_dir / "simple_pm.json"
    assert processor_file.exists()

    params_file = test_resources_dir / "relaxed_simple_params.json"
    assert params_file.exists()

    # run the relaxeduops algorithm
    script_path = base_dir.parent / "scripts" / "infer-relaxeduops.py"

    result_file = test_dir / "result.json"
    cmd = [
        "/usr/bin/env",
        "python3",
        script_path,
        "--output", result_file,
        "--report-dir", test_dir / "reports",
        "--params", params_file,
        processor_file
        ]
    res = subprocess.run(cmd)

    # check the results
    assert res.returncode == 0
    assert result_file.exists()

    with open(result_file, "r") as f:
        mapping = Mapping.read_from_json(f)

    print(mapping)

    assert True


@pytest.mark.parametrize("split_after", list(range(1, 8)))
def test_relaxeduops_split(tmp_path, split_after):
    """ Test that the relaxeduops algorithm can be split into two runs at any
    stage.
    """

    base_dir = Path(__file__).parent
    test_resources_dir = base_dir / "resources"

    # create a temporary directory
    test_dir1 = tmp_path / "relaxeduops01"
    test_dir1.mkdir()

    # use different directories for the second run to avoid conflicts with
    # timestamp-based subdirectory names
    test_dir2 = tmp_path / "relaxeduops02"
    test_dir2.mkdir()

    # set up the test environment
    processor_file = test_resources_dir / "simple_pm.json"
    assert processor_file.exists()

    params_file = test_resources_dir / "relaxed_simple_params.json"
    assert params_file.exists()

    # run the relaxeduops algorithm
    script_path = base_dir.parent / "scripts" / "infer-relaxeduops.py"

    result_file = test_dir1 / "result.json"
    cmd = [
        "/usr/bin/env",
        "python3",
        script_path,
        "--output", result_file,
        "--report-dir", test_dir1 / "reports",
        "--steps", str(split_after),
        "--params", params_file,
        processor_file
        ]
    res = subprocess.run(cmd)
    assert res.returncode == 0

    matches = glob.glob(str(test_dir1 / "reports") + f"/relaxed_uops_*/checkpoint_{split_after:02}_*.pickle")
    assert len(matches) == 1
    checkpoint_file = matches[0]
    print(f"Checkpoint file: {checkpoint_file}")

    result_file = test_dir2 / "result.json"
    cmd = [
        "/usr/bin/env",
        "python3",
        script_path,
        "--output", result_file,
        "--report-dir", test_dir2 / "reports",
        "--start-with", checkpoint_file,
        "--params", params_file,
        processor_file
        ]
    res = subprocess.run(cmd)
    assert res.returncode == 0

    # check the results
    assert result_file.exists()

    with open(result_file, "r") as f:
        mapping = Mapping.read_from_json(f)

    print(mapping)

    assert True
