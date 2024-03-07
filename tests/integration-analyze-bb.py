#!/usr/bin/env pytest

import pytest

import glob
from pathlib import Path
import subprocess
import time

import os
import sys

from utils import run_script

import_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.append(import_path)

from pmtestbench.common.portmapping import Mapping


def test_relaxeduops_plain(tmp_path):
    """ Test that the analyze-bb.py script works
    """

    base_dir = Path(__file__).parent
    test_resources_dir = base_dir / "resources"

    # create a temporary directory
    test_dir = tmp_path / "analyze-bb"
    test_dir.mkdir()

    # set up the test environment
    mapping_file = test_resources_dir / "inferred_mapping_zenp.json"
    assert mapping_file.exists()

    asm_file = test_resources_dir / "test.s"
    assert asm_file.exists()

    # run the relaxeduops algorithm
    script_path = base_dir.parent / "scripts" / "analyze-bb.py"

    res = run_script(script_path, [
            "--file", "--asm", asm_file,
            mapping_file
        ], capture_output=True, text=True)

    assert res.returncode == 0

    output = res.stdout

    assert "Inverse throughput according to the port mapping: 1.00 cycles/iteration" in output
    assert "Inverse throughput adjusted for the peak IPC:     1.20 cycles/iteration" in output

    assert True


