#!/usr/bin/env pytest

import pytest

import os
import sys

import_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.append(import_path)

from pmtestbench.common.architecture import Architecture

from pmtestbench.common.experiments import Experiment

from pmtestbench.common.portmapping import Mapping, Mapping2, Mapping3

from pmtestbench.common.processors.portmapping_processor import PurePortMappingProcessor

@pytest.fixture()
def simple_arch():
    arch = Architecture()
    arch.add_insns(['a', 'b', 'c', 'd'])
    return arch


@pytest.fixture
def simple_mapping2(simple_arch):
    a, b, c, d, *ri = simple_arch.insn_list

    pm = Mapping2.from_model(simple_arch, 3, {
            (a, 0): True,
            (a, 1): True,
            (b, 0): True,
            (b, 1): True,
            (c, 0): True,
            (d, 2): True,
        })
    return pm

@pytest.fixture
def simple_mapping3(simple_arch):
    a, b, c, d, *ri = simple_arch.insn_list

    pm = Mapping3(simple_arch)
    pm.assignment[a] = [[0], [0]]
    pm.assignment[b] = [[0, 1]]
    pm.assignment[c] = [[0, 1]]
    pm.assignment[d] = [[0, 1], [2]]

    return pm


def test_simple_arch(simple_arch):
    assert len(simple_arch.insn_list) == 4

def test_simple_arch_json(simple_arch):
    json_str = simple_arch.to_json_str()

    new_arch = Architecture.from_json_str(json_str)

    assert len(new_arch.insn_list) == 4


def test_experiment_01(simple_arch):
    a, b, c, d, *r = simple_arch.insn_list

    exp = Experiment([a, a, b])

    assert exp.num_occurrences(a) == 2
    assert exp.num_occurrences(b) == 1
    assert exp.num_occurrences(c) == 0

def test_experiment_01_json(simple_arch):
    a, b, c, d, *r = simple_arch.insn_list

    exp = Experiment([a, a, b])

    json_str = exp.to_json_str()
    new_exp = Experiment.from_json_str(json_str)

    assert new_exp == exp

def test_experiment_02(simple_arch):
    a, b, c, d, *r = simple_arch.insn_list

    exp1 = Experiment([a, a, b])
    exp2 = Experiment([a, b, b])

    assert exp1 != exp2


def test_experiment_03(simple_arch):
    a, b, c, d, *r = simple_arch.insn_list

    exp1 = Experiment([a, a, b])
    exp2 = Experiment([a, b, a])

    assert exp1 == exp2


def test_mapping2_01_json(simple_mapping2):
    json_str = simple_mapping2.to_json_str()

    new_mapping = Mapping.read_from_json_str(json_str)

    assert simple_mapping2.assignment == new_mapping.assignment


def test_bn_proc_01(simple_mapping2):
    arch = simple_mapping2.arch
    proc = PurePortMappingProcessor(simple_mapping2)

    a, b, c, d, *r = arch.insn_list

    exp1 = Experiment([a, a, b])
    assert proc.get_cycles(exp1) == 1.5

def test_bn_proc_02(simple_mapping2):
    arch = simple_mapping2.arch
    proc = PurePortMappingProcessor(simple_mapping2)

    a, b, c, d, *r = arch.insn_list

    exp1 = Experiment([a, a, b, d, d])
    assert proc.get_cycles(exp1) == 2.0


def test_mapping3_01_json(simple_mapping3):
    json_str = simple_mapping3.to_json_str()

    new_mapping = Mapping.read_from_json_str(json_str)

    assert simple_mapping3.assignment == new_mapping.assignment

def test_bn_proc3_01(simple_mapping3):
    arch = simple_mapping3.arch
    proc = PurePortMappingProcessor(simple_mapping3)

    a, b, c, d, *r = arch.insn_list

    exp1 = Experiment([a, a, b])
    assert proc.get_cycles(exp1) == 4.0

def test_bn_proc3_02(simple_mapping3):
    arch = simple_mapping3.arch
    proc = PurePortMappingProcessor(simple_mapping3)

    a, b, c, d, *r = arch.insn_list

    exp1 = Experiment([b, b, c])
    assert proc.get_cycles(exp1) == 1.5

def test_bn_proc3_03(simple_mapping3):
    arch = simple_mapping3.arch
    proc = PurePortMappingProcessor(simple_mapping3)

    a, b, c, d, *r = arch.insn_list

    exp1 = Experiment([d, d])
    assert proc.get_cycles(exp1) == 2.0

def test_bn_proc3_04(simple_mapping3):
    arch = simple_mapping3.arch
    proc = PurePortMappingProcessor(simple_mapping3)

    a, b, c, d, *r = arch.insn_list

    exp1 = Experiment([b, d, d])
    assert proc.get_cycles(exp1) == 2.0


