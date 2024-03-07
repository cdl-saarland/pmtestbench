#!/usr/bin/env pytest

import pytest

import itertools
import os
import sys

import_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.append(import_path)

from pmtestbench.common.architecture import Architecture
from pmtestbench.common.experiments import Experiment
from pmtestbench.common.portmapping import Mapping, Mapping2, Mapping3

from pmtestbench.common.processors import Processor
from pmtestbench.common.synthesizers import Synthesizer

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
def simple_proc(simple_mapping2):
    proc = Processor(config=simple_mapping2)
    return proc

@pytest.fixture
def simple_mapping3(simple_arch):
    a, b, c, d, *ri = simple_arch.insn_list

    pm = Mapping3(simple_arch)
    pm.assignment[a] = [[0], [0]]
    pm.assignment[b] = [[0, 1]]
    pm.assignment[c] = [[0, 1]]
    pm.assignment[d] = [[0, 1], [2]]

    return pm

@pytest.fixture
def simple_proc3(simple_mapping3):
    proc = Processor(config=simple_mapping3)
    return proc


@pytest.mark.parametrize("insnbound", [2, 3, 4])
def test_synthesis_01(simple_proc, insnbound):
    synth = Synthesizer(config={
            "synthesizer_kind": "smt",
            "mapping_class": "Mapping2",
            "num_ports": 3,
            "smt_slack_val": 0.0,
            "smt_insn_bound": insnbound,
            "smt_exp_limit_strategy": "incremental_bounded",
        })

    m = synth.synthesize(simple_proc)

    assert m is not None

    new_proc = Processor(config=m)

    arch = simple_proc.get_arch()
    insns = arch.insn_list

    for ilist in itertools.combinations_with_replacement(insns, insnbound):
        exp = Experiment(ilist)
        assert new_proc.get_cycles(exp) == simple_proc.get_cycles(exp)

def test_synthesis_02(simple_proc3):
    # This test tries to find a Mapping2 for a Mapping3 that is not
    # representable.
    insnbound = 3
    synth = Synthesizer(config={
            "synthesizer_kind": "smt",
            "mapping_class": "Mapping2",
            "num_ports": 3,
            "smt_slack_val": 0.0,
            "smt_insn_bound": insnbound,
            "smt_exp_limit_strategy": "incremental_bounded",
        })

    m = synth.synthesize(simple_proc3)

    assert m is None

@pytest.mark.parametrize("insnbound", [2, 3, 4])
def test_synthesis3_01(simple_proc3, insnbound):
    synth = Synthesizer(config={
            "synthesizer_kind": "smt",
            "mapping_class": "Mapping3",
            "num_ports": 3,
            "smt_slack_val": 0.0,
            "smt_insn_bound": insnbound,
            "smt_exp_limit_strategy": "incremental_bounded",
            "num_uops": 4,
        })

    m = synth.synthesize(simple_proc3)

    assert m is not None

    new_proc = Processor(config=m)

    arch = simple_proc3.get_arch()
    insns = arch.insn_list

    for ilist in itertools.combinations_with_replacement(insns, insnbound):
        exp = Experiment(ilist)
        assert new_proc.get_cycles(exp) == simple_proc3.get_cycles(exp)



@pytest.mark.parametrize("insnbound", [2, 3, 4])
def test_synthesis3_constrained_01(simple_proc3, insnbound):
    synth = Synthesizer(config={
            "synthesizer_kind": "smt",
            "mapping_class": "Mapping3",
            "num_ports": 3,

            "smt_use_constrained_mapping3": True,
            "num_uops_per_insn": {
                    "a": 2,
                    "b": 1,
                    "c": 1,
                    "d": 2,
                },

            "smt_slack_val": 0.0,
            "smt_insn_bound": insnbound,
            "smt_exp_limit_strategy": "incremental_bounded",
        })

    m = synth.synthesize(simple_proc3)

    assert m is not None

    new_proc = Processor(config=m)

    arch = simple_proc3.get_arch()
    insns = arch.insn_list

    for ilist in itertools.combinations_with_replacement(insns, insnbound):
        exp = Experiment(ilist)
        assert new_proc.get_cycles(exp) == simple_proc3.get_cycles(exp)


