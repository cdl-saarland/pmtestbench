#!/usr/bin/env pytest

import pytest

from collections import Counter
import itertools
import os
import sys

from z3 import *

import_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.append(import_path)

from pmtestbench.common.architecture import Architecture
from pmtestbench.common.experiments import Experiment
from pmtestbench.common.portmapping import Mapping, Mapping2, Mapping3

from pmtestbench.common.processors import Processor, ProcessorImpl
from pmtestbench.common.synthesizers import Synthesizer
from pmtestbench.cegpmi.smt_synthesizer import MappingHandler
from pmtestbench.common.utils import popcount

@pytest.fixture()
def simple_arch():
    arch = Architecture()
    arch.add_insns(['a', 'b', 'c', 'd'])
    return arch

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


class SynthWrapperProc(ProcessorImpl):
    def __init__(self, mapping, synth_config):
        self.mapping = mapping
        self.arch = mapping.arch
        self.synth = Synthesizer(config=synth_config)

    def get_arch(self):
        return self.arch

    def get_cycles(self, iseq):
        synth = self.synth.impl
        synth.reset_fields()
        synth.arch = self.arch
        s = Solver()
        synth.solver = s
        mh = MappingHandler.for_class(synth.mapping_cls, synth.config, synth=synth)

        mapping_enc = mh.add_mapping_vars('map1')
        mh.encode_valid_mapping_constraints(mapping_enc)
        mh.encode_mapping(mapping_enc, self.mapping)
        emap = Counter(iseq)
        exp_enc = mh.add_experiment_vars(emap.keys(), 'exp1')
        mh.encode_experiment(mapping_enc, exp_enc, emap)
        res = s.check()
        assert str(res) == 'sat'
        model = s.model()
        return float(model[exp_enc['t_var']].as_fraction())


def make_experiments(insns, bound):
    for ilist in itertools.combinations_with_replacement(insns, bound):
        yield ilist

def procs_agree(iseq, proc_a, proc_b, eps=0.0001):
    aval = proc_a.get_cycles(iseq)
    bval = proc_b.get_cycles(iseq)
    print(f"aval: {aval}")
    print(f"bval: {bval}")
    return abs(aval - bval) <= 0.0001


def test_bn_handler(simple_proc3):
    mapping = simple_proc3.impl.mapping
    arch = mapping.arch
    proc = SynthWrapperProc(mapping, {
            "synthesizer_kind": "smt",
            "mapping_class": "Mapping3",
            "num_ports": 3,
            "smt_slack_val": 0.0,
            "smt_use_bn_handler": True,
            "smt_insn_bound": 3,
            "smt_exp_limit_strategy": "incremental_bounded",
        })

    for iseq in make_experiments(arch.insn_list, 5):
        assert procs_agree(iseq, proc, simple_proc3)

def test_default_handler(simple_proc3):
    mapping = simple_proc3.impl.mapping
    arch = mapping.arch
    proc = SynthWrapperProc(mapping, {
            "synthesizer_kind": "smt",
            "mapping_class": "Mapping3",
            "num_ports": 3,
            "smt_slack_val": 0.0,
            "smt_use_bn_handler": False,
            "smt_insn_bound": 3,
            "smt_exp_limit_strategy": "incremental_bounded",
        })

    for iseq in make_experiments(arch.insn_list, 5):
        assert procs_agree(iseq, proc, simple_proc3)

def test_bn_handler_regression01():
    arch = Architecture()
    arch.add_insns(['I_0', 'I_1'])
    i0, i1 = arch.insn_list

    pm = Mapping3(arch)
    pm.assignment[i0] = [[1], [1], [2]]
    pm.assignment[i1] = [[0], [0]]

    pm_proc = Processor(config=pm)

    smt_proc = SynthWrapperProc(pm, {
            "synthesizer_kind": "smt",
            "mapping_class": "Mapping3",
            "num_ports": 4,
            "num_uops": 4,
            "smt_slack_val": 0.0,
            "smt_use_bn_handler": True,
            "smt_insn_bound": 3,
            "smt_exp_limit_strategy": "incremental_bounded",
        })

    iseq = [i0] * 7 + [i1] * 3
    assert procs_agree(iseq, pm_proc, smt_proc)

    assert smt_proc.get_cycles(iseq) == 14

