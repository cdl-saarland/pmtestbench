#!/usr/bin/env pytest

import pytest

import os
import sys

import_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.append(import_path)

import pmtestbench.relaxeduops.core as ru

import logging

def test_secondary_uops_interfere_01(caplog):
    caplog.set_level(logging.DEBUG)
    secondary_uops_for_blocking_insn = [ frozenset({1, 2})]
    subsequent_blocking_insns = [ 'b' ]
    ports_for = {'b': frozenset({0, 1})}
    blocked_ports = frozenset({0})
    port_usage = {frozenset({1, 2}): 1}

    num_uops = 2
    num_of_blk_insns = 2

    width = len(blocked_ports)
    num_uops_characterized = sum([n for k, n in port_usage.items()])
    min_cycles = num_of_blk_insns / width

    res = ru.secondary_uops_interfere(
            secondary_uops_for_blocking_insn = secondary_uops_for_blocking_insn,
            subsequent_blocking_insns = subsequent_blocking_insns,
            ports_for = ports_for,
            blocked_ports = blocked_ports,
            port_usage = port_usage,
            num_uops = num_uops,
            num_uops_characterized = num_uops_characterized,
            min_cycles = min_cycles,
            num_of_blk_insns = num_of_blk_insns,
        )

    assert res == False


def test_secondary_uops_interfere_02(caplog):
    caplog.set_level(logging.DEBUG)
    secondary_uops_for_blocking_insn = [ frozenset({1, 2})]
    subsequent_blocking_insns = [ 'b' ]
    ports_for = {'b': frozenset({0, 1})}
    blocked_ports = frozenset({0})
    port_usage = {frozenset({1, 2}): 1}

    num_uops = 2
    num_of_blk_insns = 10

    width = len(blocked_ports)
    num_uops_characterized = sum([n for k, n in port_usage.items()])
    min_cycles = num_of_blk_insns / width

    res = ru.secondary_uops_interfere(
            secondary_uops_for_blocking_insn = secondary_uops_for_blocking_insn,
            subsequent_blocking_insns = subsequent_blocking_insns,
            ports_for = ports_for,
            blocked_ports = blocked_ports,
            port_usage = port_usage,
            num_uops = num_uops,
            num_uops_characterized = num_uops_characterized,
            min_cycles = min_cycles,
            num_of_blk_insns = num_of_blk_insns,
        )

    assert res == False

def test_secondary_uops_interfere_03(caplog):
    caplog.set_level(logging.DEBUG)
    secondary_uops_for_blocking_insn = [ frozenset({1, 2})]
    subsequent_blocking_insns = [ 'b' ]
    ports_for = {'b': frozenset({0, 1})}
    blocked_ports = frozenset({0})
    port_usage = {frozenset({1, 2}): 1}

    num_uops = 4
    num_of_blk_insns = 3

    width = len(blocked_ports)
    num_uops_characterized = sum([n for k, n in port_usage.items()])
    min_cycles = num_of_blk_insns / width

    res = ru.secondary_uops_interfere(
            secondary_uops_for_blocking_insn = secondary_uops_for_blocking_insn,
            subsequent_blocking_insns = subsequent_blocking_insns,
            ports_for = ports_for,
            blocked_ports = blocked_ports,
            port_usage = port_usage,
            num_uops = num_uops,
            num_uops_characterized = num_uops_characterized,
            min_cycles = min_cycles,
            num_of_blk_insns = num_of_blk_insns,
        )

    assert res == True

