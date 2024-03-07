#!/usr/bin/env pytest

import pytest

import os
import sys

import_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.append(import_path)

import iwho
import iwho.x86 as x86

from pmtestbench.common.processors import Processor

import pmtestbench.common.processors.iwho_processor as ip

proc_config = {
        "processor_kind": "iwho",
        "iwho_config_path": os.path.join(os.path.dirname(__file__), "resources", "iwhocfg_x86_full.json"),
        "iwho_predictor_config": {
            "kind": "test",
            "mnemonic_costs": {"add": 2.0}
        },
        "iwho_unroll_factors": [8],
    }

def check_not_recently_written(bb, num):
    for idx, insn in enumerate(bb):
        for k, opscheme in insn.scheme.explicit_operands.items():
            if opscheme.is_read:
                operand = insn.get_operand((iwho.InsnScheme.OperandKind.EXPLICIT, k))
                for j in range(max(0, idx - num), idx):
                    prev_insn = bb[j]
                    for prev_operand, (k2, prev_opscheme) in prev_insn.get_operands():
                        if prev_opscheme.is_written:
                            assert not bb.context.must_alias(operand, prev_operand), (
                                    f"operand {k} ({operand}) of instruction {insn} ({idx}) may alias with\n" +
                                    f"operand {k2} ({prev_operand}) of instruction {prev_insn} ({j})")
                            # semantically, we would want may_alias here, expanded with knowledge about unchanging memory operands

# op_alloc_kind = "wmrr"
op_alloc_kind = "partitioned"

def test_reg_alloc_64():
    proc = Processor(proc_config)

    unroll_factor = proc_config['iwho_unroll_factors'][0]
    operand_allocator = ip.OperandAllocator.get(proc.impl.iwho_ctx, op_alloc_kind)

    iseq = [
        "add RW:GPR:64, R:GPR:64",
        "adc RW:GPR:64, R:GPR:64",
        ]

    bb = proc.impl.concretize_bb(iseq, unroll_factor, operand_allocator)

    print(bb)

    assert len(bb) == len(iseq) * unroll_factor

    check_not_recently_written(bb, 5)


def test_reg_alloc_64_32():
    proc = Processor(proc_config)

    unroll_factor = proc_config['iwho_unroll_factors'][0]
    operand_allocator = ip.OperandAllocator.get(proc.impl.iwho_ctx, op_alloc_kind)

    iseq = [
        "add RW:GPR:64, R:GPR:64",
        "adc RW:GPR:32, R:GPR:32",
        ]

    bb = proc.impl.concretize_bb(iseq, unroll_factor, operand_allocator)

    print(bb)

    assert len(bb) == len(iseq) * unroll_factor

    check_not_recently_written(bb, 5)


def test_reg_alloc_mem_read():
    proc = Processor(proc_config)

    unroll_factor = proc_config['iwho_unroll_factors'][0]
    operand_allocator = ip.OperandAllocator.get(proc.impl.iwho_ctx, op_alloc_kind)

    iseq = [
        "add RW:GPR:64, qword ptr R:MEM(64)",
        "adc RW:GPR:64, R:GPR:64",
        ]

    bb = proc.impl.concretize_bb(iseq, unroll_factor, operand_allocator)

    print(bb)

    assert len(bb) == len(iseq) * unroll_factor

    check_not_recently_written(bb, 5)


def test_reg_alloc_mem():
    proc = Processor(proc_config)

    unroll_factor = proc_config['iwho_unroll_factors'][0]
    operand_allocator = ip.OperandAllocator.get(proc.impl.iwho_ctx, op_alloc_kind)

    iseq = [
        "add RW:GPR:64, qword ptr R:MEM(64)",
        "adc qword ptr RW:MEM(64), R:GPR:64",
        ]

    bb = proc.impl.concretize_bb(iseq, unroll_factor, operand_allocator)

    print(bb)

    assert len(bb) == len(iseq) * unroll_factor

    check_not_recently_written(bb, 5)


def test_reg_alloc_ymm():
    proc = Processor(proc_config)

    unroll_factor = proc_config['iwho_unroll_factors'][0]
    operand_allocator = ip.OperandAllocator.get(proc.impl.iwho_ctx, op_alloc_kind)

    iseq = [
        "vaddpd W:YMM0..15, R:YMM0..15, R:YMM0..15",
        "vaddps W:YMM0..15, R:YMM0..15, R:YMM0..15",
        ]

    bb = proc.impl.concretize_bb(iseq, unroll_factor, operand_allocator)

    print(bb)

    assert len(bb) == len(iseq) * unroll_factor

    check_not_recently_written(bb, 5)


def test_reg_alloc_mixed():
    proc = Processor(proc_config)

    unroll_factor = proc_config['iwho_unroll_factors'][0]
    operand_allocator = ip.OperandAllocator.get(proc.impl.iwho_ctx, op_alloc_kind)

    iseq = [
        "add RW:GPR:64, qword ptr R:MEM(64)",
        "vaddpd W:YMM0..15, R:YMM0..15, R:YMM0..15",
        "add RW:GPR:64, R:GPR:64",
        "vaddps W:YMM0..15, R:YMM0..15, R:YMM0..15",
        "add RW:GPR:64, qword ptr R:MEM(64)",
        "adc RW:GPR:32, R:GPR:32",
        ]

    bb = proc.impl.concretize_bb(iseq, unroll_factor, operand_allocator)

    print(bb)

    assert len(bb) == len(iseq) * unroll_factor

    check_not_recently_written(bb, 5)

