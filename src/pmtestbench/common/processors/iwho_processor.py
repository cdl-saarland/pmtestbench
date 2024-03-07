""" A processor using the IWHO predictor interface.

IWHO predictors are not limited to port-mapping-bound microbenchmarks, data
dependencies are relevant for them. This module provides startegies to allocate
operands for the instructions in an experiment such that data dependencies are
avoided and the experiment can be handed to an iwho predictor.

The strategy is described in Chapter 3 of the Dissertation "Inferring and
Analyzing Microarchitectural Performance Models" by Fabian Ritter.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
import math
import numbers
from typing import *
import os
import random
import sys

from . import ProcessorImpl

from ..architecture import Architecture

from iwho.configurable import load_json_config
from iwho.predictors import Predictor, get_sudo
import iwho
import iwho.x86 as x86


import logging
logger = logging.getLogger(__name__)

class OperandAllocationInfo(ABC):

    @abstractmethod
    def get_considered_operands(self, op_scheme):
        """ Given an OperandScheme, get a list of tuples of operands that can
        be used for operand allocation and their alias class. This is
        architecture specific and needs to be provided by a subclass.
        """
        pass

    @abstractmethod
    def needs_tracking(self, operand):
        """ Return True if a writing timestamp for the operand needs to be
        tracked. That is usually the case if it can be written. This is
        architecture specific and needs to be provided by a subclass.
        """
        pass

    @abstractmethod
    def get_alias_class(self, operand):
        """ Get a unique identifier for all possible considered operands that
        alias with operand.
        """
        pass


class X86OperandAllocationInfo(OperandAllocationInfo):
    def __init__(self, iwho_ctx):
        self.iwho_ctx = iwho_ctx

        # we use defaults for the boring operands
        self.instor = iwho.x86.DefaultInstantiator(self.iwho_ctx)

        self.forbidden_alias_classes = {
                # x86.RegAliasClass.GPR_BP,  # reserved for memory
                # x86.RegAliasClass.GPR_SI,  # reserved for memory
                # x86.RegAliasClass.GPR_DI,  # reserved for memory
                x86.RegAliasClass.GPR_R15, # reserved for loop counter
                x86.RegAliasClass.GPR_R14, # reserved for memory
                # x86.RegAliasClass.GPR_C,   # reserved for shift amounts
                x86.RegAliasClass.GPR_SP,  # try to not mess up the stack pointer
            }
        # base = x86.all_registers["rbp"]
        base = x86.all_registers["r14"]

        num_mem_ops = 62 # 32 + 62 * 64 = 4000 fits within a page
        self.allowed_memops = [
                self.iwho_ctx.dedup_store.get(x86.MemoryOperand, base=base, displacement=disp, width=64)
                for disp in range(32, 32 + num_mem_ops * 64, 64)
            ]

        mem_constraint = self.iwho_ctx.dedup_store.get(
                x86.MemConstraint, unhashed_kwargs={"context": self.iwho_ctx}, width=64)

        self.normalized_mem_scheme = self.iwho_ctx.dedup_store.get(iwho.OperandScheme, constraint=mem_constraint)

        self.considered_operands_cache = dict()

    def get_considered_operands(self, op_scheme):
        cached = self.considered_operands_cache.get(op_scheme, None)
        if cached is not None:
            return cached

        if op_scheme.is_fixed():
            res = [ (op_scheme.fixed_operand, self.get_alias_class(op_scheme.fixed_operand)) ]
            self.considered_operands_cache[op_scheme] = res
            return res
        constraint = op_scheme.operand_constraint
        if isinstance(constraint, x86.RegisterConstraint):
            candidates = [ o for o in constraint.acceptable_operands if o.alias_class not in self.forbidden_alias_classes ]
            candidates.sort(key=str)
            res = candidates
        elif isinstance(constraint, x86.MemConstraint):
            res = [ self.iwho_ctx.adjust_operand(o, op_scheme) for o in self.allowed_memops ]
        else:
            assert (isinstance(constraint, x86.ImmConstraint)
                    or isinstance(constraint, x86.SymbolConstraint)), f"Unsupported constraint type: {type(constraint)}!"
            res = [self.instor(op_scheme)]

        res = [(op, self.get_alias_class(op)) for op in res]
        self.considered_operands_cache[op_scheme] = res
        return res

    def needs_tracking(self, operand):
        return isinstance(operand, x86.RegisterOperand) or isinstance(operand, x86.MemoryOperand)

    def get_alias_class(self, operand):
        if isinstance(operand, x86.RegisterOperand):
            res = operand.alias_class
        elif isinstance(operand, x86.MemoryOperand):
            res = self.iwho_ctx.adjust_operand(operand, self.normalized_mem_scheme)
        else:
            res = None
        return res


class OperandAllocator(ABC):
    """ Abstract base class to represent strategies of instantiating a list of
    instruction schemes with operands such that data dependencies are avoided.

    Subclasses need to implement several target-specific abstract methods,
    which are used by `OperandAllocator.allocate_operands()` for the actual
    instantiation.
    """

    @staticmethod
    def get(iwho_ctx, kind):
        if isinstance(iwho_ctx, iwho.x86.Context):
            alloc_info = X86OperandAllocationInfo(iwho_ctx)
        else:
            raise NotImplementedError(f"Cannot handle iwho context of type {type(iwho_ctx)}")

        if kind == "partitioned":
            return PartitionedOperandAllocator(iwho_ctx, alloc_info)
        if kind == "wmrr":
            return WMRROperandAllocator(iwho_ctx, alloc_info)
        if kind == "write-same":
            return WriteSameOperandAllocator(iwho_ctx, alloc_info)
        if kind == "random":
            return RandomOperandAllocator(iwho_ctx, alloc_info)
        raise NotImplementedError(f"Unknown operand allocator: '{kind}'")


    def __init__(self, iwho_ctx, alloc_info):
        self.iwho_ctx = iwho_ctx
        self.alloc_info = alloc_info

    @abstractmethod
    def allocate_operands(self, schemes):
        """ This is the entry point method for the operand allocator to
        instantiate the provided list of instruction schemes.
        Returns a iwho.BasicBlock with the produced instruction instances.
        """
        pass


class PartitionedOperandAllocator(OperandAllocator):
    """ Partitions the available operands into groups that are used exclusively
    for read operands, exclusively for written operands, or exclusively
    read-and-written operands.
    """

    def __init__(self, iwho_ctx, alloc_info):
        assert isinstance(iwho_ctx, iwho.x86.Context), "The PartitioningOperandAllocator only supports the x86 ISA!"

        super().__init__(iwho_ctx, alloc_info)

        mem_base = x86.all_registers["r14"]

        def mem_op(displacement_idx):
            disp = 32 + displacement_idx * 64
            return self.iwho_ctx.dedup_store.get(x86.MemoryOperand, base=mem_base, displacement=disp, width=64)

        self.class_written = {
                x86.RegAliasClass.GPR_B,
                x86.RegAliasClass.vMM0,
                mem_op(0),
            }
        self.class_read = {
                x86.RegAliasClass.GPR_C,
                x86.RegAliasClass.GPR_R12,
                x86.RegAliasClass.vMM1,
                x86.RegAliasClass.vMM2,
                x86.RegAliasClass.vMM3,
                mem_op(1),
            }

        self.class_read_or_written = set()
        self.class_read_or_written.update(self.class_read)
        self.class_read_or_written.update(self.class_written)

    def allocate_operands(self, schemes):
        alloc_info = self.alloc_info
        insn_list = []

        last_used = defaultdict(int)

        time = 1
        for s in schemes:
            operands = dict()
            for k, op_scheme in s.explicit_operands.items():
                considered_operands = self.alloc_info.get_considered_operands(op_scheme)

                if len(considered_operands) > 1:
                    if op_scheme.is_read and op_scheme.is_written:
                        filter_fun = lambda op_ac: (op_ac[1] not in self.class_read_or_written)
                    elif op_scheme.is_written:
                        filter_fun = lambda op_ac: (op_ac[1] in self.class_written)
                    elif op_scheme.is_read:
                        filter_fun = lambda op_ac: (op_ac[1] in self.class_read)
                    else:
                        # this concerns immediate and symbol operands
                        filter_fun = lambda op_ac: True

                    considered_operands = list(filter(filter_fun, considered_operands))
                    considered_operands.sort(key=lambda op_ac: last_used[op_ac[1]])

                try:
                    chosen_operand, chosen_alias_class = considered_operands[0]
                except:
                    logger.error(f"OperandScheme with missing operand: {op_scheme}")
                    raise
                operands[k] = chosen_operand
                last_used[chosen_alias_class] = time
                time += 1

            insn = s.instantiate(operands)
            insn_list.append(insn)

        return self.iwho_ctx.make_bb(insn_list)

class WMRROperandAllocator(OperandAllocator):
    """ 'Write Most Recently Read' Operand allocator
    """

    def __init__(self, iwho_ctx, alloc_info):
        super().__init__(iwho_ctx, alloc_info)

    def get_last_written(self, operand):
        ac = self.alloc_info.get_alias_class(operand)
        return self._last_written[ac]

    def get_last_written_or_read(self, operand):
        """ Return the timestamp when the operand has been last written or
        read after a write, whichever happened last.
        """
        ac = self.alloc_info.get_alias_class(operand)
        last_read = self._last_read[ac]
        if last_read == -1:
            return self._last_written[ac]
        else:
            return last_read

    def register_write(self, operand):
        ac = self.alloc_info.get_alias_class(operand)
        self._last_read.clear() # last read counters only need to count since the last write, so reset them
        self._last_written[ac] = self._curr_timestamp
        self._curr_timestamp += 1

    def register_read(self, operand):
        ac = self.alloc_info.get_alias_class(operand)
        self._last_read[ac] = self._curr_timestamp
        self._curr_timestamp += 1

    def reset_last_written(self):
        # A mapping of alias classes to the timestamp when they have last been
        # written. (-1) means that they have not been written at all
        self._last_written = defaultdict(lambda: -1)

        self._last_read = defaultdict(lambda: -1)

        # the current (abstract) timestamp used for the last_written dict
        self._curr_timestamp = 0

    def allocate_operands(self, schemes):
        """ This is the entry point method for the operand allocator to
        instantiate the provided list of instruction schemes.
        Returns a iwho.BasicBlock with the produced instruction instances.
        """
        insn_list = []

        alloc_info = self.alloc_info

        self.reset_last_written()

        for s in schemes:
            op_schemes = s.explicit_operands
            operands = dict()

            written_schemes = []
            remaining_schemes = []
            for k, op_scheme in op_schemes.items():
                if op_scheme.is_written:
                    written_schemes.append((k, op_scheme))
                else:
                    remaining_schemes.append((k, op_scheme))

            for k, op_scheme in written_schemes:
                # of the allowed operands, take the one that has been written
                # least recently
                considered_operands = alloc_info.get_considered_operands(op_scheme)
                considered_operands = [op for (op, ac) in considered_operands]

                assert len(considered_operands) > 0

                if len(considered_operands) > 1:
                    # sort by ascending timestamp of last use, i.e. least recently used first
                    considered_operands.sort(key=lambda x: self.get_last_written(x))

                op_instance = considered_operands[0]

                if alloc_info.needs_tracking(op_instance):
                    self.register_write(op_instance)

                operands[k] = op_instance

            for k, op_scheme in remaining_schemes:
                # of the allowed operands that haven't been used for reading in
                # this instruction, take the one that has been written least
                # recently
                considered_operands = alloc_info.get_considered_operands(op_scheme)
                considered_operands = [op for (op, ac) in considered_operands]

                assert len(considered_operands) > 0

                if len(considered_operands) > 1:
                    # sort by ascending timestamp of last use, i.e. least recently used first
                    considered_operands.sort(key=lambda x: self.get_last_written_or_read(x))

                op_instance = considered_operands[0]

                if alloc_info.needs_tracking(op_instance):
                    self.register_read(op_instance)
                operands[k] = op_instance

            insn = s.instantiate(operands)
            insn_list.append(insn)

        return self.iwho_ctx.make_bb(insn_list)


class WriteSameOperandAllocator(OperandAllocator):
    """ Operand allocator that uses the same operand for each written operand,
    as far as possible. Not very reasonable when instructions are involved that
    read and write the same operand.
    """
    def __init__(self, iwho_ctx, alloc_info):
        super().__init__(iwho_ctx, alloc_info)

    def get_last_written_or_read(self, operand):
        """ Return the timestamp when the operand has been last written or
        read after a write, whichever happened last.
        """
        ac = self.alloc_info.get_alias_class(operand)
        last_read = self._last_read[ac]
        if last_read == -1:
            return self._last_written[ac]
        else:
            return last_read

    def register_write(self, operand):
        ac = self.alloc_info.get_alias_class(operand)
        self._last_read.clear() # last read counters only need to count since the last write, so reset them
        self._last_written[ac] = self._curr_timestamp
        self._curr_timestamp += 1

    def register_read(self, operand):
        ac = self.alloc_info.get_alias_class(operand)
        self._last_read[ac] = self._curr_timestamp
        self._curr_timestamp += 1

    def reset_last_written(self):
        # A mapping of alias classes to the timestamp when they have last been
        # written. (-1) means that they have not been written at all
        self._last_written = defaultdict(lambda: -1)

        self._last_read = defaultdict(lambda: -1)

        # the current (abstract) timestamp used for the last_written dict
        self._curr_timestamp = 0

    def allocate_operands(self, schemes):
        # this method is more complicated than it needs to be
        insn_list = []

        alloc_info = self.alloc_info

        self.reset_last_written()

        for s in schemes:
            op_schemes = s.explicit_operands
            operands = dict()

            written_schemes = []
            remaining_schemes = []
            for k, op_scheme in op_schemes.items():
                if op_scheme.is_written:
                    written_schemes.append((k, op_scheme))
                else:
                    remaining_schemes.append((k, op_scheme))

            for k, op_scheme in written_schemes:
                # of the allowed operands, take the one that has been written
                # least recently
                considered_operands = alloc_info.get_considered_operands(op_scheme)
                considered_operands = [op for (op, ac) in considered_operands]

                assert len(considered_operands) > 0

                op_instance = considered_operands[0]

                if alloc_info.needs_tracking(op_instance):
                    self.register_write(op_instance)

                operands[k] = op_instance

            for k, op_scheme in remaining_schemes:
                # of the allowed operands that haven't been used for reading in
                # this instruction, take the one that has been written least
                # recently
                considered_operands = alloc_info.get_considered_operands(op_scheme)

                assert len(considered_operands) > 0

                if len(considered_operands) > 1:
                    # sort by ascending timestamp of last use, i.e. least recently used first
                    considered_operands.sort(key=lambda x: self.get_last_written_or_read(x))

                op_instance = considered_operands[0]

                if alloc_info.needs_tracking(op_instance):
                    self.register_read(op_instance)
                operands[k] = op_instance

            insn = s.instantiate(operands)
            insn_list.append(insn)

        return self.iwho_ctx.make_bb(insn_list)


class RandomOperandAllocator(OperandAllocator):

    def __init__(self, iwho_ctx, alloc_info):
        super().__init__(iwho_ctx, alloc_info)

    def allocate_operands(self, schemes):
        insn_list = []

        for s in schemes:
            operands = dict()
            for k, op_scheme in s.explicit_operands.items():
                considered_operands = self.alloc_info.get_considered_operands(op_scheme)
                considered_operands = [op for (op, ac) in considered_operands]
                chosen_operand = random.choice(considered_operands)
                operands[k] = chosen_operand

            insn = s.instantiate(operands)
            insn_list.append(insn)

        return self.iwho_ctx.make_bb(insn_list)


def translate_zenp_macro_ops_to_uops(in_bb, unroll_factor, res):
    """ In the result dict `res` for executing the basic block `in_bb`, modify
    the uop counter according to the following rules:
      - for each memory operand with a width of 256 bits, add 2 uops (since
        those are double-pumped and require two load/store operations)
      - for each memory operand with less than 256 bits of width that is not
        part of a direct memory-to-register or register-to-memory "mov"
        instruction, add 1 uop. UPDATE: actually, according to measurements,
        register-to-memory "mov" instructions also require an additional uop.
        See the paper/thesis for a discussion.

    This adjustment is motivated by the apparent mismatch in the official AMD
    documentation and the results of the uop-counting performance counter. A
    likely explanation is that the performance counter actually counts
    macro-ops rather than uops (meaning that the Processor Programming
    Reference is wrong). According to the "Software Optimization Guide for AMD
    Family 17h Processors", these "can normally contain up to two (2) micro
    ops."
    We generalize these rules from examples shown in Table 1 ("Typical
    Instruction Mappings") of the Optimization Guide.
    This table seems to be wrong for the storing "mov" instruction, however.
    """

    if res['cycles'] <= 0.0:
        return res

    ctx = in_bb.context

    counter_increase = 0
    for i in in_bb:
        if ctx.extract_mnemonic(i) == "mov":
            mem_scheme = i.scheme.get_operand_scheme((iwho.InsnScheme.OperandKind.EXPLICIT, 'mem0'))
            reg_scheme = i.scheme.get_operand_scheme((iwho.InsnScheme.OperandKind.EXPLICIT, 'reg0'))

            if (mem_scheme is not None) and (not mem_scheme.is_written) and (reg_scheme is not None):
                # this is a simple loading mov operation that is implemented
                # with only one uop. movs with immediates are intentionally not
                # included since they require two uops according to the
                # Optimization Manual.
                # movs that store to memory are not included even though the
                # Optimization Manual suggests otherwise since our experiments
                # indicate that they need to have two uops.
                continue

        if ctx.extract_mnemonic(i) == "lea":
            # lea instructions are (probably) implemented with only one uop,
            # since they do not access memory. This is somewhat backed by the
            # AMD instruction table.
            continue

        for op_instance, (key, op_scheme) in i.get_operands():
            if isinstance(op_instance, x86.MemoryOperand):
                if op_instance.width == 256:
                    # double-pumped => two mem ops
                    counter_increase += 2
                else:
                    # width > 256 should not really occur since Zen+ does not
                    # support AVX512. An exception are fxsave/rstor
                    # instructions, whose schemes note a memory operand with
                    # 4096 bytes width. But for those, we probably cannot hope
                    # to infer a reasonable port mapping anyway.
                    counter_increase += 1

                    # if op_scheme.is_read and op_scheme.is_written:
                    #     # According to the docs, R/W memory operands only cause
                    #     # a single uop, but that wouldn't be the first error in
                    #     # there...
                    #     counter_increase += 1


    assert 'num_uops' in res, "Adjusting uop counters in a setting where uops counters are not read!"

    assert counter_increase % unroll_factor == 0, "encountered a number of memory operations that is not a multiple of the unroll_factor"

    normalized_counter_increase = counter_increase // unroll_factor

    res['num_uops'] += normalized_counter_increase

    res['num_uops_increased_by'] = normalized_counter_increase

    return


class IWHOProcessor(ProcessorImpl):

    def __init__(self, config):
        config_path = config['iwho_config_path']
        if config_path is not None:
            if isinstance(config_path, dict):
                # embedded config
                iwhoconfig = config_path
            else:
                iwhoconfig = load_json_config(config_path)
        else:
            iwhoconfig = {} # use the defaults
        self.iwho_ctx = iwho.Config(config=iwhoconfig).context

        insn_schemes = list(map(str, self.iwho_ctx.filtered_insn_schemes))

        self.arch = Architecture()
        self.arch.add_insns(insn_schemes)

        self.predictor = Predictor.get(config['iwho_predictor_config'])

        if self.predictor.requires_sudo():
            get_sudo()

        self.unroll_factors = config['iwho_unroll_factors']

        self.unroll_mode = config['iwho_unroll_mode']
        if self.unroll_mode not in ['simple', 'instructions']:
            raise NotImplementedError(f"Unknown unroll mode: '{self.unroll_mode}'")

        self.operand_allocator_kinds = config['iwho_operand_allocators']

        # A list of functions (type: (basic block, unroll_factor, resultdict)
        # -> resultdict) that are applied to the results when executing an
        # experiment. Ideally, this functionality would be unnecessary. It
        # provides a means of correcting wrong results from measurements. The
        # specific reason for the existence of this hack is that the uops
        # counters on Zen+ apparently are wrong for instructions with memory
        # operands: Simple such instructions are reported to use only one uop
        # while the optimization manual, common sense, and run-time
        # measurements suggest that they use two uops.
        self.result_converters = []

        if config.get('iwho_adjust_zenp_uopcount', False):
            self.result_converters.append(translate_zenp_macro_ops_to_uops)


    def get_arch(self) -> Architecture:
        return self.arch


    def concretize_bb(self, iseq, unroll_factor, operand_allocator):
        """ Given a sequence of instruction schemes, create a basic block,
        i.e., a sequence of instruction instances with operands instantiated to
        avoid data dependencies.

        This includes unrolling by unroll_factor before selecting
        operands, to enable more independence.
        """
        # get the right iwho insn schemes
        schemes = []
        for i in iseq:
            schemes.append(self.iwho_ctx.str_to_scheme[i])

        # unroll
        unrolled_schemes = []
        for x in range(unroll_factor):
            unrolled_schemes += schemes

        return operand_allocator.allocate_operands(unrolled_schemes)


    def execute(self, iseq: List[str], *args, excessive_log=None, **kwargs) -> Dict[str, Union[float, str]]:
        all_results = []
        for unroll_factor in self.unroll_factors:
            if self.unroll_mode == 'instructions':
                # compute the minimal number of copies of iseq required to get
                # at least unroll_factor instructions
                unroll_factor = math.ceil(unroll_factor / len(iseq))
            else:
                assert self.unroll_mode == 'simple'
            for op_alloc_id, op_alloc_kind in enumerate(self.operand_allocator_kinds):
                operand_allocator = OperandAllocator.get(self.iwho_ctx, op_alloc_kind)

                try:
                    bb = self.concretize_bb(iseq, unroll_factor, operand_allocator)
                except:
                    return {
                            'cycles': -1.0,
                            'message': "Concretization Failed!",
                        }

                local_excessive_log = None
                if excessive_log is not None:
                    local_excessive_log = []

                # call the predictor
                iwho_res = self.predictor.evaluate(bb, *args, excessive_log=local_excessive_log, **kwargs)

                if excessive_log is not None:
                    for entry in local_excessive_log:
                        entry['unroll_factor'] = unroll_factor
                        entry['op_alloc_id'] = op_alloc_id
                        entry['op_alloc_kind'] = op_alloc_kind
                        excessive_log.append(entry)

                res = {}

                failed = False
                if iwho_res.get('TP', -1.0) <= 0:
                    res['cycles'] = -1.0
                    failed = True
                else:
                    cycles_per_copy = iwho_res['TP'] / unroll_factor
                    res['cycles'] = cycles_per_copy

                for k, v in iwho_res.items():
                    if k not in ['TP', 'rt', 'unroll_factor'] and isinstance(v, numbers.Number):
                        res[k] = v / unroll_factor
                    else:
                        res[k] = v

                # apply the result converter functions
                for fun in self.result_converters:
                    fun(bb, unroll_factor, res)

                all_results.append(res)

        res = min(all_results, key=lambda x: x['cycles'])
        return res

