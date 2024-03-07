""" An interface for port mapping inference methods.
"""

from abc import ABC, abstractmethod
from typing import *
from pathlib import Path

from iwho.configurable import ConfigMeta

from .experiments import Experiment, ExperimentList
from .architecture import Architecture
from .portmapping import Mapping


class Synthesizer(metaclass=ConfigMeta):
    config_options = dict(
        synthesizer_kind = ('smt',
            'the kind of synthesizer to use. One of "smt", "pmevo", "partition", ...'), #TODO
        add_singleton_exps = (False,
            'if True: add and evaluate experiments for all individual instructions before synthesis'),
        mapping_class = ('Mapping3',
            'the kind of port mapping to find. One of "Mapping2", "Mapping3"'),
        num_ports = (8,
            'number of ports to assume'),
        smt_slack_val = (0.1,
            'for smt synthesizer: allowed deviation from measurements'),
        smt_slack_kind = ('cpi',
            'for smt synthesizer: how to interpret the slack value. Options: "absolute" (on cycles per iteration), "cpi" (on the cycles per instruction), "cycle-relative" (relative to the simulated cycles). Default: "cpi"'),
        smt_insn_bound = (3,
            'for smt synthesizer: maximal number of instructions to consider in counter examples'),
        smt_exp_limit_strategy = ("incremental_bounded",
            'for smt synthesizer: strategy for choosing the experiment size limit. Options: "incremental_bounded", "unbounded", "incremental_optimistic"'),
        smt_dump_constraints = (False,
            'for smt synthesizer: if true, dump relevant constraint sets in SMT-LIB2 format in the working directory'),
        smt_use_bn_handler = (False,
            'for smt synthesizer: if true, use the BN MappingHandler rather than the default one'),
        smt_bn_handler_bits = (3,
            'for smt synthesizer with BN MappingHandler: use this many bits to bit-blast the "number of instruction instances in experiment" * "number of uops for instruction" multiplication'),
        smt_use_full_mul_handler = (False,
            'for smt synthesizer: if true, use the Full Multiply MappingHandler rather than the default one'),
        smt_full_mul_uopsize_limit = (-1,
            'for smt synthesizer with Full Multiply MappingHandler: uops may use at most this many ports (default: no limit)'),
        num_uops = (3,
            'for smt Mapping3 synthesizer: maximal number of uops to use per instruction'),
        smt_use_constrained_mapping3 = (False,
            'for smt synthesizer: if true, use the constrained Mapping3 Handler'),
        smt_constrain_improper_uops = (False,
            'for constraint Mapping3 SMT synthesizer: add additional constraints on the nature of non-first uops. These constraints rule out general applicability.'),
        num_uops_per_insn = ({},
            'for constrained smt Mapping3 synthesizer: maximal number of uops to use per instruction. Default: 1'),
        pmevo_bin_path = (None,
            'for pmevo synthesizer: path to the PMEvo binary to use'),
        pmevo_config_path = (None,
            'for pmevo synthesizer: PMEvo run config to use'),
        pmevo_temp_directory = ("/tmp/",
            'for pmevo synthesizer: directory to use for logging and temporary files'),
        wrapped_config = (None,
            'for partition synthesizer: another synthesizer config dictionary, for the synthesizer that it wraps'),
        wrapped_config_path = (None,
            'for partition synthesizer: a path to another synthesizer config, for the synthesizer that it wraps'),
        equivalence_epsilon = (0.1,
            'for partition synthesizer: allowed relative deviation for two measurements to be considered equal'),
    )

    def __init__(self, config):
        self.configure(config)

        if self.synthesizer_kind == 'smt':
            from ..cegpmi.smt_synthesizer import SMTSynthesizer
            self.impl = SMTSynthesizer(self.get_config())
        elif self.synthesizer_kind == 'pmevo':
            from ..pmevo.pmevo_synthesizer import PMEvoSynthesizer
            self.impl = PMEvoSynthesizer(self.get_config())
        elif self.synthesizer_kind == 'partition':
            from ..pmevo.partition_synthesizer import PartitionSynthesizer
            self.impl = PartitionSynthesizer(self.get_config())
        else:
            raise NotImplementedError(f"synthesizer kind: {self.synthesizer_kind}")

    def synthesize(self, proc, *, exps=None, **kwargs):
        """
        """
        if exps is None:
            exps = ExperimentList(arch=proc.get_arch())

        if self.add_singleton_exps:
            for i in proc.get_arch().insn_list:
                proc.eval(exps.create_exp([i]))

        return self.impl.synthesize(proc, exps, **kwargs)

    def infer(self, exps, **kwargs):
        """
        """
        return self.impl.infer(exps, **kwargs)



class SynthesizerImpl(ABC):

    @abstractmethod
    def synthesize(self, proc, exps, **kwargs):
        """
        """
        pass

    @abstractmethod
    def infer(self, exps, **kwargs):
        """
        """
        pass



