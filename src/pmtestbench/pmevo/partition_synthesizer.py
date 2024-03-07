""" A Synthesizer wrapper that performs inference on a restricted set of
instructions and experiments. The instructions are restricted to the
representatives of a partitioning into sets of instructions that are
indistinguishable wrt. the given set of experiments. Experiments are expected
to include a singleton experiment for each instruction and exhaustive
experiments for all pairs of instructions.
"""

from iwho.configurable import load_json_config

from ..common.synthesizers import SynthesizerImpl, Synthesizer
from ..common.experiments import ExperimentList
from .partitioning import compute_representatives, restrict_elist, generalize_mapping

import logging
logger = logging.getLogger(__name__)

class PartitionSynthesizer(SynthesizerImpl):

    def __init__(self, config):
        self.equivalence_epsilon = config['equivalence_epsilon']
        wrapped_config = config.get('wrapped_config', None)
        if wrapped_config is None:
            config_path = config.get('wrapped_config_path', None)
            if config_path is None:
                raise RuntimeError("Neither wrapped_config nor wrapped_config_path given in partition synthesizer config!")
            wrapped_config = load_json_config(config_path)

        self.wrapped_synthesizer = Synthesizer(wrapped_config)


    def infer(self, exps):
        old_exps = exps
        old_arch = old_exps.arch

        singleton_exps = ExperimentList(exps.arch)
        singleton_exps.exps = [e for e in exps if len(e.iseq) == 1]

        complex_exps = ExperimentList(exps.arch)
        complex_exps.exps = [e for e in exps if len(e.iseq) > 1]

        reps, insn_to_rep = compute_representatives(complex_exps, singleton_exps, epsilon=self.equivalence_epsilon)

        exps = restrict_elist(exps, reps)
        logger.info("Restricted input to {insns} out of {old_insns} instructions and {exps} out of {old_exps} experiments.".format(
                insns=len(reps),
                old_insns=len(old_arch.insn_list),
                exps=len(exps.exps),
                old_exps=len(old_exps.exps)
            ))

        mapping = self.wrapped_synthesizer.infer(exps)

        if mapping is None:
            return None

        mapping = generalize_mapping(old_arch, mapping, insn_to_rep)

        return mapping


    def synthesize(self, proc, exps):
        raise RuntimeError("Synthesis is not supported by this synthesizer!")
