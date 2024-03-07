""" Processor implementations that use plain port mappings for throughput
estimations.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from fractions import Fraction
from typing import *
import os
import sys

from . import ProcessorImpl
from ..portmapping import Mapping, Mapping2, Mapping3
from ..utils import popcount

import logging
logger = logging.getLogger(__name__)

class SimProcessor(ProcessorImpl):

    def __init__(self, mapping: Mapping):
        self.arch = mapping.arch
        self.mapping = mapping

        num_ports = mapping.get_num_ports()
        all_ports = list(range(num_ports))

        self.port2idx = dict()
        for x, p in enumerate(all_ports):
            self.port2idx[p] = x

        self.max_uop = self.uop2bv(all_ports)

    def uop2bv(self, u):
        """ Compute a bitvector representing the list p of ports.
        """
        res = 0
        for p in u:
            res += (1 << self.port2idx[p])
        return res

    def get_arch(self):
        return self.arch

    def execute(self, iseq: List[str], *args, **kwargs) -> Dict[str, Union[float, str]]:
        weights = defaultdict(lambda : 0)
        num_uops = 0
        res = None
        if isinstance(self.mapping, Mapping3):
            for i in iseq:
                for u in self.mapping.assignment[i]:
                    weights[self.uop2bv(u)] += 1
                    num_uops += 1
        elif isinstance(self.mapping, Mapping2):
            for i in iseq:
                weights[self.uop2bv(self.mapping.assignment[i])] += 1
                num_uops += 1
        else:
            raise NotImplementedError("get_cycles")
        res = self.cycles_for_weights(weights)
        if isinstance(res, tuple) and len(res) > 1:
            cycles, additional = res
            res = {'cycles': cycles, 'num_uops': num_uops, **additional}
        else:
            cycles = res
            res = {'cycles': cycles, 'num_uops': num_uops}

        return res

    @abstractmethod
    def cycles_for_weights(self, weights):
        """ Compute the number of cycles required to execute the experiment
            represented by the dictionary weights that maps operations that can
            be executed on ports to the number how often they occur in the
            experiment.
        """
        pass

def pure_core_algorithm(max_set, weights):
    max_val = Fraction(0)
    for q in range(1, max_set + 1):
        val = 0
        for u, w in weights.items():
            if (~q & u) == 0: # all ports of u are contained in q
                val += w
        val = Fraction(val) / popcount(q)
        max_val = max(max_val, val)
    return float(max_val)

class PurePortMappingProcessor(SimProcessor):
    """ Slow, but most portable simulation processor implementation, fully
        self-contained python.
    """
    def __init__(self, mapping: Mapping):
        super().__init__(mapping)

    def cycles_for_weights(self, weights):
        return pure_core_algorithm(self.max_uop, weights)


cpp_proc_path = os.path.join(os.path.dirname(__file__), *("../../../../lib/cppfastproc/build".split('/')))
sys.path.append(cpp_proc_path)

has_cppfastproc = False
try:
    from cppfastproc import FP
    has_cppfastproc = True
except:
    logger.warning("No cppfastproc module found, fast port mapping processor cannot be used.\nRun make in lib/cppfastproc to create the module")

class NativePortMappingProcessor(SimProcessor):
    """ Fast, but not so portable simulation processor implementation, uses
        external C++ code.
    """
    def __init__(self, mapping: Mapping):
        if not has_cppfastproc:
            raise RuntimeError("External C++ module for NativePortMappingProcessor not available!")
        super().__init__(mapping)
        num_ports = mapping.get_num_ports()
        self.fp = FP(num_ports)

    def cycles_for_weights(self, weights):
        self.fp.clear()

        for u, v in weights.items():
            self.fp.add(u, v)

        return self.fp.compute()

