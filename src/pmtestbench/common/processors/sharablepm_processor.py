""" Wrapper that uses a port mapping in the sharable format exported via the scripts/relaxed_uops_export_portmapping.py script for throughput predictions.
"""

from collections import defaultdict
from functools import reduce
import json
import os
import sys
from typing import Dict, List, Union

from . import ProcessorImpl
from ..architecture import Architecture

import logging
logger = logging.getLogger(__name__)


def portset2uop(port_index, portset):
    res = 0
    for x, port in enumerate(port_index):
        if port in portset:
            res |= 1 << x
    return res

def popcount(n):
    """ Return the number of 1s in the binary representation of the number n.
    """
    return bin(n).count("1")

def predict_itp(portmapping, insnschemes):
    weights = defaultdict(int)
    port_index = portmapping['ports']

    peak_itp = len(insnschemes) / portmapping['peakipc']

    num_uops = 0

    for insn in insnschemes:
        pm_entry = portmapping['data'].get(insn)
        if pm_entry is None:
            logger.warning("Instruction {} not found in port mapping, skipping".format(insn))
            continue
        for uop, num in pm_entry['portusage']:
            weights[portset2uop(port_index, uop)] += num
            num_uops += num

    max_set = reduce(lambda x, y: x | y, weights.keys())

    max_val = 0.0
    for q in range(1, max_set + 1):
        val = 0.0
        for u, w in weights.items():
            if (~q & u) == 0: # all ports of u are contained in q
                val += w
        val = val / popcount(q)
        max_val = max(max_val, val)

    sim_result = max_val


    return max(sim_result, peak_itp), num_uops


class SharablePMProcessor(ProcessorImpl):

    def __init__(self, sharable_mapping, restrict):
        if not isinstance(sharable_mapping, dict):
            with open(sharable_mapping, 'r') as f:
                sharable_mapping = json.load(f)

        new_arch = Architecture()
        for i in sharable_mapping['data'].keys():
            if restrict and len(sharable_mapping['data'][i]['portusage']) == 0:
                continue
            new_arch.add_insn(i)
        self.arch = new_arch
        self.sharable_mapping = sharable_mapping

    def get_arch(self):
        return self.arch

    def execute(self, iseq: List[str], *args, **kwargs) -> Dict[str, Union[float, str]]:
        cycles, num_uops = predict_itp(self.sharable_mapping, iseq)
        return {'cycles': cycles, 'num_uops': num_uops}

