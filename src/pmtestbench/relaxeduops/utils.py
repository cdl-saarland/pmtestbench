""" Utilities for relaxeduops algorithm.
"""

from dataclasses import dataclass
from collections import defaultdict, namedtuple

from ..common.portmapping import Mapping2, Mapping3

@dataclass
class AlgoParameters:
    """ Parameters for the relaxeduops algorithm.
    """
    use_cpi: bool = False
    num_uops_epsilon: float = 0.1
    num_ports_epsilon: float = 0.1
    tp_epsilon: float = 0.2
    true_tp_epsilon: float = 0.1
    surplus_uops_epsilon: float = 0.1
    insn_skip_cycles: float = 10.0
    num_ports: int = 8
    smt_slack_val: float = 0.1
    smt_slack_kind: str = 'absolute'
    smt_insn_bound: int = 0
    use_bottleneck_ipc: bool = True

    @classmethod
    def from_json_file(cls, f):
        import json
        d = json.load(f)
        return cls(**d)


InsnWithMeasurement = namedtuple('InsnWithMeasurement', ['insn', 'cycles', 'num_uops'])

RejectedInsn = namedtuple('RejectedInsn', ['insn', 'reason'])
RejectedExp = namedtuple('RejectedExp', ['exp', 'reason'])

def translate_port(port_before, translate_map, force_int=True):
    # For internal representation in port mappings, ports need to be integers.
    # Only for pretty printing, other representations can be used.
    if translate_map is None:
        res = port_before
    else:
        res = translate_map.get(str(port_before), port_before)

    if force_int:
        return int(res)
    else:
        return res

def translate_portmapping(port_mapping, translate_map):
    if isinstance(port_mapping, Mapping2):
        new_mapping = Mapping2(port_mapping.arch)
        for insn, port_usage in port_mapping.assignment.items():
            new_mapping.assignment[insn] = [translate_port(port, translate_map, force_int=True) for port in port_usage]
    elif isinstance(port_mapping, Mapping3):
        new_mapping = Mapping3(port_mapping.arch)
        for insn, port_usage in port_mapping.assignment.items():
            new_mapping.assignment[insn] = [
                    [translate_port(port, translate_map, force_int=True) for port in uop] for uop in port_usage
                ]
    else:
        raise ValueError("Unknown port mapping type")
    return new_mapping

