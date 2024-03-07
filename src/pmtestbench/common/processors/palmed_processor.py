""" An implementation of the conjunctive resource model of Palmed.
The implementation is ours, but the resource model was proposed by Derumigny et
al. in their article "PALMED: throughput characterization for superscalar architectures".
"""

from collections import defaultdict
import json
import re

import iwho
from iwho import x86
from iwho.configurable import load_json_config

from . import ProcessorImpl
from ..architecture import Architecture

vcmp_pat = re.compile("VCMP(.*)(PD|PS|SD|SS)")

def make_regexp(scheme):
    """ This was used in the process of creating the mapping from our
    insnschemes to theirs (with manual fixes).
    """
    mnemonic = x86.extract_mnemonic(scheme).upper()

    mat = vcmp_pat.match(mnemonic)
    add_imm = False
    if mat:
        mnemonic = f"VCMP{mat.group(2)}"
        add_imm = True

    pattern = f"{mnemonic}"

    for k, op_scheme in scheme.explicit_operands.items():
        if op_scheme.is_fixed():
            op = op_scheme.fixed_operand
            if isinstance(op, x86.RegisterOperand):
                pattern += f"_GPR{op.width}[^_]+"
            elif isinstance(op, x86.ImmediateOperand):
                pattern += f"_IMM[iu]{op.width}"
            else:
                pattern += f"_[^_]+"
        else:
            constraint = op_scheme.operand_constraint
            if isinstance(constraint, x86.RegisterConstraint):
                if constraint.width <= 64:
                    pattern += f"_GPR{constraint.width}[^_R]+"
                else:
                    pattern += f"_VR{constraint.width}[^_]+"
            elif isinstance(constraint, x86.MemConstraint):
                if constraint.width <= 64:
                    pattern += f"_MEM64[iu]{constraint.width}"
                else:
                    pattern += f"_MEM64[^_]+"
            elif isinstance(constraint, x86.ImmConstraint):
                pattern += f"_IMM[iu]{constraint.width}"
            else:
                pattern += f"_[^_]+"

    if add_imm:
        pattern += "_IMM[iu]8"

    print(pattern)

    return re.compile(pattern)



class PalmedProcessor(ProcessorImpl):
    def __init__(self, config):
        mapping_path = config['spm_path']
        with open(mapping_path, 'r') as f:
            self.mapping = json.load(f)

        insnmap_path = config['palmed_insnmap_path']
        with open(insnmap_path, 'r') as f:
            self.insnmap = json.load(f)

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


    def get_resources(self, insn):
        palmed_schemes = self.insnmap.get(insn, None)
        if palmed_schemes is None:
            return []

        assert len(palmed_schemes) == 1

        palmed_class = self.mapping['instr_to_class'].get(palmed_schemes[0], None)

        assert palmed_class is not None

        return self.mapping['class_to_resources'][palmed_class]['resource_use']


    def get_arch(self) -> Architecture:
        return self.arch

    def execute(self, iseq, *args, **kwargs):
        pressure = defaultdict(int)

        for i in iseq:
            resources = self.get_resources(i)
            for utilization, resource in resources:
                pressure[resource] += utilization

        if len(pressure) == 0:
            cycles = 0
        else:
            cycles = max(pressure.values())

        return {'cycles': cycles}
