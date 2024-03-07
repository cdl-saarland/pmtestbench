""" A data structure to manage a collection of instructions supported by a processor.
"""

from typing import *

from .jsonable import JSONable

import logging
logger = logging.getLogger(__name__)

class Architecture(JSONable):
    def __init__(self):
        super().__init__()
        self._insns = set()

    @property
    def insn_list(self):
        # TODO this could be wasteful
        return sorted(self._insns)

    def add_insn(self, name: str):
        if name in self._insns:
            raise RuntimeError(f'Trying to insert a duplicate instruction: {name}')
        self._insns.add(name)

    def add_insns(self, names: List[str]):
        for n in names:
            self.add_insn(n)

    def __repr__(self):
        return self.to_json_str()

    def __str__(self):
        return self.to_json_str()

    def verify_json_dict(self, jsondict):
        """ Check whether new architecture is identical to the current one
        """
        curr_insns = self._insns
        new_insns = set(jsondict["insns"])
        if curr_insns != new_insns:
            in_new = new_insns - curr_insns
            in_old = curr_insns - new_insns
            msg = ""
            if len(in_new) > 0:
                msg += "\ninstructions in new architecture that were not in the old one:\n" + "\n".join(map(str, in_new))
            if len(in_old) > 0:
                msg += "\ninstructions in old architecture that are not in the new one:\n" + "\n".join(map(str, in_old))
            logger.info("Incompatible Arichtectures!" + msg)
            raise RuntimeError(f'Incompatible Architectures (insns)')

    def from_json_dict(self, jsondict):
        if jsondict['kind'] != 'Architecture' and 'arch' in jsondict:
            # enable reading architectures from mapping jsons, etc.
            jsondict = jsondict['arch']

        kind = jsondict.get('kind', 'no')
        if kind != 'Architecture':
            raise RuntimeError(f'Trying to create an Architecture from {kind} json')

        input_insn_list = jsondict["insns"]
        self.add_insns(input_insn_list)


    def to_json_dict(self):
        res = dict()
        res["kind"] = "Architecture"
        res["insns"] = list(self.insn_list)
        return res


