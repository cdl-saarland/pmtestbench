""" Datastructures for different forms of instruction-to-port mappings.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
import copy
import itertools
import json

from .jsonable import JSONable
from .architecture import Architecture
from .experiments import Experiment

class Mapping(JSONable):
    """ Abstract base class for port mappings.
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def read_from_json_dict(jsondict, arch: Architecture = None):
        kind = jsondict.get('kind', 'no')
        if kind not in ['Mapping2', 'Mapping3']:
            raise RuntimeError(f'Trying to create a Mapping from {kind} json')

        if arch is None:
            arch = Architecture()
            arch.from_json_dict(jsondict["arch"])
        else:
            arch.verify_json_dict(jsondict["arch"])

        if jsondict["kind"] == "Mapping3":
            res = Mapping3(arch)
            res.from_json_dict(jsondict)
            return res

        if jsondict["kind"] == "Mapping2":
            res = Mapping2(arch)
            res.from_json_dict(jsondict)
            return res

        raise NotImplementedError("read_from_json")

    @abstractmethod
    def get_num_ports(self):
        pass

    @staticmethod
    def read_from_json(infile, arch: Architecture = None):
        jsondict = json.load(infile)
        return Mapping.read_from_json_dict(jsondict, arch)

    @staticmethod
    def read_from_json_str(instr, arch: Architecture = None):
        jsondict = json.loads(instr)
        return Mapping.read_from_json_dict(jsondict, arch)

class Mapping2(Mapping):
    """ Class representing port mappings where instructions are directly
        executed on ports.
    """
    def __init__(self, arch: Architecture):
        super().__init__()
        self.arch = arch

        # an assignment from instructions to lists of ports
        self.assignment = dict()

    def get_num_ports(self):
        return max( (max(ps) for i, ps in self.assignment.items()) ) + 1

    def __getitem__(self, key):
        assert key in self.assignment
        return self.assignment[key]

    def __repr__(self):
        res = "Mapping2(arch={}, assignment={})".format(repr(self.arch), repr(self.assignment))
        return res

    def to_json_dict(self):
        res = dict()
        res["kind"] = "Mapping2"
        res["arch"] = self.arch.to_json_dict()
        res["assignment"] = copy.deepcopy(self.assignment)
        return res

    def from_json_dict(self, jsondict):
        kind = jsondict.get('kind', 'no')
        if kind != 'Mapping2':
            raise RuntimeError(f'Trying to create a Mapping2 from {kind} json')

        self.assignment = jsondict["assignment"]

    @classmethod
    def from_model(cls, arch: Architecture, num_ports, model):
        """ Create a Mapping2 from a model, i.e. a dictionary i2p.

        i2p maps pairs of instructions i from arch and ports to a True value
        iff i can be executed on p according to the mapping.
        """
        i2p = model
        I = arch.insn_list
        res = cls(arch)

        for i in I:
            res.assignment[i] = []

        for (i, p), v in i2p.items():
            if v:
                res.assignment[i].append(p)
        return res


class Mapping3(Mapping):
    """ Class representing port mappings where instructions are decomposed into
        uops that can be executed on ports.
    """

    def __init__(self, arch: Architecture):
        super().__init__()

        self.arch = arch

        # an assignment from instructions to lists of lists of ports
        self.assignment = { i: [] for i in  self.arch.insn_list }

    def get_num_ports(self):
        return max( (max((0, *(max((0, *ps)) for ps in us) )) for i, us in self.assignment.items()) ) + 1

    def __getitem__(self, key):
        assert key in self.assignment
        return self.assignment[key]

    def __repr__(self):
        res = "Mapping3(arch={}, assignment={})".format(repr(self.arch), repr(self.assignment))
        return res

    def to_json_dict(self):
        res = dict()
        res["kind"] = "Mapping3"
        res["arch"] = self.arch.to_json_dict()
        res["assignment"] = copy.deepcopy(self.assignment)
        return res

    def from_json_dict(self, jsondict):
        kind = jsondict.get('kind', 'no')
        if kind != 'Mapping3':
            raise RuntimeError(f'Trying to create a Mapping3 from {kind} json')

        arch = self.arch
        assignment_dict = jsondict["assignment"]
        for i, us in assignment_dict.items():
            curr_uops = []
            for ps in us:
                curr_uops.append(list(ps))
            self.assignment[i] = curr_uops

    @classmethod
    def from_model(cls, arch: Architecture, num_ports, model):
        """ Create a Mapping3 from a model, i.e. a tuple (i2u, u2p) of
            dictionaries.
            i2u maps pairs of instructions i from arch and some objects u
            representing uops to a True value iff i should be decomposed into u
            according to the mapping.
            u2p does the same for tuples of uop representations u and ports p to
            indicate that u can be executed on p.
        """
        (i2u, u2p) = model
        P = list(range(num_ports))
        res = cls(arch)
        for (i, u), v in i2u.items():
            uop = []
            for p in P:
                if u2p.get((u, p), False):
                    uop.append(p)
            if len(uop) > 0:
                res.assignment[i].append(uop)
        return res


name_to_class = {
        'Mapping2': Mapping2,
        'Mapping3': Mapping3,
    }

