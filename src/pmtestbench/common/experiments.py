""" Datastructures for representing experiments and collections of experiments.

An experiment is a multiset of instructions (as specified in an Architecture)
that can be annotated with a measured throughput and multiple other results,
e.g., from throughput predictors under evaluation.
"""

from typing import *
from functools import cached_property
import copy
import decimal

from collections import Counter

from .architecture import Architecture
from .jsonable import JSONable


class Experiment(JSONable):
    def __init__(self, iseq=None, result=None):
        super().__init__()
        self._iseq = iseq
        self.result = result
        self.other_results = []
        self.rid = None

    @property
    def iseq(self):
        return self._iseq

    def add_other_result(self, result_id, result):
        result['result_id'] = result_id
        self.other_results.append(result)

    def get_other_results(self, result_id):
        return [ r for r in self.other_results if r['result_id'] == result_id]

    def __iter__(self):
        return iter(self.iseq)

    @cached_property
    def counter(self):
        return Counter(self.iseq)

    def items(self):
        return self.counter.items()

    def get_distinct_insns(self):
        return list(self.counter.keys())

    def num_occurrences(self, insn):
        return self.counter[insn]

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return ", ".join(map(lambda x: str(x[0]) + ":" + str(x[1]), self.counter.items()))

    def get_name(self):
        return str(self.rid)

    def get_cycles(self):
        """ Returns the number of cycles of the experiment, rounded to three
        decimal digits.

        Rounding happens here in the getter because decimals are not
        JSON-encodable.
        """
        # properly round to 3 decimal digits to avoid large fractions in SMT
        # formulas using these values
        cycles = self.result["cycles"]
        cycles = decimal.Decimal(cycles).quantize(decimal.Decimal('0.001'), rounding=decimal.ROUND_HALF_UP)
        return cycles

    def __eq__(self, other):
        if type(other) is type(self):
            return self.counter == other.counter
        return False

    def from_json_dict(self, jsondict):
        kind = jsondict.get('kind', 'no')
        if kind != 'Experiment':
            raise RuntimeError(f'Trying to create an Experiment from {kind} json')

        self._iseq = jsondict["iseq"]

        self.result = jsondict["result"]
        if self.result is not None:
            self.result["cycles"] = float(self.result["cycles"])

        if "other_results" in jsondict:
            self.other_results = jsondict["other_results"]

    def to_json_dict(self):
        res = dict()
        res["kind"] = "Experiment"
        res["iseq"] = copy.deepcopy(self.iseq)
        res["result"] = copy.deepcopy(self.result)
        if self.other_results is not None and len(self.other_results) != 0:
            res["other_results"] = copy.deepcopy(self.other_results)
        return res

class ExperimentList(JSONable):
    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.exps = []
        self.experiment_id = 0

    def __iter__(self):
        return iter(self.exps)

    def __len__(self):
        return len(self.exps)


    def clear(self):
        self.exps.clear()

    def insert_exp(self, e):
        e.rid = self.experiment_id
        self.experiment_id += 1
        self.exps.append(e)

    def create_exp(self, ilist):
        new_exp = Experiment(ilist)
        self.insert_exp(new_exp)
        return new_exp

    def from_json_dict(self, jsondict):
        kind = jsondict.get('kind', 'no')
        if kind != 'ExperimentList':
            raise RuntimeError(f'Trying to create an ExperimentList from {kind} json')

        if self.arch is None:
            self.arch = Architecture()
            self.arch.from_json_dict(jsondict["arch"])
        else:
            self.arch.verify_json_dict(jsondict["arch"])

        for edict in jsondict["exps"]:
            e = Experiment()
            e.from_json_dict(edict)
            self.insert_exp(e)

    def to_json_dict(self):
        res = dict()
        res["kind"] = "ExperimentList"
        res["arch"] = self.arch.to_json_dict()
        res["exps"] = [ e.to_json_dict() for e in self.exps ]
        return res


