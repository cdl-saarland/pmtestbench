""" An implementation of the linear-programming-based port mapping throughput
model using Gurobi. It requires a licensed installation of Gurobi to work.
"""

from .portmapping_processor import SimProcessor
from ..portmapping import Mapping

import logging
logger = logging.getLogger(__name__)

try:
    from gurobipy import *
    has_gurobi = True
except ImportError as e:
    has_gurobi = False


class GurobiProcessor(SimProcessor):
    """ Simulation processor using an LP solved by Gurobi for computing cycle
        numbers.
    """
    def __init__(self, mapping: Mapping):
        if not has_gurobi:
            raise RuntimeError("No Gurobi version available for solving LPs!")
        super().__init__(mapping)

    def cycles_for_weights(self, weights):
        num_ports = self.mapping.get_num_ports()
        P = list(range(num_ports))

        m = Model("schedule")
        m.setParam('OutputFlag', 0)

        x_vars = m.addVars([(i, k)
            for i in weights.keys() for k in P], name="x")

        lat = m.addVar(name="latency", obj=1.0)

        m.addConstrs(( quicksum([ x_vars[(i, k)] for k in P ]) == n
            for i, n in weights.items()))

        m.addConstrs(( quicksum([ x_vars[(i, k)] for i in weights.keys() ]) <= lat
            for k in P))

        m.addConstrs((x_vars[(i, k)] == 0
            for i in weights.keys() for k in P if (self.uop2bv((k,)) & i) == 0 ))

        m.optimize()

        assert m.status == GRB.Status.OPTIMAL

        per_port = dict()
        for k in P:
            mass_parts = []
            for i in weights.keys():
                v = x_vars[(i, k)].x
                if v > 0.0:
                    mass_parts.append((i, v))
            per_port[k] = mass_parts

        return m.objVal, {'per_port': per_port}


