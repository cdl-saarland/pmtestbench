""" A port mapping synthesis algorithm using the z3 SMT solver.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
import itertools
from typing import *
import copy
import time
import json
from datetime import datetime

from z3 import *

from ..common.synthesizers import SynthesizerImpl

from ..common.experiments import Experiment, ExperimentList
from ..common.portmapping import Mapping2, Mapping3
from ..common.utils import popcount

from iwho.utils import timestamped_filename

import logging
logger = logging.getLogger(__name__)


# enable parallel z3
set_param("parallel.enable", True)


def SumIfNecessary(items):
    # z3 is fine with sums with fewer than 2 operands. cvc5 isn't.
    if len(items) == 0:
        return 0
    if len(items) == 1:
        return items[0]
    return Sum(items)

class SMTSynthesizer(SynthesizerImpl):

    def __init__(self, config):
        self.config = config

        self.mapping_cls = {
                "Mapping2": Mapping2,
                "Mapping3": Mapping3,
            }.get(config['mapping_class'], None)

        self.num_ports = config['num_ports']

        self.num_uops = config.get("num_uops", None)
        assert self.num_uops is not None or self.mapping_cls == Mapping2

        self.slack_val = config["smt_slack_val"]
        self.slack_kind = config["smt_slack_kind"]
        self.insn_bound = config["smt_insn_bound"]
        self.exp_limit_strategy = config["smt_exp_limit_strategy"]

        self.full_mul_uopsize_limit = config['smt_full_mul_uopsize_limit']

        self.num_uops_per_insn = config["num_uops_per_insn"]
        self.constrain_improper_uops = config["smt_constrain_improper_uops"]

        self.dump_constraints = config["smt_dump_constraints"]

        # set this to True to dump an unsat core when find_mapping fails
        self.dump_unsat_core = False

        self.reset_fields()


    def reset_fields(self):
        self.arch = None
        self.solver = None
        self.num_exp_insns = 1
        self.found_with_curr_num_exp_insns = 0
        self.mapping_handler = None
        self.m1_encoding = None
        self.m2_encoding = None
        self.new_exp_vars = None
        self.assumption_var = None
        self.exp1_encoding = None
        self.exp2_encoding = None
        self.exp_ids = dict()
        self.next_id = 0
        self.all_experiments = []

    def synthesize(self, proc, exps, solver=None, stats=None, known_portset_sizes=None, bottleneck_ipc=None):
        # stats may be a list where additional statistics are added
        self.reset_fields()

        self.arch = proc.get_arch()
        if solver is None:
            self.solver = Solver()
        else:
            self.solver = solver
        self.mapping_handler = MappingHandler.for_class(self.mapping_cls, self.config, synth=self)
        self.mapping_handler.bottleneck_ipc = bottleneck_ipc
        self.add_initial_constraints()

        self.add_experiments(exps.exps)

        if known_portset_sizes is not None:
            self.mapping_handler.encode_mapping_portset_sizes(self.m1_encoding, known_portset_sizes)

        m1 = None

        iter_no = 0

        logger.info('start synthesizing a mapping')
        while True:
            iter_no += 1
            logger.info(f'synthesis iteration: {iter_no}')

            logger.info('  trying to find mapping')
            start = time.perf_counter()
            m1 = self.find_mapping()
            end = time.perf_counter()
            fm_time = end - start
            if m1 is None:
                logger.info('  none found -> synthesis failed')
                if stats is not None:
                    stats.append({'result': 'synthesis failed', 'find_mapping_time': fm_time})
                return None

            logger.info('  found one')

            logger.info('  trying to find other mapping')
            start = time.perf_counter()
            res = self.find_other_mapping(m1)
            end = time.perf_counter()
            fom_time = end - start

            if res is None:
                logger.info('  none found -> synthesis successful')
                if stats is not None:
                    if self.num_exp_insns > self.insn_bound:
                        stats.append({
                                'result': 'synthesis terminated at insn bound',
                                'find_mapping_time': fm_time,
                                'find_other_mapping_time': fom_time,
                            })
                    else:
                        stats.append({
                                'result': 'synthesis successful',
                                'find_mapping_time': fm_time,
                                'find_other_mapping_time': fom_time,
                            })
                return m1

            logger.info('  found one')

            m2, es = res

            new_exp = Experiment(es)
            proc.eval(new_exp)
            exps.insert_exp(new_exp)

            if stats is not None:
                stats.append({
                        'result': 'synthesis continues',
                        'find_mapping_time': fm_time,
                        'find_other_mapping_time': fom_time,
                        'exp_len': len(es),
                    })

            logger.info(f'  adding new experiment {new_exp} with {new_exp.get_cycles()} cycle(s)')
            self.add_experiments([new_exp])


    def why_not(self, mapping, elist, known_portset_sizes=None, bottleneck_ipc=None):
        """ Check if (and why) a mapping is not valid for a set of experiments.
        Returns true if the mapping is valid, false otherwise.
        """

        self.reset_fields()
        self.arch = elist.arch
        self.solver = Solver()
        self.mapping_handler = MappingHandler.for_class(self.mapping_cls, self.config, synth=self)
        self.mapping_handler.bottleneck_ipc = bottleneck_ipc

        self.add_initial_constraints()

        # m1 should satisfy the experiments
        self.add_experiments(elist)

        # m1 should be equal to the given mapping
        self.mapping_handler.encode_mapping(self.m1_encoding, mapping)

        return self.find_mapping() is not None

    def get_experimentlist(self):
        elist = ExperimentList(self.arch)
        for e in self.all_experiments:
            elist.insert_exp(copy.deepcopy(e))
        return elist

    def infer(self, exps, solver=None):
        self.reset_fields()

        self.arch = exps.arch
        if solver is None:
            self.solver = Solver()
        else:
            self.solver = solver
        self.mapping_handler = MappingHandler.for_class(self.mapping_cls, self.config, synth=self)

        self.add_initial_constraints()
        self.add_experiments(exps)
        return self.find_mapping()

    def find_mapping(self):
        s = self.solver
        res = s.check()

        if res != sat:
            logger.info("    Non-sat z3 result: " + str(res))
            if res == unsat and self.dump_unsat_core:
                ucore = s.unsat_core()
                core_experiments = []
                for ev in ucore:
                    core_experiments.append(self.exp_ids[str(ev)])
                unsat_core_data = {
                        'architecture': self.arch.to_json_dict(),
                        'config': self.config,
                        'core_experiments': [ e.to_json_dict() for e in core_experiments],
                    }
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                fn = f"./unsat_core_{timestamp}.json"
                with open(fn, "w") as f:
                    json.dump(unsat_core_data, f, indent=4)
                logger.info(f"    unsat core data written to {fn}")
                # print(s.to_smt2())
            return None

        model = s.model()
        return self.mapping_handler.decode_mapping(self.m1_encoding, model)

    def find_other_mapping(self, m1):
        solver = self.solver
        solver.push()

        # bind mapping to m2 variables
        self.mapping_handler.encode_mapping(self.m2_encoding, m1)

        success = False

        if self.exp_limit_strategy == 'unbounded':
            # do not limit the length of the experiment
            # that's usually a bad idea, since the queried experiments tend to
            # be HUGE
            res = solver.check(self.assumption_var)
            if res == sat:
                success = True
                model = solver.model()
            else:
                assert res == unsat
                solver.pop()
                return None
        elif self.exp_limit_strategy == 'incremental_bounded':
            # Increment the limit whenever it is not enough, up to the
            # insn_bound. This does not produce a statement about unlimited
            # experiments.
            while True:
                solver.push()
                self.mapping_handler.encode_free_experiment_bound(self.new_exp_vars, self.num_exp_insns)

                res = solver.check(self.assumption_var)

                if res == sat:
                    success = True
                    model = solver.model()
                    solver.pop()
                    break
                else:
                    assert res == unsat
                    if self.dump_constraints:
                        solver.push()
                        solver.add(self.assumption_var)
                        constraints = solver.to_smt2()
                        with open(timestamped_filename('constraint', 'smt', modifier='unsat'), 'w') as f:
                            f.write(constraints)
                        solver.pop()
                    solver.pop()
                    self.num_exp_insns += 1
                    if self.num_exp_insns > self.insn_bound:
                        solver.pop()
                        return None
                    logger.info("    increased experiment length to {}".format(self.num_exp_insns))
        elif self.exp_limit_strategy == 'incremental_optimistic':
            # Start by incrementing the limit when it is not enough to find a
            # new experiment. Once a limit increase does not find a new
            # experiment, try without a bound next.
            while True:
                solver.push()
                self.mapping_handler.encode_free_experiment_bound(self.new_exp_vars, self.num_exp_insns)

                res = solver.check(self.assumption_var)

                if res == sat:
                    self.found_with_curr_num_exp_insns += 1
                    success = True
                    model = solver.model()
                    solver.pop()
                    break
                else:
                    assert res == unsat
                    logger.info("    nothing found")
                    if self.dump_constraints:
                        solver.push()
                        solver.add(self.assumption_var)
                        constraints = solver.to_smt2()
                        with open(timestamped_filename('constraint', 'smt', modifier='unsat'), 'w') as f:
                            f.write(constraints)
                        solver.pop()
                    solver.pop()

                    if self.found_with_curr_num_exp_insns == 0:
                        logger.info("    trying to find an experiment of unlimited size")
                        # try without limit
                        res = solver.check(self.assumption_var)
                        if res == unsat:
                            logger.info("    found none!")
                            # we have shown that there is no counterexample
                            solver.pop()
                            return None
                        else:
                            assert res == sat
                            logger.info("    found one, continuing")
                            # we found a counterexample, but it might be very
                            # large, so proceed by incrementing num_exp_insns

                    self.num_exp_insns += 1
                    self.found_with_curr_num_exp_insns = 0
                    if self.num_exp_insns > self.insn_bound:
                        solver.pop()
                        logger.info("    instruction bound hit!")
                        return None
                    logger.info("    increased experiment length to {}".format(self.num_exp_insns))

        else:
            raise RuntimeError(f"Unknown experiment limit strategy: {self.exp_limit_strategy}")

        result_map = self.mapping_handler.decode_mapping(self.m1_encoding, model)

        # extract experiment
        new_exp = self.mapping_handler.extract_free_experiment(self.new_exp_vars, model)

        solver.pop()
        return result_map, new_exp

    def add_experiments(self, new_exps):
        mh = self.mapping_handler

        for e in new_exps:
            self.all_experiments.append(e)

            exp_map = { i: n for i, n in e.items() }
            exp_encoding = mh.add_experiment_vars(exp_map.keys(), e.get_name())

            id_str = "exp_{}".format(self.next_id)
            self.exp_ids[id_str] = e
            self.next_id += 1

            mh.encode_experiment(self.m1_encoding, exp_encoding, exp_map, track_as=id_str)

            tv = exp_encoding['t_var']
            measured_t = e.get_cycles()

            if (self.slack_val == 0.0):
                self.solver.add(tv == measured_t)
            else:
                if self.slack_kind == "absolute":
                    self.solver.add(And(tv - self.slack_val < measured_t, measured_t < tv + self.slack_val))
                elif self.slack_kind == "cycle-relative":
                    self.solver.add(And(tv * (1.0 - self.slack_val) < measured_t, measured_t < tv * (1.0 + self.slack_val)))
                elif self.slack_kind == 'cpi':
                    # this basically means that we need to multiply the slack/epsilon by the number of instructions in the experiment
                    exp_len = sum((n for i, n in exp_map.items()))
                    self.solver.add(And(tv - (self.slack_val * exp_len) < measured_t, measured_t < tv + (self.slack_val * exp_len)))
                else:
                    assert False, f"unsupported slack kind: {self.slack_kind}"

    def add_initial_constraints(self):
        mh = self.mapping_handler
        # initial constraints
        self.m1_encoding = mh.add_mapping_vars("m1")
        mh.encode_valid_mapping_constraints(self.m1_encoding)
        self.m2_encoding = mh.add_mapping_vars("m2")

        # create free exp vars
        I = self.arch.insn_list
        self.new_exp_vars = mh.add_free_experiment_vars(I)

        # add encodings of distinguishing experiment for m1 and m2 under assumption variable
        self.assumption_var = Bool("assumption_var")

        def addAssumed(f):
            self.solver.add(Implies(self.assumption_var, f))

        # valid free exp vars
        mh.encode_valid_free_experiment_constraints(self.new_exp_vars, self.assumption_var)

        self.exp1_encoding = mh.add_experiment_vars(self.new_exp_vars, "new_exp1")
        mh.encode_experiment(self.m1_encoding, self.exp1_encoding, self.new_exp_vars, assumption_var=self.assumption_var)

        self.exp2_encoding = mh.add_experiment_vars(self.new_exp_vars, "new_exp2")
        mh.encode_experiment(self.m2_encoding, self.exp2_encoding, self.new_exp_vars, assumption_var=self.assumption_var)

        # add constraint for t1 != t2 (with or without slack) under assumption
        # variable
        # On slack:
        # When we talk about cycle timings, there are two categories:
        #   - ideal timings, which we would observe if every measurement was
        #     precise and scheduling would happen precisely according to our
        #     model and
        #   - measured timings, which we actually do observe.
        # The core logics of the formulas requires the ideal timings, which is
        # why we have to couple the measured timings with some ideal timings.
        #
        # When we are considering absolute slack values, we interpret a
        # measurement (e, tm) as follows:
        #   The difference between ideal timing ti and measured timing tm to
        #   execute experiment e is less than or equal to the specified slack
        #   value.
        #
        # For relative slack, things are slightly more complicated since we
        # need to decide to which value the slack should be relative.
        # We interpret a measurement (e, tm) here as follows:
        #   The measured timing tm lies in an interval of size 2*slack_val*ti
        #   around the ideal timing ti for e.
        t1 = self.exp1_encoding["t_var"]
        t2 = self.exp2_encoding["t_var"]

        if (self.slack_val == 0.0):
            addAssumed(t1 != t2)
        else:
            if self.slack_kind == 'absolute':
                # we need to ensure that the difference between the times is
                # larger than twice the slack_val, otherwise there could be an
                # actual timing between them that is in the allowed slack range
                # for both
                addAssumed(Or(t1 - t2 > 2 * self.slack_val, t2 - t1 > 2 * self.slack_val))
            elif self.slack_kind == 'cycle-relative':
                # The t variables describe the ideal values that we would
                # like to observe, we have to make sure that there cannot be an
                # experiment that would be in both of their allowed intervals.
                # Therefore we just have to check that the allowed intervals do
                # not intersect.
                addAssumed(If(t1 < t2,
                    t1 * (1.0 + self.slack_val) < t2 * (1.0 - self.slack_val),
                    t2 * (1.0 + self.slack_val) < t1 * (1.0 - self.slack_val)
                    ))
            elif self.slack_kind == 'cpi':
                # this basically means that we need to multiply the slack/epsilon by the number of instructions in the experiment
                exp_len = mh.free_experiment_length(self.new_exp_vars)
                addAssumed(Or(t1 - t2 > (2 * self.slack_val * exp_len), t2 - t1 > (2 * self.slack_val * exp_len)))
            else:
                assert False, f"unsupported slack kind: {self.slack_kind}"



def add_to_solver(solver, constraint, assumption_var=None):
    """ Add the given constraint to the solver.
    If assumption_var is not None, a constraint of the form
    (assumption_var => constraint) is added instead.
    """
    if assumption_var is not None:
        solver.add(Implies(assumption_var, constraint))
    else:
        solver.add(constraint)


class MappingHandler(ABC):
    """ An abstract base class that declares the methods needed to implement
    the synthesis algorithm in the SMTSynthesizer.
    Different instantiations are possible, for different port mapping kinds or
    using different encoding strategies.
    """
    @staticmethod
    def for_class(cls, config, **kwargs):
        if cls == Mapping3:
            if config.get('smt_use_constrained_mapping3', False):
                return ConstrainedMapping3Handler(**kwargs)
            if config.get('smt_use_full_mul_handler', False):
                return FullMultiplyMapping3Handler(**kwargs)
            if config.get('smt_use_bn_handler', False):
                if config.get('smt_bn_handler_bits', -1) > 0:
                    return BNMapping3HandlerBitBlasted(**kwargs)
                else:
                    return BNMapping3Handler(**kwargs)
            return Mapping3Handler(**kwargs)
        elif cls == Mapping2:
            return Mapping2Handler(**kwargs)
        raise NotImplementedError("MappingHandler.for_class")

    @abstractmethod
    def add_mapping_vars(self, identifier):
        """ Add SMT variables for a new mapping, referred to by the given
        unique identifier.
        (The user has to make sure that it is indeed unique.)

        Returns a data structure containing the introduced SMT variables, i.e.
        the encoding, which can be used by other methods of this class.
        """
        pass

    @abstractmethod
    def encode_valid_mapping_constraints(self, encoding, assumption_var):
        """ Add SMT constraints to assert that the given encoding fulfills
        the basic requirements for the chosen port mapping model.
        """
        pass

    def encode_mapping_portset_sizes(self, encoding, known_portset_sizes, assumption_var):
        """ Encode the known sizes of port sets for individual instructions
        given in the (insn -> size) dict `known_portset_sizes`.
        Intentionally not abstract, since it is not required for correctness.
        Only the two-level handler currently implements this.
        """
        pass

    @abstractmethod
    def encode_mapping(self, encoding, mapping, assumption_var):
        """ Add SMT constraints to assert that the given encoding is hardwired
        to represent the given port mapping.
        """
        pass

    @abstractmethod
    def decode_mapping(self, encoding, model):
        """ Interpret the values of the variables in the given port mapping
        encoding in the given model as a port mapping.
        The extracted port mapping is returned.
        """
        pass

    @abstractmethod
    def add_experiment_vars(self, insns, identifier):
        """ Add SMT variables for a new experiment that contains only
        instructions from the given list, referred to by the given unique
        identifier.
        (The user has to make sure that it is indeed unique.)

        Returns a data structure containing the introduced SMT variables, i.e.
        the encoding, which can be used by other methods of this class.
        """
        pass

    @abstractmethod
    def encode_experiment(self, mapping_encoding, exp_encoding, exp_map, *, assumption_var, track_as):
        """ Add SMT constraints to assert that the experiment that is
        represented by the exp encoding and with instruction occurrence numbers
        specified in the exp map is executed with the port mapping implied by
        the given mapping encoding.

        At most one of assumption_var and track_as can be given. If track_as is
        provided (with a string), solver.assert_and_track() is used to register
        all constraints with this id for unsat core generation.
        Currently only supported for the two-level model (because it was only
        needed there, it should be simple to do in the others as well if
        necessary).
        """
        pass

    @abstractmethod
    def add_free_experiment_vars(self, insns):
        """ Add the SMT variables to represent a free experiment that can use
        any of the given instructions.
        The encoding is returned.
        """
        pass

    @abstractmethod
    def free_experiment_length(self, free_exp_vars):
        """ Return an SMT formula that is equal to the total number of
        instructions (equal or distinct) in the free experiment.
        """
        pass

    @abstractmethod
    def encode_valid_free_experiment_constraints(self, exp_vars, assumption_var):
        """ Add constraints to assert that the given free experiment satisfies
        basic properties.
        """
        pass

    @abstractmethod
    def encode_free_experiment_bound(self, free_exp_vars, num_exp_insns, assumption_var):
        """ Add constraints to assert that the given free experiment satisfies
        uses no more than num_exp_insns instructions.
        """
        pass

    @abstractmethod
    def extract_free_experiment(self, free_exp_vars, model):
        """ Interpret the values of the variables in the given free experiment
        encoding in the given model as a new experiment.
        The extracted experiment is returned.
        """
        pass


class Mapping2Handler(MappingHandler):

    def __init__(self, synth):
        self.solver = synth.solver
        self.arch = synth.arch
        self.handled_class = synth.mapping_cls
        self.ports = list(range(synth.num_ports))
        self.bottleneck_ipc = None

    def add_free_experiment_vars(self, insns):
        return { i: Int("new_exp_num_i{}".format(i)) for i in insns }

    def free_experiment_length(self, free_exp_vars):
        return Sum([v for i, v in free_exp_vars.items()])

    def encode_valid_free_experiment_constraints(self, exp_vars, assumption_var=None):
        geq_zero = And([ v >= 0 for i, v in exp_vars.items() ])
        add_to_solver(self.solver, geq_zero, assumption_var)

    def encode_free_experiment_bound(self, free_exp_vars, num_exp_insns, assumption_var=None):
        bound_constr = self.free_experiment_length(free_exp_vars) == num_exp_insns
        add_to_solver(self.solver, bound_constr, assumption_var)

    def extract_free_experiment(self, free_exp_vars, model):
        new_exp = []

        for i, v in free_exp_vars.items():
            num_i = model[v].as_long()
            for x in range(num_i):
                new_exp.append(i)

        return new_exp

    def add_mapping_vars(self, identifier):
        I = self.arch.insn_list
        P = self.ports
        m_vars = {
                (i, p): Bool("mapping_{}_m_i{}_p{}".format(identifier, i, p))
                for i in I for p in P
            }
        encoding = { "m_vars": m_vars}
        return encoding

    def encode_valid_mapping_constraints(self, encoding, assumption_var=None):
        I = self.arch.insn_list
        P = self.ports
        m_vars = encoding["m_vars"]
        solver = self.solver
        # every instruction has at least one port to execute on
        add_to_solver(solver, And([Or([m_vars[(i, k)] for k in P]) for i in I]), assumption_var)

    def encode_mapping_portset_sizes(self, encoding, known_portset_sizes, assumption_var=None):
        solver = self.solver
        I = self.arch.insn_list
        P = self.ports
        m_vars = encoding["m_vars"]
        constraint = And([
                SumIfNecessary([If(m_vars[(i, p)], 1, 0) for p in P]) == known_portset_sizes[i] for i in I if i in known_portset_sizes ])
        add_to_solver(solver, constraint, assumption_var)


    def encode_mapping(self, encoding, mapping, assumption_var=None):
        m_vars = encoding["m_vars"]
        solver = self.solver
        for (i, k), mv in m_vars.items():
            ps = mapping.assignment[i]
            if k in ps:
                add_to_solver(self.solver, mv, assumption_var)
            else:
                add_to_solver(self.solver, Not(mv), assumption_var)

    def decode_mapping(self, encoding, model):
        i2p = { (i, k): model[v] for ((i, k), v) in encoding["m_vars"].items() }
        return self.handled_class.from_model(self.arch, len(self.ports), i2p)

    def add_experiment_vars(self, insns, identifier):
        P = self.ports
        q_vars = {
                k: Bool("exp_{}_q_p{}".format(identifier, k))
                for k in P
            }
        j_vars = {
                i: Bool("exp_{}_j_i{}".format(identifier, i))
                for i in insns
            }
        x_vars = {
                (i, k): Real("exp_{}_x_i{}_p{}".format(identifier, i, k))
                for i in insns for k in P
            }
        p_vars = {
                k: Real("exp_{}_p_p{}".format(identifier, k))
                for k in P
            }

        t_ideal_var = Real("exp_{}_t_ideal".format(identifier))
        t_var = Real("exp_{}_t".format(identifier))

        res = dict()
        res["q_vars"] = q_vars
        res["j_vars"] = j_vars
        res["x_vars"] = x_vars
        res["p_vars"] = p_vars
        res["t_ideal_var"] = t_ideal_var
        res["t_var"] = t_var
        return res

    def encode_experiment(self, mapping_encoding, exp_encoding, exp_map, *, assumption_var=None, track_as=None):
        bottleneck_ipc = self.bottleneck_ipc

        m_vars = mapping_encoding["m_vars"]

        q_vars = exp_encoding["q_vars"]
        j_vars = exp_encoding["j_vars"]
        x_vars = exp_encoding["x_vars"]
        p_vars = exp_encoding["p_vars"]
        t_ideal_var = exp_encoding["t_ideal_var"]
        t_var = exp_encoding["t_var"]

        P = self.ports
        IE = exp_map.keys()

        conjuncts = []

        # there is a non-empty set of bottleneck ports
        conjuncts.append(Or([q_vars[k] for k in P]))
        # there is a non-empty set of bottleneck insns
        conjuncts.append(Or([j_vars[i] for i in IE]))
        # ports have non-negative pressure
        conjuncts.append(And([p_vars[k] >= 0 for k in P]))
        # no port has more pressure than the measured time
        conjuncts.append(And([p_vars[k] <= t_ideal_var for k in P]))
        # a port is a bottleneck port iff it has maximal pressure
        conjuncts.append(And([q_vars[k] == (p_vars[k] == t_ideal_var) for k in P]))
        # the total pressure on bottleneck ports is equal to the number of bottleneck insns
        conjuncts.append(Sum([If(j_vars[i], exp_map[i], 0) for i in IE]) == Sum([If(q_vars[k], t_ideal_var, 0) for k in P]))
        # bottleneck insns can only be executed on bottleneck ports
        conjuncts.append(And([And([Implies(And(j_vars[i], Not(q_vars[k])), Not(m_vars[(i, k)])) for k in P]) for i in IE]))
        # x vars are non-negative
        conjuncts.append(And([x_vars[(i, k)] >= 0 for i in IE for k in P]))
        # there has to be a way to execute all the insns of the experiment
        conjuncts.append(And([Sum([x_vars[(i, k)] for k in P]) == exp_map[i] for i in IE]))
        # a way to execute should also explain the port pressures
        conjuncts.append(And([Sum([x_vars[(i, k)] for i in IE]) == p_vars[k] for k in P]))
        # insns can only be executed on ports that are allowed according to the mapping
        conjuncts.append(And([Implies(x_vars[(i, k)] > 0, m_vars[(i, k)]) for i in IE for k in P]))

        if bottleneck_ipc is not None:
            exp_size = SumIfNecessary([v for i, v in exp_map.items()])
            if isinstance(exp_size, int):
                exp_size = float(exp_size)
            else:
                exp_size = ToReal(exp_size)
            min_cycles = exp_size / bottleneck_ipc
            conjuncts.append(t_var == If(t_ideal_var < min_cycles, min_cycles, t_ideal_var))
        else:
            conjuncts.append(t_var == t_ideal_var)


        constraint = And(conjuncts)
        if track_as is not None:
            assert assumption_var is None, "Encoding experiment with assumption variable and tracking id at the same time!"
            self.solver.assert_and_track(constraint, track_as)
        else:
            add_to_solver(self.solver, constraint, assumption_var)

class ConstrainedMapping3Handler(MappingHandler):
    # For port mappings where most insns have only one uop and we know how many
    # uops the other ones have.
    # This duplicates code from the Mapping2 handler and the Mapping3 handler,
    # it would be nice to unify this at some point.
    def __init__(self, synth):
        self.num_uops_per_insn = synth.num_uops_per_insn
        self.constrain_improper_uops = synth.constrain_improper_uops
        self.solver = synth.solver
        self.arch = synth.arch
        self.handled_class = synth.mapping_cls
        self.ports = list(range(synth.num_ports))
        self.uops = {
                i: [(i, n) for n in range(self.num_uops_per_insn.get(i, 1))]
                for i in self.arch.insn_list
            }
        self.all_uops = list(itertools.chain.from_iterable(self.uops.values()))

    def add_free_experiment_vars(self, insns):
        return { i: Int("new_exp_num_i{}".format(i)) for i in insns }

    def free_experiment_length(self, free_exp_vars):
        return Sum([v for i, v in free_exp_vars.items()])

    def encode_valid_free_experiment_constraints(self, exp_vars, assumption_var=None):
        geq_zero = And([ v >= 0 for i, v in exp_vars.items() ])
        add_to_solver(self.solver, geq_zero, assumption_var)

    def encode_free_experiment_bound(self, free_exp_vars, num_exp_insns, assumption_var=None):
        bound_constr = self.free_experiment_length(free_exp_vars) == num_exp_insns
        add_to_solver(self.solver, bound_constr, assumption_var)

    def extract_free_experiment(self, free_exp_vars, model):
        new_exp = []

        for i, v in free_exp_vars.items():
            num_i = model[v].as_long()
            for x in range(num_i):
                new_exp.append(i)

        return new_exp

    def add_mapping_vars(self, identifier):
        P = self.ports
        m_vars = {
                (u, p): Bool("mapping_{}_m_u{}_p{}".format(identifier, str(u), str(p)))
                for u in self.all_uops for p in P
            }
        encoding = { "m_vars": m_vars}
        return encoding

    def encode_valid_mapping_constraints(self, encoding, assumption_var=None):
        P = self.ports
        m_vars = encoding["m_vars"]
        solver = self.solver

        # every used uop has at least one port to execute on
        add_to_solver(solver, And([Or([m_vars[(u, k)] for k in P]) for u in self.all_uops]), assumption_var)

        if self.constrain_improper_uops:
            # Every secondary uop needs to be equal to some proper uop.
            # That is an aribtrary assumption that is close enough to reality for
            # Zen+. The algorithm also works without it, but it's slower.
            proper_uops = []
            improper_uops = []
            for (i, n) in self.all_uops:
                if n == 0 and len(self.uops[i]) == 1:
                    proper_uops.append((i, n))
                if n > 0:
                    improper_uops.append((i, n))

            for u_improper in improper_uops:
                add_to_solver(solver,
                    Or([And([m_vars[(u_improper, p)] == m_vars[(u_proper, p)] for p in P]) for u_proper in proper_uops]),
                    assumption_var)


    def encode_mapping_portset_sizes(self, encoding, known_portset_sizes, assumption_var=None):
        solver = self.solver
        I = self.arch.insn_list
        P = self.ports
        m_vars = encoding["m_vars"]
        # this does not make sense for instructions with > 1 uop
        for i, size in known_portset_sizes.items():
            assert len(self.uops[i]) == 1, "Portset size constraint does not make sense for instructions with > 1 uop"
        constraint = And([
                SumIfNecessary([If(m_vars[(self.uops[i][0], p)], 1, 0) for p in P]) == known_portset_sizes[i] for i in I if i in known_portset_sizes ])
        add_to_solver(solver, constraint, assumption_var)


    def encode_mapping(self, encoding, mapping, assumption_var=None):
        m_vars = encoding["m_vars"]
        solver = self.solver

        for (u, k), mv in m_vars.items():
            (i, x) = u
            us = mapping.assignment[i]

            assert len(us) == self.num_uops_per_insn.get(i, 1), "Mapping does not match constrained number of uops per insn"

            if k in us[x]:
                add_to_solver(self.solver, mv, assumption_var)
            else:
                add_to_solver(self.solver, Not(mv), assumption_var)

    def decode_mapping(self, encoding, model):
        i2u = { (i, u): True for i, us in self.uops.items() for u in us }
        u2p = { (u, k): model[v] for ((u, k), v) in encoding["m_vars"].items() }
        return self.handled_class.from_model(self.arch, len(self.ports), (i2u, u2p))

    def add_experiment_vars(self, insns, identifier):
        P = self.ports
        q_vars = {
                k: Bool("exp_{}_q_p{}".format(identifier, k))
                for k in P
            }
        j_vars = {
                u: Bool("exp_{}_j_i{}".format(identifier, str(u)))
                for i in insns for u in self.uops[i]
            }
        x_vars = {
                (u, k): Real("exp_{}_x_u{}_p{}".format(identifier, str(u), k))
                for i in insns for u in self.uops[i] for k in P
            }
        p_vars = {
                k: Real("exp_{}_p_p{}".format(identifier, k))
                for k in P
            }

        t_ideal_var = Real("exp_{}_t_ideal".format(identifier))
        t_var = Real("exp_{}_t".format(identifier))

        res = dict()
        res["q_vars"] = q_vars
        res["j_vars"] = j_vars
        res["x_vars"] = x_vars
        res["p_vars"] = p_vars
        res["t_ideal_var"] = t_ideal_var
        res["t_var"] = t_var
        return res

    def encode_experiment(self, mapping_encoding, exp_encoding, exp_map, *, assumption_var=None, track_as=None):
        bottleneck_ipc = self.bottleneck_ipc

        m_vars = mapping_encoding["m_vars"]

        q_vars = exp_encoding["q_vars"]
        j_vars = exp_encoding["j_vars"]
        x_vars = exp_encoding["x_vars"]
        p_vars = exp_encoding["p_vars"]
        t_ideal_var = exp_encoding["t_ideal_var"]
        t_var = exp_encoding["t_var"]

        P = self.ports
        IE = exp_map.keys()
        UE = [ u for i in IE for u in self.uops[i] ]

        conjuncts = []

        # there is a non-empty set of bottleneck ports
        conjuncts.append(Or([q_vars[k] for k in P]))
        # there is a non-empty set of bottleneck uops
        conjuncts.append(Or([j_vars[u] for u in UE]))
        # ports have non-negative pressure
        conjuncts.append(And([p_vars[k] >= 0 for k in P]))
        # no port has more pressure than the measured time
        conjuncts.append(And([p_vars[k] <= t_ideal_var for k in P]))
        # a port is a bottleneck port iff it has maximal pressure
        conjuncts.append(And([q_vars[k] == (p_vars[k] == t_ideal_var) for k in P]))
        # the total pressure on bottleneck ports is equal to the number of bottleneck uops
        conjuncts.append(Sum([If(j_vars[u], exp_map[i], 0) for i in IE for u in self.uops[i]]) == Sum([If(q_vars[k], t_ideal_var, 0) for k in P]))
        # bottleneck uops can only be executed on bottleneck ports
        conjuncts.append(And([And([Implies(And(j_vars[u], Not(q_vars[k])), Not(m_vars[(u, k)])) for k in P]) for u in UE]))
        # x vars are non-negative
        conjuncts.append(And([x_vars[(u, k)] >= 0 for u in UE for k in P]))
        # there has to be a way to execute all the uops of the experiment
        conjuncts.append(And([Sum([x_vars[(u, k)] for k in P]) == exp_map[i] for i in IE for u in self.uops[i]]))
        # a way to execute should also explain the port pressures
        conjuncts.append(And([Sum([x_vars[(u, k)] for u in UE]) == p_vars[k] for k in P]))
        # uops can only be executed on ports that are allowed according to the mapping
        conjuncts.append(And([Implies(x_vars[(u, k)] > 0, m_vars[(u, k)]) for u in UE for k in P]))

        if bottleneck_ipc is not None:
            # We really restrict instructions per cycle here, not uops per
            # cycle. In reality, there are limits on both, and they do not need
            # to be the same.
            exp_size = SumIfNecessary([v for i, v in exp_map.items()])
            if isinstance(exp_size, int):
                exp_size = float(exp_size)
            else:
                exp_size = ToReal(exp_size)
            min_cycles = exp_size / bottleneck_ipc
            conjuncts.append(t_var == If(t_ideal_var < min_cycles, min_cycles, t_ideal_var))
        else:
            conjuncts.append(t_var == t_ideal_var)


        constraint = And(conjuncts)
        if track_as is not None:
            assert assumption_var is None, "Encoding experiment with assumption variable and tracking id at the same time!"
            self.solver.assert_and_track(constraint, track_as)
        else:
            add_to_solver(self.solver, constraint, assumption_var)


class Mapping3Handler(MappingHandler):
    def __init__(self, synth):
        self.num_uops = synth.num_uops
        self.solver = synth.solver
        self.arch = synth.arch
        self.handled_class = synth.mapping_cls
        self.ports = list(range(synth.num_ports))
        self.uops = {
                i: [(i, n) for n in range(self.num_uops)]
                for i in self.arch.insn_list
            }

    def add_free_experiment_vars(self, insns):
        return { i: Int("new_exp_num_i{}".format(i)) for i in insns }

    def free_experiment_length(self, free_exp_vars):
        return Sum([v for i, v in free_exp_vars.items()])

    def encode_valid_free_experiment_constraints(self, exp_vars, assumption_var=None):
        geq_zero = And([ v >= 0 for i, v in exp_vars.items() ])
        add_to_solver(self.solver, geq_zero, assumption_var)

    def encode_free_experiment_bound(self, free_exp_vars, num_exp_insns, assumption_var=None):
        bound_constr = Sum([v for i, v in free_exp_vars.items()]) == num_exp_insns
        add_to_solver(self.solver, bound_constr, assumption_var)

    def extract_free_experiment(self, free_exp_vars, model):
        new_exp = []

        for i, v in free_exp_vars.items():
            num_i = model[v].as_long()
            for x in range(num_i):
                new_exp.append(i)

        return new_exp

    def add_mapping_vars(self, identifier):
        I = self.arch.insn_list
        P = self.ports
        n_vars = {
                (u): Bool("mapping_{}_n_u{}".format(identifier, str(u)))
                for i in I for u in self.uops[i]
            }
        m_vars = {
                (u, p): Bool("mapping_{}_m_u{}_p{}".format(identifier, str(u), str(p)))
                for i in I for u in self.uops[i] for p in P
            }
        encoding = { "n_vars": n_vars, "m_vars": m_vars}
        return encoding

    def encode_valid_mapping_constraints(self, encoding, assumption_var=None):
        I = self.arch.insn_list
        P = self.ports
        n_vars = encoding["n_vars"]
        m_vars = encoding["m_vars"]
        solver = self.solver

        # every instruction is decomposed into at least one uop
        add_to_solver(solver, And([Or([nv for (j, x), nv in n_vars.items() if j == i]) for i in I]), assumption_var)
        # every used uop has at least one port to execute on
        add_to_solver(solver, And([nv == Or([m_vars[(u, k)] for k in P]) for u, nv in n_vars.items()]), assumption_var)

        # uops are used densely
        for i in I:
            prev_nv = None
            for u, nv in n_vars.items():
                j, x = u
                if i != j:
                    continue
                if prev_nv is not None:
                    add_to_solver(solver, Or(prev_nv, Not(nv)), assumption_var)
                prev_nv = nv


    def encode_mapping(self, encoding, mapping, assumption_var=None):
        n_vars = encoding["n_vars"]
        m_vars = encoding["m_vars"]
        solver = self.solver

        for u, nv in n_vars.items():
            (i, x) = u
            us = mapping.assignment[i]
            if x < len(us):
                add_to_solver(self.solver, nv, assumption_var)
            else:
                add_to_solver(self.solver, Not(nv), assumption_var)

        for (u, k), mv in m_vars.items():
            (i, x) = u
            us = mapping.assignment[i]
            if x < len(us) and k in us[x]:
                add_to_solver(self.solver, mv, assumption_var)
            else:
                add_to_solver(self.solver, Not(mv), assumption_var)

    def decode_mapping(self, encoding, model):
        i2u = { (i, (i, x)): model[v] for ((i, x), v) in encoding["n_vars"].items() }
        u2p = { (u, k): model[v] for ((u, k), v) in encoding["m_vars"].items() }
        return self.handled_class.from_model(self.arch, len(self.ports), (i2u, u2p))

    def add_experiment_vars(self, insns, identifier):
        P = self.ports
        q_vars = {
                k: Bool("exp_{}_q_p{}".format(identifier, k))
                for k in P
            }
        j_vars = {
                u: Bool("exp_{}_j_i{}".format(identifier, str(u)))
                for i in insns for u in self.uops[i]
            }
        x_vars = {
                (u, k): Real("exp_{}_x_u{}_p{}".format(identifier, str(u), k))
                for i in insns for u in self.uops[i] for k in P
            }
        p_vars = {
                k: Real("exp_{}_p_p{}".format(identifier, k))
                for k in P
            }

        t_var = Real("exp_{}_t".format(identifier))

        res = dict()
        res["q_vars"] = q_vars
        res["j_vars"] = j_vars
        res["x_vars"] = x_vars
        res["p_vars"] = p_vars
        res["t_var"] = t_var
        return res

    def encode_experiment(self, mapping_encoding, exp_encoding, exp_map, *, assumption_var=None, track_as=None):
        def add(constraint):
            add_to_solver(self.solver, constraint, assumption_var)

        n_vars = mapping_encoding["n_vars"]
        m_vars = mapping_encoding["m_vars"]

        q_vars = exp_encoding["q_vars"]
        j_vars = exp_encoding["j_vars"]
        x_vars = exp_encoding["x_vars"]
        p_vars = exp_encoding["p_vars"]
        t_var = exp_encoding["t_var"]

        P = self.ports
        IE = exp_map.keys()
        UE = [ u for i in IE for u in self.uops[i] ]

        # there is a non-empty set of bottleneck ports
        add(Or([q_vars[k] for k in P]))
        # there is a non-empty set of bottleneck uops
        add(Or([j_vars[u] for u in UE]))
        # only used uops can be bottleneck uops
        add(And([Implies(j_vars[u], n_vars[u]) for u in UE]))
        # ports have non-negative pressure
        add(And([p_vars[k] >= 0 for k in P]))
        # no port has more pressure than the measured time
        add(And([p_vars[k] <= t_var for k in P]))
        # a port is a bottleneck port iff it has maximal pressure
        add(And([q_vars[k] == (p_vars[k] == t_var) for k in P]))
        # the total pressure on bottleneck ports is equal to the number of bottleneck uops
        add(Sum([If(j_vars[u], exp_map[i], 0) for i in IE for u in self.uops[i]]) == Sum([If(q_vars[k], t_var, 0) for k in P]))
        # bottleneck uops can only be executed on bottleneck ports
        add(And([And([Implies(And(j_vars[u], Not(q_vars[k])), Not(m_vars[(u, k)])) for k in P]) for u in UE]))
        # x vars are non-negative
        add(And([x_vars[(u, k)] >= 0 for u in UE for k in P]))
        # there has to be a way to execute all the uops of the experiment
        add(And([Implies(n_vars[u], Sum([x_vars[(u, k)] for k in P]) == exp_map[i]) for i in IE for u in self.uops[i]]))
        # uops that are not used should not be executed
        add(And([Implies(Not(n_vars[u]), x_vars[(u, k)] == 0) for u in UE for k in P]))
        # a way to execute should also explain the port pressures
        add(And([Sum([x_vars[(u, k)] for u in UE]) == p_vars[k] for k in P]))
        # uops can only be executed on ports that are allowed according to the mapping
        add(And([Implies(x_vars[(u, k)] > 0, m_vars[(u, k)]) for u in UE for k in P]))



class FullMultiplyMapping3Handler(MappingHandler):
    # Keep the integer multiplication in the formulas, use all different uops
    # (up to an optional maximal length)
    def __init__(self, synth):
        max_uop_length = synth.full_mul_uopsize_limit
        if max_uop_length <= 0:
            max_uop_length = synth.num_ports

        # self.num_uops = synth.num_uops
        self.solver = synth.solver
        self.arch = synth.arch
        self.handled_class = synth.mapping_cls
        self.ports = list(range(synth.num_ports))
        self.uops = [tuple(sorted(comb)) for l in range(1, max_uop_length+1) for comb in itertools.combinations(self.ports, l)]

    def add_free_experiment_vars(self, insns):
        return { i: Int("new_exp_num_i{}".format(i)) for i in insns }

    def free_experiment_length(self, free_exp_vars):
        return Sum([v for i, v in free_exp_vars.items()])

    def encode_valid_free_experiment_constraints(self, exp_vars, assumption_var=None):
        geq_zero = And([ v >= 0 for i, v in exp_vars.items() ])
        add_to_solver(self.solver, geq_zero, assumption_var)

    def encode_free_experiment_bound(self, free_exp_vars, num_exp_insns, assumption_var=None):
        bound_constr = self.free_experiment_length(free_exp_vars) == num_exp_insns
        add_to_solver(self.solver, bound_constr, assumption_var)

    def extract_free_experiment(self, free_exp_vars, model):
        new_exp = []

        for i, v in free_exp_vars.items():
            num_i = model[v].as_long()
            for x in range(num_i):
                new_exp.append(i)

        return new_exp

    def free_experiment_length(self, free_exp_vars):
        return Sum([v for i, v in free_exp_vars.items()])

    def add_mapping_vars(self, identifier):
        I = self.arch.insn_list
        P = self.ports
        n_vars = {
                (i, u): Int("mapping_{}_n_i{}_u{}".format(identifier, str(i), str(u)))
                for i in I for u in self.uops
            }
        encoding = { "n_vars": n_vars}
        return encoding

    def encode_valid_mapping_constraints(self, encoding, assumption_var=None):
        I = self.arch.insn_list
        P = self.ports
        n_vars = encoding["n_vars"]
        solver = self.solver

        # every instruction is decomposed into at least one uop
        add_to_solver(solver, And([Sum([nv for (j, x), nv in n_vars.items() if j == i]) >= 1 for i in I]), assumption_var)

        # no negative numbers of uops
        add_to_solver(solver, And([
                (nv >= 0) for k, nv in n_vars.items()
            ]))

    def encode_mapping(self, encoding, mapping, assumption_var=None):
        n_vars = encoding["n_vars"]
        solver = self.solver


        for (i, u), nv in n_vars.items():
            us = list(map(lambda x: tuple(sorted(x)), mapping.assignment[i]))
            count = us.count(u)
            add_to_solver(self.solver, nv == count, assumption_var)

    def decode_mapping(self, encoding, model):
        res = Mapping3(self.arch)
        for (i, u), nv in encoding["n_vars"].items():
            for x in range(model[nv].as_long()):
                res.assignment[i].append(list(u))
        return res

    def add_experiment_vars(self, insns, identifier):
        P = self.ports
        q_vars = {
                k: Bool("exp_{}_q_p{}".format(identifier, k))
                for k in P
            }
        j_vars = {
                u: Bool("exp_{}_j_u{}".format(identifier, str(u)))
                for u in self.uops
            }
        x_vars = {
                (u, k): Real("exp_{}_x_u{}_p{}".format(identifier, str(u), k))
                for u in self.uops for k in P
            }
        p_vars = {
                k: Real("exp_{}_p_p{}".format(identifier, k))
                for k in P
            }

        t_var = Real("exp_{}_t".format(identifier))

        res = dict()
        res["q_vars"] = q_vars
        res["j_vars"] = j_vars
        res["x_vars"] = x_vars
        res["p_vars"] = p_vars
        res["t_var"] = t_var
        return res

    def encode_experiment(self, mapping_encoding, exp_encoding, exp_map, *, assumption_var=None, track_as=None):
        def add(constraint):
            add_to_solver(self.solver, constraint, assumption_var)

        n_vars = mapping_encoding["n_vars"]

        q_vars = exp_encoding["q_vars"]
        j_vars = exp_encoding["j_vars"]
        x_vars = exp_encoding["x_vars"]
        p_vars = exp_encoding["p_vars"]
        t_var = exp_encoding["t_var"]

        P = self.ports
        IE = exp_map.keys()
        UE = [ u for u in self.uops ]

        # there is a non-empty set of bottleneck ports
        add(Or([q_vars[k] for k in P]))

        # there is a non-empty set of bottleneck uops
        add(Or([j_vars[u] for u in UE]))

        # only used uops can be bottleneck uops
        # add(And([Implies(j_vars[u], n_vars[u] >= 1) for u in UE]))

        # ports have non-negative pressure
        add(And([p_vars[k] >= 0 for k in P]))

        # no port has more pressure than the measured time
        add(And([p_vars[k] <= t_var for k in P]))

        # a port is a bottleneck port iff it has maximal pressure
        add(And([q_vars[k] == (p_vars[k] == t_var) for k in P]))

        # bottleneck uops can only be executed on bottleneck ports
        add(And([And([Implies(j_vars[u], q_vars[k]) for k in u]) for u in UE]))

        # the total pressure on bottleneck ports is equal to the number of bottleneck uops
        add(Sum([If(j_vars[u], exp_map[i] * n_vars[(i, u)], 0) for i in IE for u in self.uops]) == Sum([If(q_vars[k], t_var, 0) for k in P]))

        # x vars are non-negative
        add(And([x_vars[(u, k)] >= 0 for u in UE for k in P]))

        # there has to be a way to execute all the uops of the experiment
        add(And([Sum([x_vars[(u, k)] for k in u]) == Sum([exp_map[i] * n_vars[(i, u)] for i in IE]) for u in self.uops]))

        # a way to execute should also explain the port pressures
        add(And([Sum([x_vars[(u, k)] for u in UE]) == p_vars[k] for k in P]))
        # uops can only be executed on ports that are allowed according to the mapping
        add(And([x_vars[(u, k)] == 0 for u in UE for k in P if k not in u]))



def mymax(a, b):
    return If(a > b, a, b)

def list_max_linear(ls):
    res = 0
    for l in ls:
        res = mymax(l, res)
    return res

def list_max_tree(ls):
    if len(ls) == 0:
        return 0
    if len(ls) == 1:
        return ls[0]
    middle = len(ls) // 2
    return mymax(list_max_tree(ls[:middle]), list_max_tree(ls[middle:]))

def to_portlist(u):
    res = []
    idx = 0
    while u > 0:
        if u & 1:
            res.append(idx)
        u = u >> 1
        idx += 1

    return res


class BNMapping3HandlerBitBlasted(MappingHandler):
    def __init__(self, synth):
        self.solver = synth.solver
        self.arch = synth.arch
        self.handled_class = synth.mapping_cls
        assert self.handled_class == Mapping3
        self.ports = list(range(synth.num_ports))
        num_uops = (2 ** synth.num_ports) - 1 # we don't take the one from the config, since the corresponding restriction is not shared here
        self.port_subsets = list(range(1, num_uops + 1))    # those may not be restricted

        # those may be restricted
        self.uops = list(range(1, num_uops + 1))

        # we could decompose either the mappings variables or the free experiment variables to bits.
        self.mapping_bits = synth.config['smt_bn_handler_bits']
        # This is a restriction on what experiments can look like.
        # In particular, if experiments are defined in terms of variables, this can rule out valid cases!

        self.temp_counter = 0

    def filter_uops(self, pred):
        self.uops = [ x for x in self.port_subsets if pred(x) ]

    def add_free_experiment_vars(self, insns):
        return { i: [ Bool("new_exp_num_i{}_b{}".format(i, b)) for b in range(self.mapping_bits) ] for i in insns }
        # return { i: Int("new_exp_num_i{}".format(i)) for i in insns }

    def free_experiment_length(self, free_exp_vars):
        raise NotImplemented("free_experiment_length is not yet implemented for this mapping handler")
        # return Sum([v for i, v in free_exp_vars.items()])

    def encode_valid_free_experiment_constraints(self, exp_vars, assumption_var=None):
        # the bit encoding can only encode non-negative numbers, so nothing required here
        # geq_zero = And([ v >= 0 for i, v in exp_vars.items() ])
        # add_to_solver(self.solver, geq_zero, assumption_var)
        pass

    def encode_free_experiment_bound(self, free_exp_vars, num_exp_insns, assumption_var=None):
        bound_constr = SumIfNecessary([If(v, (2 ** b), 0) for i, bmap in free_exp_vars.items() for b, v in enumerate(bmap)]) == num_exp_insns
        # bound_constr = SumIfNecessary([v for i, v in free_exp_vars.items()]) == num_exp_insns
        add_to_solver(self.solver, bound_constr, assumption_var)

    def extract_free_experiment(self, free_exp_vars, model):
        new_exp = []

        for i, bmap in free_exp_vars.items():
            for b, v in enumerate(bmap):
                if model[v]:
                    for x in range(2**b):
                        new_exp.append(i)

        # for i, v in free_exp_vars.items():
        #     num_i = model[v].as_long()
        #     for x in range(num_i):
        #         new_exp.append(i)

        return new_exp

    def add_mapping_vars(self, identifier):
        I = self.arch.insn_list
        n_vars = {
                (i, u): Int("mapping_{}_n_i{}_u{}".format(identifier, str(i), str(u)))
                for i in I for u in self.uops
            }
        encoding = { "n_vars": n_vars }
        return encoding

    def encode_valid_mapping_constraints(self, encoding, assumption_var=None):
        I = self.arch.insn_list
        n_vars = encoding["n_vars"]
        solver = self.solver

        # every instruction is decomposed into at least one uop

        # TODO this can be done with a sum or with an OR
        valid_mapping_use_sum = True

        if valid_mapping_use_sum:
            add_to_solver(solver, And([SumIfNecessary([nv for (j, x), nv in n_vars.items() if j == i]) >= 1 for i in I]), assumption_var)
        else:
            add_to_solver(solver, And([Or([(nv >= 1) for (j, x), nv in n_vars.items() if j == i]) for i in I]), assumption_var)

        # no negative numbers of uops
        add_to_solver(solver, And([
                (nv >= 0) for k, nv in n_vars.items()
            ]))

    def encode_mapping(self, encoding, mapping, assumption_var=None):
        n_vars = encoding["n_vars"]
        solver = self.solver

        iu2val = defaultdict(lambda: 0)
        for i, us in mapping.assignment.items():
            for u in us:
                uval = 0
                for p in u:
                    uval += 2 ** p
                iu2val[(i, uval)] += 1

        for iu, nv in n_vars.items():
            add_to_solver(self.solver, nv == iu2val.get(iu, 0), assumption_var)

    def encode_portusage_constraint(self, encoding, insn, portusage, assumption_var=None):
        """ Add a constraint that states that the given instruction needs to
        use exactly the given port usage.
        The portusage should be given as a dictionary mapping port-tuples to
        numbers. Non-present keys are considered 0.
        """
        n_vars = encoding["n_vars"]
        solver = self.solver

        # translate the portusage to uop bitvectors
        constraint_map = dict()

        for u, v in portusage.items():
            assert v >= 0
            uval = 0
            for p in u:
                uval += 2 ** p
            constraint_map[uval] = v

        for (i, u), nv in n_vars.items():
            if i != insn:
                continue
            constraint_val = constraint_map.get(u, 0)
            add_to_solver(self.solver, nv == constraint_val, assumption_var)

    def decode_mapping(self, encoding, model):
        res = self.handled_class(self.arch)
        for (i, u), nv in encoding["n_vars"].items():
            nval = model[nv].as_long()
            if nval > 0:
                ps = to_portlist(u)
                for x in range(nval):
                    res.assignment[i].append(ps)

        return res

    def add_experiment_vars(self, insns, identifier):
        P = self.ports
        v_vars = {
                u: Real("exp_{}_v_u{}".format(identifier, str(u))) for u in self.uops
            }

        t_var = Real("exp_{}_t".format(identifier))

        res = dict()
        res["v_vars"] = v_vars
        res["t_var"] = t_var
        return res

    def encode_experiment(self, mapping_encoding, exp_encoding, exp_map, *, assumption_var=None, track_as=None):
        def add(constraint):
            add_to_solver(self.solver, constraint, assumption_var)

        n_vars = mapping_encoding["n_vars"]
        v_vars = exp_encoding["v_vars"]

        # TODO do we want integer or real addition?

        n_vars_per_uop = defaultdict(dict)
        for (i, u), nv in n_vars.items():
            n_vars_per_uop[u][i] = nv

        # fill the v variables with the total mass per uop
        for u in self.uops:
            exprs = []
            for i, nv in n_vars_per_uop[u].items():
                emap = exp_map.get(i, 0)
                if isinstance(emap, int): # a constant
                    if emap != 0:
                        exprs.append(nv * emap)
                elif isinstance(emap, list) or isinstance(emap, tuple): # a variable that is properly decomposed into bits
                    for b in range(self.mapping_bits):
                        exprs.append(If(emap[b], (2**b) * nv, 0))
                else: # anything else, i.e., an int-valued variable
                    # TODO this can be a problem for PMEvo exps
                    temp_vars = [ Bool("temp_exp{}_i{}_b{}".format(self.temp_counter, i, b)) for b in range(self.mapping_bits) ]
                    self.temp_counter += 1

                    add(emap == SumIfNecessary([If(temp_vars[b], (2**b), 0) for b in range(self.mapping_bits)]))
                    for b in range(self.mapping_bits):
                        exprs.append(If(temp_vars[b], (2**b) * nv, 0))

            add(v_vars[u] == SumIfNecessary(exprs))

        max_terms = []
        for q in self.port_subsets:
            sum_terms = []
            for u in self.uops:
                if (u & ~q) == 0: # u is completely contained in q
                    sum_terms.append(v_vars[u])
            if len(sum_terms) == 0:
                continue
            denominator = popcount(q)
            max_terms.append(SumIfNecessary(sum_terms) / denominator)

        t_var = exp_encoding["t_var"]

        # add(t_var == list_max_linear(max_terms))
        add(t_var == list_max_tree(max_terms))


class BNMapping3Handler(MappingHandler):
    def __init__(self, synth):
        self.solver = synth.solver
        self.arch = synth.arch
        self.handled_class = synth.mapping_cls
        assert self.handled_class == Mapping3
        self.ports = list(range(synth.num_ports))
        num_uops = (2 ** synth.num_ports) - 1 # we don't take the one from the config, since the corresponding restriction is not shared here
        self.port_subsets = list(range(1, num_uops + 1))    # those may not be restricted

        # those may be restricted
        self.uops = list(range(1, num_uops + 1))

        # we could decompose either the mappings variables or the free experiment variables to bits.
        # self.mapping_bits = synth.config['smt_bn_handler_bits']
        # This is a restriction on what experiments can look like.
        # In particular, if experiments are defined in terms of variables, this can rule out valid cases!

        self.temp_counter = 0

    def filter_uops(self, pred):
        self.uops = [ x for x in self.port_subsets if pred(x) ]

    def add_free_experiment_vars(self, insns):
        return { i: Int("new_exp_num_i{}".format(i)) for i in insns }

    def encode_valid_free_experiment_constraints(self, exp_vars, assumption_var=None):
        geq_zero = And([ v >= 0 for i, v in exp_vars.items() ])
        add_to_solver(self.solver, geq_zero, assumption_var)

    def encode_free_experiment_bound(self, free_exp_vars, num_exp_insns, assumption_var=None):
        bound_constr = SumIfNecessary([v for i, v in free_exp_vars.items()]) == num_exp_insns
        add_to_solver(self.solver, bound_constr, assumption_var)

    def free_experiment_length(self, free_exp_vars):
        return Sum([v for i, v in free_exp_vars.items()])

    def extract_free_experiment(self, free_exp_vars, model):
        new_exp = []

        for i, v in free_exp_vars.items():
            num_i = model[v].as_long()
            for x in range(num_i):
                new_exp.append(i)

        return new_exp

    def add_mapping_vars(self, identifier):
        I = self.arch.insn_list
        n_vars = {
                (i, u): Int("mapping_{}_n_i{}_u{}".format(identifier, str(i), str(u)))
                for i in I for u in self.uops
            }
        encoding = { "n_vars": n_vars }
        return encoding

    def encode_valid_mapping_constraints(self, encoding, assumption_var=None):
        I = self.arch.insn_list
        n_vars = encoding["n_vars"]
        solver = self.solver

        # every instruction is decomposed into at least one uop

        # TODO this can be done with a sum or with an OR
        valid_mapping_use_sum = True

        if valid_mapping_use_sum:
            add_to_solver(solver, And([SumIfNecessary([nv for (j, x), nv in n_vars.items() if j == i]) >= 1 for i in I]), assumption_var)
        else:
            add_to_solver(solver, And([Or([(nv >= 1) for (j, x), nv in n_vars.items() if j == i]) for i in I]), assumption_var)

        # no negative numbers of uops
        add_to_solver(solver, And([
                (nv >= 0) for k, nv in n_vars.items()
            ]))

    def encode_mapping(self, encoding, mapping, assumption_var=None):
        n_vars = encoding["n_vars"]
        solver = self.solver

        iu2val = defaultdict(lambda: 0)
        for i, us in mapping.assignment.items():
            for u in us:
                uval = 0
                for p in u:
                    uval += 2 ** p
                iu2val[(i, uval)] += 1

        for iu, nv in n_vars.items():
            add_to_solver(self.solver, nv == iu2val.get(iu, 0), assumption_var)

    def encode_portusage_constraint(self, encoding, insn, portusage, assumption_var=None):
        """ Add a constraint that states that the given instruction needs to
        use exactly the given port usage.
        The portusage should be given as a dictionary mapping port-tuples to
        numbers. Non-present keys are considered 0.
        """
        n_vars = encoding["n_vars"]
        solver = self.solver

        # translate the portusage to uop bitvectors
        constraint_map = dict()

        for u, v in portusage.items():
            assert v >= 0
            uval = 0
            for p in u:
                uval += 2 ** p
            constraint_map[uval] = v

        for (i, u), nv in n_vars.items():
            if i != insn:
                continue
            constraint_val = constraint_map.get(u, 0)
            add_to_solver(self.solver, nv == constraint_val, assumption_var)

    def decode_mapping(self, encoding, model):
        res = self.handled_class(self.arch)
        for (i, u), nv in encoding["n_vars"].items():
            nval = model[nv].as_long()
            if nval > 0:
                ps = to_portlist(u)
                for x in range(nval):
                    res.assignment[i].append(ps)

        return res

    def add_experiment_vars(self, insns, identifier):
        P = self.ports
        v_vars = {
                u: Real("exp_{}_v_u{}".format(identifier, str(u))) for u in self.uops
            }

        # true iff port k is part of the bottleneck
        bottleneck_port_vars = {
                k: Bool("exp_{}_bottleneck_port_k{}".format(identifier, str(k))) for k in self.ports
            }

        num_uops_var = Real("exp_{}_num_uops".format(identifier))

        t_var = Real("exp_{}_t".format(identifier))

        res = dict()
        res["v_vars"] = v_vars
        res["bottleneck_port_vars"] = bottleneck_port_vars
        res["t_var"] = t_var
        res["num_uops_var"] = num_uops_var
        return res

    def encode_experiment(self, mapping_encoding, exp_encoding, exp_map, *, assumption_var=None, track_as=None):
        def add(constraint):
            add_to_solver(self.solver, constraint, assumption_var)

        n_vars = mapping_encoding["n_vars"]
        v_vars = exp_encoding["v_vars"]

        # TODO do we want integer or real addition?

        n_vars_per_uop = defaultdict(dict)
        for (i, u), nv in n_vars.items():
            n_vars_per_uop[u][i] = nv

        # fill the v variables with the total mass per uop
        for u in self.uops:
            exprs = []
            for i, nv in n_vars_per_uop[u].items():
                emap = exp_map.get(i, 0)
                if isinstance(emap, int): # a constant
                    if emap != 0:
                        exprs.append(nv * emap)
                elif isinstance(emap, list) or isinstance(emap, tuple): # a variable that is properly decomposed into bits
                    assert False
                else: # anything else, i.e., an int-valued variable
                    exprs.append(emap * nv)

            add(v_vars[u] == SumIfNecessary(exprs))

        max_terms = []
        per_ports = defaultdict(list)
        for q in self.port_subsets:
            sum_terms = []
            for u in self.uops:
                if (u & ~q) == 0: # u is completely contained in q
                    sum_terms.append(v_vars[u])
            if len(sum_terms) == 0:
                continue
            denominator = popcount(q)
            quotient_term = SumIfNecessary(sum_terms) / denominator
            max_terms.append(quotient_term)
            for k in self.ports:
                if (1 << k) & q != 0: # port k is in q
                    per_ports[k].append(quotient_term)

        t_var = exp_encoding["t_var"]

        # add(t_var == list_max_linear(max_terms))
        add(t_var == list_max_tree(max_terms))

        num_uops_var = exp_encoding["num_uops_var"]
        add(num_uops_var == SumIfNecessary([v for u, v in v_vars.items()]))

        for k, term_list in per_ports.items():
            # k is a bottleneck port iff it is part of a port subset whose
            # quotient term is equal to the overall throughput
            add(Or([term == t_var for term in term_list]) == exp_encoding["bottleneck_port_vars"][k])

