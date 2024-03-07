""" The core algorithm of our relaxed port mapping inference algorithm.

The algorithm is discussed in the article "Explainable Port Mapping Inference
with Sparse Performance Counters for AMD's Zen Architectures".

The entry point is the `find_mapping` function in this file.
"""

from collections import defaultdict
from copy import deepcopy, copy
import itertools
import json
import math
import random
import textwrap


from pmtestbench.common.processors import Processor
from pmtestbench.common.architecture import Architecture
from pmtestbench.common.portmapping import Mapping3
from pmtestbench.common.synthesizers import Synthesizer
from .utils import *

import pmtestbench.common.processors.portmapping_processor as pmproc

import pmtestbench.relaxeduops.checkpointer as cp

import logging
logger = logging.getLogger(__name__)


def measure_singletons(params, proc, rejects=None):
    """ Run experiments with all instructions individually and return the
    results with a list of instructions whose results indicates they are
    feasible.
    """
    arch = proc.get_arch()
    insns = arch.insn_list

    insns_with_measurements = []

    iseqs = []
    for i in insns:
        iseqs.append([i])

    results = proc.execute_batch(iseqs)

    for i, res in zip(insns, results):
        # run singleton experiments for all instructions
        cycles = res.get('cycles', -1.0)
        if cycles <= 0.0:
            msg = "Instruction with non-positive inverse throughput ({}):\n  {}".format(cycles, i)
            if rejects is not None:
                rejects.append(RejectedInsn(i, msg))
            logger.warning(msg)
            continue
        num_uops = res.get('num_uops', -1.0)
        if num_uops <= 0.6:
            msg = "Instruction with implausible number of uops ({}):\n  {}".format(num_uops, i)
            if rejects is not None:
                rejects.append(RejectedInsn(i, msg))
            logger.warning(msg)
            continue

        if cycles > num_uops + params.num_uops_epsilon:
            msg = "Instruction runs in more cycles ({}) than it has uops ({}):\n  {}".format(cycles, num_uops, i)
            if rejects is not None:
                rejects.append(RejectedInsn(i, msg))
            logger.warning(msg)
            continue

        rounded_num_uops = round(num_uops)
        if abs(rounded_num_uops - num_uops) > params.num_uops_epsilon:
            msg = "Instruction with significantly non-integer number of uops ({}):\n  {}".format(num_uops, i)
            if rejects is not None:
                rejects.append(RejectedInsn(i, msg))
            logger.warning(msg)
            continue
        num_uops = int(rounded_num_uops)

        if cycles > params.insn_skip_cycles:
            msg = "Instruction executes in too many cycles ({}):\n  {}".format(cycles, i)
            if rejects is not None:
                rejects.append(RejectedInsn(i, msg))
            logger.warning(msg)
            continue
        insns_with_measurements.append(InsnWithMeasurement(insn=i, cycles=cycles, num_uops=num_uops))

    return insns_with_measurements

def check_instruction_pairs(params, proc, insns_with_measurements, rejects=None):
    """ Measure all pairs of instructions and check whether the results are
    consistent with the theoretical port mapping model and the singleton
    measurements.
    """
    insns = { iwm.insn: iwm for iwm in insns_with_measurements }

    iseqs = []

    for i1, i2 in itertools.combinations_with_replacement(insns.keys(), 2):
        iseqs.append([i1, i2])

    results = proc.execute_batch(iseqs)

    for (i1, i2), res in zip(itertools.combinations_with_replacement(insns.keys(), 2), results):
        exp = [i1, i2]

        cycles = res.get('cycles', -1.0)
        if cycles <= 0.0:
            msg = "Experiment with non-positive inverse throughput ({}):\n  {}".format(cycles, exp)
            if rejects is not None:
                rejects.append(RejectedExp(exp, msg))
            logger.warning(msg)
            continue
        num_uops = res.get('num_uops', -1.0)
        if num_uops <= 0.6:
            msg = "Experiment with implausible number of uops ({}):\n  {}".format(num_uops, exp)
            if rejects is not None:
                rejects.append(RejectedExp(exp, msg))
            logger.warning(msg)
            continue

        rounded_num_uops = round(num_uops)
        if abs(rounded_num_uops - num_uops) > params.num_uops_epsilon:
            msg = "Experiment with significantly non-integer number of uops ({}):\n  {}".format(num_uops, exp)
            if rejects is not None:
                rejects.append(RejectedExp(exp, msg))
            logger.warning(msg)
            continue
        num_uops = int(rounded_num_uops)

        assert not params.use_cpi, "these checks are not built for a CPI epsilon a would need to be rethought for the purpose"

        max_cycles = max(insns[i].cycles for i in exp)
        if cycles < max_cycles - params.tp_epsilon:
            msg = "Experiment requires fewer cycles than one of its components ({} vs. {}):\n  {}".format(cycles, max_cycles, exp)
            if rejects is not None:
                rejects.append(RejectedExp(exp, msg))
            logger.warning(msg)
            continue

        sum_cycles = sum(insns[i].cycles for i in exp)
        if cycles > sum_cycles + params.tp_epsilon:
            msg = "Experiment requires more cycles than the sum of its components ({} vs. {}):\n  {}".format(cycles, sum_cycles, exp)
            if rejects is not None:
                rejects.append(RejectedExp(exp, msg))
            logger.warning(msg)
            continue

        sum_uops = sum(insns[i].num_uops for i in exp)
        if num_uops > sum_uops:
            msg = "Experiment requires a different number of uops than the sum of its components ({} vs. {}):\n  {}".format(num_uops, sum_uops, exp)
            if rejects is not None:
                rejects.append(RejectedExp(exp, msg))
            logger.warning(msg)
            continue


def characterize_blocking_insn_candidates(params, proc, insns_with_measurements, rejects=None):
    """ Identify all instructions that use only one uop and determine, on how
    many ports that uop can execute.
    """

    candidates_per_num_ports = defaultdict(list)

    rejected_indices = []

    for idx, meas in enumerate(insns_with_measurements):
        i = meas.insn
        cycles = meas.cycles
        num_uops = meas.num_uops

        if num_uops > 1:
            logger.debug("Instruction is not a blocking instruction because it has {} uops:\n  {}".format(num_uops, i))
            continue

        assert num_uops == 1

        # Uops should be distributed uniformly to all available ports, so the
        # inverse throughput of an experiment with only a single uop should be
        # 1 divided by the number of available ports for the uop.
        num_ports = 1 / cycles
        rounded_num_ports = round(num_ports)

        if rounded_num_ports < 1.0:
            # we need to check this first, to avoid divide-by-zero errors for the CPI epsilon comparison
            rejected_indices.append(idx)
            msg = "Instruction with less than one port ({}, probably not fully pipelined):\n  {}".format(num_ports, i)
            if rejects is not None:
                rejects.append(RejectedInsn(i, msg))
            logger.warning(msg)
            continue

        if params.use_cpi:
            # CPI is here equal to the cycles, since the experiment contains
            # only a single instruction.
            is_significantly_different = abs((1/rounded_num_ports) - (1/num_ports)) > params.num_ports_epsilon
        else:
            # this is effectively a bound to the IPC error
            is_significantly_different = abs(rounded_num_ports - num_ports) > params.num_ports_epsilon
        if is_significantly_different:
            rejected_indices.append(idx)
            msg = "Instruction with significantly non-integer number of ports ({}):\n  {}".format(num_ports, i)
            if rejects is not None:
                rejects.append(RejectedInsn(i, msg))
            logger.warning(msg)
            continue

        num_ports = rounded_num_ports

        logger.debug("Instruction considered as a blocking instruction with {} ports:\n  {}".format(num_ports, i))
        candidates_per_num_ports[num_ports].append(i)

    # remove instructions that we rejected here, going from last to first so
    # that indices are not invalidated before they are processed
    rejected_indices.reverse()
    for idx in rejected_indices:
        insns_with_measurements.pop(idx)

    return candidates_per_num_ports


def filter_equal_blocking_insns(params, proc, candidates_per_num_ports, rejects=None):
    """ Restrict blocking instruction candidates such that no two block the
    same set of ports.
    """
    blocking_insns_per_num_ports = dict()
    occuring_uop_widths = list(sorted(candidates_per_num_ports.keys()))

    equivalence_map = defaultdict(list)

    for width in occuring_uop_widths:
        unique_blockinsns = []
        candidates = candidates_per_num_ports[width]

        for c in candidates:
            for u in unique_blockinsns:
                factor = 1
                experiment = factor * [u, c]
                res = proc.execute(experiment)

                num_uops = round(res.get('num_uops', -1))
                if num_uops != factor * 2:
                    msg = "combining two blocking instruction candidates leads to unexpected number of uops ({} for {} instances): {}".format(num_uops, factor, experiment)
                    if rejects is not None:
                        rejects.append(RejectedExp(experiment, msg))
                    logger.warning(msg)
                    break

                cycles = res['cycles']
                if params.use_cpi:
                    diff = (cycles / len(experiment)) - (1/width)
                    # All involved instructions should execute on width ports,
                    # leading to a reference CPI of (1/width) in the worst case.
                else:
                    expected = (len(experiment) / width)
                    diff = cycles - expected
                if diff > params.tp_epsilon:
                    msg = "experiment took more cycles than the sum of its components ({}): {}".format(cycles, experiment)
                    if rejects is not None:
                        rejects.append(RejectedExp(experiment, msg))
                    logger.warning(msg)
                    break
                if abs(diff) <= params.tp_epsilon:
                    logger.debug("blocking instructions found equivalent: {u} and {c} -> We use {u}".format(u=u, c=c))
                    equivalence_map[u].append(c)
                    break
            else:
                # if c is not equivalent to any chosen instruction so far
                logger.info("selected unique blocking instruction: {c}".format(c=c))
                unique_blockinsns.append(c)

            blocking_insns_per_num_ports[width] = unique_blockinsns
    return blocking_insns_per_num_ports, equivalence_map


def measure_external_bottleneck(params, proc, blocking_insns_per_num_ports, rejects=None):
    """ Measure the peak achievable instructions executed per cycle (IPC), so
    that we can make the following steps aware of it.
    """
    # TODO it could be necessary to do that before we selected blocking
    # instructions in the previous step

    num_shuffles = 4
    def measure_ipc(exp_dict):
        linearized_exp = []
        for i, n in exp_dict.items():
            for x in range(n):
                linearized_exp.append(i)

        exps = []
        for x in range(num_shuffles):
            random.shuffle(linearized_exp)
            exps.append(deepcopy(linearized_exp))

        results = proc.execute_batch(exps)

        minimal_cycles = math.inf
        for exp, res in zip(exps, results):
            cycles = res['cycles']
            if cycles <= 0:
                msg = "error when running IPC experiment '{}':\n{}".format(exp, res)
                if rejects is not None:
                    rejects.append(RejectedExp(exp, msg))
                logger.warning(msg)
                return None
            if cycles < minimal_cycles:
                minimal_cycles = cycles

        return len(exp) / minimal_cycles


    num_ports_per_blocking_insn = { i: n for n, ls in blocking_insns_per_num_ports.items() for i in ls}

    observed_peak_ipcs = []

    num_order_shuffles = 8

    for x in range(num_order_shuffles):
        insn_order = list(sorted(num_ports_per_blocking_insn.keys()))
        random.shuffle(insn_order)

        while True:
            assert len(insn_order) > 1
            current_exp = defaultdict(int)
            seed_insn = insn_order[0]
            num_ports = num_ports_per_blocking_insn[seed_insn]
            current_exp[seed_insn] = num_ports
            insn_order = insn_order[1:]
            current_ipc = measure_ipc(current_exp)
            if params.use_cpi:
                is_sufficiently_equal = abs((1/current_ipc) - (1/num_ports)) <= params.num_ports_epsilon
            else:
                is_sufficiently_equal = abs(current_ipc - num_ports) <= params.num_ports_epsilon
            assert is_sufficiently_equal, "unexpected IPC for individual instruction: got {}, expected {}".format(current_ipc, num_ports)
            if current_ipc is not None:
                break

        for i in insn_order:
            while True:
                new_exp = copy(current_exp)
                new_exp[i] += 1
                new_ipc = measure_ipc(new_exp)
                if new_ipc is None:
                    continue
                if new_ipc > current_ipc:
                    current_exp = new_exp
                    current_ipc = new_ipc
                else:
                    break

        logger.info("observed peak instructions per cycle (IPC): {}\n with experiment:{}".format(
                current_ipc, json.dumps(current_exp, indent=2)))

        observed_peak_ipcs.append((current_ipc, current_exp))

    # TODO improvement: check if that is equal to the num_ports of a blocking
    # instruction, that would be problematic for the previous stage

    sorted_ipcs = sorted(observed_peak_ipcs, key=lambda x: x[0])
    # optimal_ipc, optimal_exp = sorted_ipcs[-1] # take the maximum
    optimal_ipc, optimal_exp = sorted_ipcs[len(sorted_ipcs)//2] # take the median

    logger.info("chosen observed peak instructions per cycle (IPC): {}\n with experiment:{}".format(
            optimal_ipc, json.dumps(optimal_exp, indent=2)))

    return  optimal_ipc


def compute_mapping_for_blocking_insns(params, proc, blocking_insns_per_num_ports, bottleneck_ipc, *,
                                       improper_blockinginsns_with_num_uops=None,
                                       rejects=None, dump_experiments=None):
    """ Compute the port mapping of the blocking instructions.
    """

    if improper_blockinginsns_with_num_uops is None:
        improper_blockinginsns_with_num_uops = {}
    # We encode the improper blocking instructions as shape for the constrained
    # mapping in the synth config.

    synth_config = {
            "synthesizer_kind": "smt",
            "mapping_class": "Mapping3",
            "num_ports": params.num_ports,

            "smt_use_constrained_mapping3": True,
            "smt_constrain_improper_uops": True,
            "num_uops_per_insn": improper_blockinginsns_with_num_uops,

            "smt_slack_val": params.smt_slack_val,
            "smt_slack_kind": params.smt_slack_kind,
            "smt_insn_bound": params.smt_insn_bound,
            "smt_exp_limit_strategy": "incremental_optimistic",
            "smt_dump_constraints": False,
            # "smt_dump_constraints": True, # more debugging
        }

    synth = Synthesizer(config=synth_config)

    all_blocking_insns = []
    for k, insns in blocking_insns_per_num_ports.items():
        all_blocking_insns += insns

    for insn, num_uops in improper_blockinginsns_with_num_uops.items():
        all_blocking_insns.append(insn)

    restricted_proc = proc.get_restricted(all_blocking_insns)

    known_portset_sizes = { str(i): n for n, ls in blocking_insns_per_num_ports.items() for i in ls }

    if params.use_bottleneck_ipc:
        used_bottleneck_ipc = bottleneck_ipc
    else:
        used_bottleneck_ipc = None

    m = synth.synthesize(restricted_proc,
                         known_portset_sizes=known_portset_sizes,
                         bottleneck_ipc=used_bottleneck_ipc,
                        )

    if m is None:
        raise RuntimeError("Could not find a satifying characterization of the bottleneck instructions")

    # we return the experiment list, so that we can validate a preferred
    # mapping against it in the next stage
    elist = synth.impl.get_experimentlist()

    # dump the mapping and the experiments, for why-not analysis
    if dump_experiments is not None:
        with open(dump_experiments / "synth_inputs.json", "w") as f:
            json.dump({
                    'config': synth_config,
                    'bottleneck_ipc': float(used_bottleneck_ipc), # json can't handle decimals, this conversion might cause inaccuracies
                    'known_portset_sizes': known_portset_sizes,
                }, f, indent=2)
        with open(dump_experiments / "mapping.json", "w") as f:
            m.to_json(f)
        with open(dump_experiments / "experiments.json", "w") as f:
            elist.to_json(f)

    occuring_uop_widths = list(sorted(blocking_insns_per_num_ports.keys()))
    ports_for = dict()

    # handle the proper blocking instructions
    blocked_uops = set()
    for curr_width in occuring_uop_widths:
        for curr_insn in blocking_insns_per_num_ports[curr_width]:
            port_usage = m.assignment[curr_insn]
            # these assertions should be guaranteed by the SMT formulation
            assert len(port_usage) == 1
            ports_for_curr_insn = frozenset(next(iter(port_usage)))
            assert len(ports_for_curr_insn) == curr_width, f"The found mapping uses {len(ports_for_curr_insn)} ports for the instruction '{curr_insn}', which we determined to use {curr_width} ports."
            ports_for[curr_insn] = ports_for_curr_insn
            blocked_uops.add(ports_for_curr_insn)

    # handle the improper blocking instructions
    secondary_uops = {}
    for insn, num_uops in improper_blockinginsns_with_num_uops.items():
        port_usage = m.assignment[insn]
        # these assertions should be guaranteed by the SMT formulation
        assert len(port_usage) == num_uops
        set_port_usage = [ frozenset(uop) for uop in port_usage ]
        not_covered_port_usage = [ uop for uop in set_port_usage if uop not in blocked_uops ]
        assert len(not_covered_port_usage) == 1

        # we take insn as an (improper) blocking instruction for this port set
        ports_for[insn] = not_covered_port_usage[0]

        # these secondary uops come with the improper blocking instruction
        secondary_uops[insn] = [ uop for uop in set_port_usage if uop in blocked_uops ]
        # the SMT formulation ensures that these are all uops of the port_usage
        # except not_covered_port_usage[0]

    return ports_for, secondary_uops, synth_config, elist

def validate_preferred_mapping(params, proc, synth_config, elist, blocking_insns_per_num_ports, bottleneck_ipc, preferred_mapping, rejects=None):
    """ Check that the preferred mapping is valid for the blocking instructions.
    """

    # the synth_config contains the shape of the constrained mapping3
    synth = Synthesizer(config=synth_config)

    num_uops_per_insn = synth_config["num_uops_per_insn"]

    all_blocking_insns = []
    for k, insns in blocking_insns_per_num_ports.items():
        all_blocking_insns += insns

    restricted_proc = proc.get_restricted(all_blocking_insns)

    # improper blocking instructions are contained in the
    # blocking_insns_per_num_ports mapping here, but we don't want them in the
    # known_portset_sizes (since their portset sizes were infered by the
    # algorithm and need not be mandatory).
    known_portset_sizes = { str(i): n for n, ls in blocking_insns_per_num_ports.items() for i in ls if num_uops_per_insn.get(i, 1) == 1 }

    if params.use_bottleneck_ipc:
        used_bottleneck_ipc = bottleneck_ipc
    else:
        used_bottleneck_ipc = None

    # convert the preferred Mapping2 with optional secondary_uops field to a
    # Mapping3.
    preferred_mapping3 = Mapping3(preferred_mapping.arch)

    secondary_uops = getattr(preferred_mapping, "secondary_uops", {})
    secondary_uops = { k: list(map(frozenset, v)) for k, v in secondary_uops.items()}
    for insn, portset in preferred_mapping.assignment.items():
        preferred_mapping3.assignment[insn] = [portset]
        secondary_uops_for_insns = secondary_uops.get(insn, [])
        for uop in secondary_uops_for_insns:
            preferred_mapping3.assignment[insn].append(uop)

    is_ok = synth.impl.why_not(preferred_mapping3,
                               elist,
                               known_portset_sizes=known_portset_sizes,
                               bottleneck_ipc=used_bottleneck_ipc,)

    if not is_ok:
        raise RuntimeError("The preferred mapping is not valid for the encountered experiments!")

    logger.info("Preferred mapping successfully validated.")

    occuring_uop_widths = list(sorted(blocking_insns_per_num_ports.keys()))
    ports_for = dict()

    m = preferred_mapping
    for curr_width in occuring_uop_widths:
        for curr_insn in blocking_insns_per_num_ports[curr_width]:
            ports_for_curr_insn = frozenset(m.assignment[curr_insn])
            assert len(ports_for_curr_insn) == curr_width, f"The found mapping uses {len(ports_for_curr_insn)} ports for the instruction '{curr_insn}', which we determined to use {curr_width} ports."
            ports_for[curr_insn] = ports_for_curr_insn

    return ports_for, secondary_uops

def secondary_uops_interfere(*,
            secondary_uops_for_blocking_insn, # list of frozensets of ports
            subsequent_blocking_insns, # list of instructions
            ports_for, # dict mapping instructions to frozensets of ports
            blocked_ports, # frozenset of ports
            port_usage, # dict mapping frozensets of ports to numbers of occurrences
            num_uops, # the number of uops of the instruction under investigation
            num_uops_characterized, # the number of uops of the instruction under investigation that are already characterized
            min_cycles, # the number of cycles an experiment with a proper blocking instruction would take if the instruction under investigation has no uop on the blocked_ports
            num_of_blk_insns, # the number of blocking instructions
        ):
    # compute the set of ports affected by secondary uops
    ports_affected_by_secondary_uops = set(itertools.chain.from_iterable(secondary_uops_for_blocking_insn))

    # find the so far unchecked uops that go only to ports in the union of blocked_ports and the affected ports
    all_interesting_ports = ports_affected_by_secondary_uops.union(blocked_ports)
    relevant_uops = list()
    actual_uops = list()
    for subsequent_blocking_insn in subsequent_blocking_insns:
        subsequent_blocked_uop = set(ports_for[subsequent_blocking_insn])
        if subsequent_blocked_uop.issubset(all_interesting_ports):
            actual_uops.append(frozenset(subsequent_blocked_uop))
            subsequent_blocked_uop.difference_update(blocked_ports) # restrict it to the ports_affected_by_secondary_uops
            relevant_uops.append(frozenset(subsequent_blocked_uop))

    # gather the so far characterized uops and secondary uops that include relevant ports
    base_uop_map = defaultdict(int)
    for uop, n in itertools.chain(port_usage.items(), [(u, num_of_blk_insns) for u in secondary_uops_for_blocking_insn ]):
        if uop.issubset(all_interesting_ports):
            restricted_uop = uop.intersection(ports_affected_by_secondary_uops)
            if len(restricted_uop) > 0:
                base_uop_map[frozenset(restricted_uop)] += n

    def simulate(uop_map):
        ports = set(itertools.chain.from_iterable(uop_map.keys()))
        port_to_idx = {p: i for i, p in enumerate(sorted(ports))}
        weights = {}
        for uop, n in uop_map.items():
            bitset_uop = 0
            for p in uop:
                bitset_uop |= 1 << port_to_idx[p]
            weights[bitset_uop] = n

        max_set = (1 << len(ports)) - 1

        # the usual algorithm
        return pmproc.pure_core_algorithm(max_set, weights)

    # for each of them check if `num_uops - num_uops_characterized` of
    # it together with the secondary uops and the so far characterized
    # uops on the affected ports can be executed in min_cycles cycles
    # on the affected ports
    uop_factor = num_uops - num_uops_characterized
    for uop, actual_uop in zip(relevant_uops, actual_uops):
        exp = base_uop_map.copy()
        exp[uop] += uop_factor
        cycles = simulate(exp)
        if cycles > min_cycles:
            # we could relax this by adding the uops that we know to go
            # only on blocked_ports (divided by len(blocked_ports)) to
            # min_cycles
            logger.info(
                    "\n" +
                    "\n".join(textwrap.wrap(f"""
If all of the remaining {uop_factor} uops were {set(actual_uop)} uops, checking
for {set(blocked_ports)} uops would yield a wrong result. Combining these
{uop_factor} {set(actual_uop)} uops with the previously characterized uops
{port_usage} and {num_of_blk_insns} instances of the secondary uops
{secondary_uops_for_blocking_insn} and restricting them to the non-blocked
ports gives {cycles} cycles, wheras the blocking instructions would only reach
{min_cycles}.""", 100)) +
                    f"\nExperiment:\n    {dict(exp)}"
                )
            return True # they do interfere!

    return False # they don't interfere


def compute_full_mapping_improper_blocking_insns(params, proc, insns_with_measurements, ports_for, secondary_uops, restrict_insns=None, equiv_map=None, rejects=None):
    # Improper blocking instructions are instructions that use more than one
    # uop, but one of these uops has no blocking instruction. We therefore need
    # to use the improper blocking instruction to block for this uop.
    # ports_for contains proper and improper blocking instructions as if they
    # all were proper, secondary_uops contains for the improper ones the
    # additional uops that they use.

    # Check if we have redundant improper blocking instructions. That can make
    # sense to ensure that the port mapping is inferred correctly, but we need
    # only one of them. (Technically, having redundant ones with different
    # secondary uops could be helpful for cases where one is incompatible, but
    # we don't go there for now.)
    insns_for_portsets = defaultdict(list)
    for insn, portset in ports_for.items():
        # portset is a frozenset
        insns_for_portsets[portset].append(insn)


    skip_insns = set()
    for portset, insns in insns_for_portsets.items():
        if len(insns) > 1:
            # Redundant improper blocking instructions. We can skip all but
            # one of them.
            # We would want to take the one that imposes the least pressure on
            # the ports that it is not supposed to block. Since the occurring
            # port usages are rather simple in practice, we do not compute this
            # exactly but use a heuristic instead.
            insns.sort(key=lambda insn: sum([1/len(uop) for uop in secondary_uops.get(insn, [])]))
            skip_insns.update(insns[1:])

    for insn, uops in secondary_uops.items():
        if insn in skip_insns:
            continue
        primary_uop = ports_for[insn]
        for sec_uop in uops:
            if primary_uop.issubset(sec_uop):
                raise RuntimeError(f"Secondary uop {sec_uop} of instruction {insn} subsumes the primary uop {primary_uop}. We therefore cannot order it!")

    # construct a total order of blocking instructions
    # For every (im)proper blocking instruction I, all blocking instructions
    # for subsets of the primary port set of I and of the secondary port sets
    # of I (if any) need to be ordered before I.
    # We have a problem if this is relationship contains a loop. The SMT
    # formulation should ensure that this does not happen.
    requires = dict()
    for insn, primary_ports in ports_for.items():
        if insn in skip_insns:
            logger.info(f"Skipping improper blocking instruction '{insn}' because it is redundant.")
            continue
        # There probably is a more efficient way of doing this, but so far it's
        # not a problem.
        requires[insn] = set()
        for ports in (primary_ports, *secondary_uops.get(insn, [])):
            port_set = set(ports)
            for other_insn, other_ports in ports_for.items():
                if other_insn in skip_insns:
                    continue
                if other_insn == insn:
                    # This could be a problem for secondary uops that are a
                    # superset of the primary uop.
                    continue
                if port_set.issuperset(other_ports):
                    requires[insn].add(other_insn)

    for k, v in requires.items():
        logger.debug(f"  {k} requires {v}")

    order = []
    insns_to_do = set(requires.keys())
    while len(insns_to_do) > 0:
        take_out = []
        for i in insns_to_do:
            if len(requires[i]) == 0:
                # all prerequisits are met, we put it into the order
                take_out.append(i)
                order.append(i)

        if len(take_out) == 0:
            raise RuntimeError("The blocking instruction requirements form a loop!")

        # mark the newly ordered instructions as done in the requirements
        for i, depends in requires.items():
            depends.difference_update(take_out)
        insns_to_do.difference_update(take_out)

    logger.info("Order of blocking instructions (first to last):\n" + textwrap.indent("\n".join(order), "  "))

    out_arch = Architecture()
    for meas in insns_with_measurements:
        out_arch.add_insn(meas.insn)

    out_mapping = Mapping3(out_arch)

    def infer_portusage_for_measurement(meas, log):
        """ Return the inferred port usage dict if successful, None otherwise.
        """
        curr_insn = meas.insn
        cycles = meas.cycles
        num_uops = meas.num_uops

        log_record = dict()
        log.append(log_record)
        log_record['insn'] = curr_insn
        log_record['individual_cycles'] = cycles
        log_record['num_uops'] = num_uops

        logger.debug(f"infering the port usage of {curr_insn} ({cycles} cycles, {num_uops} uops)")

        num_uops_characterized = 0
        port_usage = dict()
        would_have_stopped_early = False

        for curr_order_idx, blocking_insn in enumerate(order):
            blocked_ports = ports_for[blocking_insn]
            curr_width = len(blocked_ports)

            # this follows the nanobench code
            num_of_blk_insns = max(2 * curr_width * max(1, int(cycles)), curr_width * num_uops, 10)
            num_of_blk_insns = min(100, num_of_blk_insns)

            min_cycles = num_of_blk_insns / curr_width # the blocking instructions alone should take this long


            if secondary_uops_interfere(
                    secondary_uops_for_blocking_insn = secondary_uops.get(blocking_insn, []),
                    subsequent_blocking_insns = order[curr_order_idx + 1:],
                    ports_for = ports_for,
                    blocked_ports = blocked_ports,
                    port_usage = port_usage,
                    num_uops = num_uops,
                    num_uops_characterized = num_uops_characterized,
                    min_cycles = min_cycles,
                    num_of_blk_insns = num_of_blk_insns,
                    ):
                msg = f"instruction {curr_insn} is incompatible with the improper blocking insn {blocking_insn}"
                if rejects is not None:
                    rejects.append(RejectedInsn(curr_insn, msg))
                logger.warning(msg)
                log_record['success'] = False
                log_record['aborted'] = True
                log_record['note'] = msg
                log_record['port_usage'] = port_usage
                log_record['offending_secondary_uop'] = sorted(ports)
                log_record['num_of_blk_insns'] = num_of_blk_insns
                # Alternatively, we could try an another improper blocking insn
                return None


            logger.debug(f"    - trying port set {blocked_ports} with {num_of_blk_insns} blocking instructions:")

            experiment = (num_of_blk_insns * [blocking_insn]) + [curr_insn]
            res = proc.execute(experiment)
            exp_cycles = res['cycles']
            exp_num_uops = res['num_uops']
            if not (round(exp_num_uops) == round(num_uops) + (num_of_blk_insns * (1 + len(secondary_uops.get(blocking_insn, []))))):
                msg = f"experiment {experiment} used more uops than expected"
                if rejects is not None:
                    rejects.append(RejectedInsn(curr_insn, msg))
                logger.warning(msg)
                log_record['success'] = False
                log_record['aborted'] = True
                log_record['note'] = msg
                log_record['port_usage'] = port_usage
                log_record['observed_num_uops'] = exp_num_uops
                log_record['expected_num_uops'] = num_uops + num_of_blk_insns
                log_record['num_of_blk_insns'] = num_of_blk_insns
                return None

            if not (exp_cycles + params.true_tp_epsilon >= min_cycles):
                msg = f"experiment {experiment} took fewer cycles than expected"
                if rejects is not None:
                    rejects.append(RejectedInsn(curr_insn, msg))
                logger.warning(msg)
                log_record['success'] = False
                log_record['aborted'] = True
                log_record['note'] = msg
                log_record['port_usage'] = port_usage
                log_record['observed_cycles'] = exp_cycles
                log_record['expected_minimum_cycles'] = min_cycles
                log_record['num_of_blk_insns'] = num_of_blk_insns
                return None

            surplus_uops = (exp_cycles - min_cycles) * curr_width
            rounded_surplus_uops = round(surplus_uops)
            # Previously, a lot of the errors came from measurements that
            # are slow by 0.05 cycles, which we would then multiply by the
            # curr_width, resulting in excessive errors. Therefore, we now
            # multiply the epsilon by the curr_width as well.
            if abs(rounded_surplus_uops - surplus_uops) > params.surplus_uops_epsilon * curr_width:
                msg = "Experiment with significantly non-integer number of surplus uops ({}):\n  {}".format(surplus_uops, experiment)
                if rejects is not None:
                    rejects.append(RejectedInsn(curr_insn, msg))
                logger.warning(msg)
                log_record['success'] = False
                log_record['aborted'] = True
                log_record['note'] = msg
                log_record['port_usage'] = port_usage
                log_record['surplus_uops'] = surplus_uops
                log_record['num_of_blk_insns'] = num_of_blk_insns
                return None
            surplus_uops = rounded_surplus_uops
            logger.debug(f"      the experiment with {curr_width} ports blocked took {exp_cycles} cycles, the instruction therefore contributes {exp_cycles - min_cycles} to the inverse throughput")
            logger.debug(f"      found {surplus_uops} uops that are executed on some ports in {blocked_ports}")
            for pset, num in port_usage.items():
                if pset.issubset(blocked_ports):
                    surplus_uops -= num
            if not (surplus_uops >= 0):
                msg = "surplus uops do not add up"
                if rejects is not None:
                    rejects.append(RejectedInsn(curr_insn, msg))
                logger.warning(msg)
                log_record['success'] = False
                log_record['aborted'] = True
                log_record['note'] = msg
                log_record['port_usage'] = port_usage
                log_record['surplus_uops'] = surplus_uops
                log_record['num_of_blk_insns'] = num_of_blk_insns
                return None
            logger.debug(f"      {surplus_uops} of these are not explained by previously inferred uops")
            port_usage[frozenset(blocked_ports)] = surplus_uops
            num_uops_characterized += surplus_uops

            if num_uops_characterized == num_uops:
                if not would_have_stopped_early:
                    logger.debug("  all observed uops are characterized, but we continue")
                    would_have_stopped_early = True
                # logger.debug("  all observed uops are characterized")
                # # all observed uops are explained, we can stop
                # log_record['success'] = True
                # log_record['aborted'] = False
                # log_record['note'] = "all characterized"
                # log_record['port_usage'] = port_usage
                # return port_usage

        if num_uops_characterized > num_uops:
            msg = "more uops were characterized for {} than were originally observed ({}):\n  {}".format(curr_insn, num_uops, num_uops_characterized)
            if rejects is not None:
                rejects.append(RejectedInsn(curr_insn, msg))
            logger.warning(msg)
            log_record['success'] = False
            log_record['aborted'] = False
            log_record['note'] = msg
            log_record['port_usage'] = port_usage
            log_record['uops_characterized'] = num_uops_characterized
            log_record['would_have_stopped_early'] = would_have_stopped_early
            # for all other cases, this last field would be trivially true or false
        elif num_uops_characterized < num_uops:
            msg = "fewer uops were characterized for {} than were originally observed ({}):\n  {}".format(curr_insn, num_uops, num_uops_characterized)
            if rejects is not None:
                rejects.append(RejectedInsn(curr_insn, msg))
            logger.warning(msg)
            log_record['success'] = False
            log_record['aborted'] = False
            log_record['note'] = msg
            log_record['port_usage'] = port_usage
            log_record['uops_characterized'] = num_uops_characterized
        else:
            assert num_uops_characterized == num_uops
            log_record['success'] = True
            log_record['aborted'] = False
            log_record['note'] = "all characterized"
            log_record['port_usage'] = port_usage
            log_record['uops_characterized'] = num_uops_characterized

        return port_usage


    log = []

    # perform the actual algorithm
    for idx, meas in enumerate(insns_with_measurements):
        curr_insn = meas.insn
        if restrict_insns is not None and curr_insn not in restrict_insns:
            continue

        # check if we found the instruction to be equivalent to a blocking instruction
        # (this is not necessary for correctness, but improves performance)
        fitting_blocking_insns = equiv_map.get(curr_insn, set()).intersection(set(ports_for.keys()))
        assert len(fitting_blocking_insns) <= 1, "more than one blocking instruction is equivalent to the instruction {}:\n  {}".format(curr_insn, fitting_blocking_insns)
        if len(fitting_blocking_insns) == 1:
            blocking_insn = next(iter(fitting_blocking_insns))
            blocked_ports = ports_for[blocking_insn]
            port_usage = {frozenset(blocked_ports): 1}
            log.append({
                    'insn': curr_insn,
                    'individual_cycles': meas.cycles,
                    'num_uops': meas.num_uops,
                    'success': True,
                    'aborted': False,
                    'note': "found equivalent to the blocking instruction '{}'".format(blocking_insn),
                    'port_usage': port_usage,
                })
        else:
            port_usage = infer_portusage_for_measurement(meas, log=log)

        if port_usage is not None:
            # enter the found port usage to the assignment
            for ps, num in port_usage.items():
                plist = list(sorted(ps))
                for x in range(num):
                    out_mapping.assignment[curr_insn].append(plist)

    return out_mapping, log



def check_singleton_throughputs_in_full_mapping(params, insns_with_measurements, mapping, rejects=None):
    """ Checks if the measured throughput of each singleton instruction is
    consistent with predictions of the inferred mapping. If this is slow,
    consider building lib/cppfastproc for a native port mapping simulation
    instead of pure python.
    """
    fails = []

    pm_proc = Processor(mapping)

    for iwm in insns_with_measurements:
        insn = iwm.insn
        if len(mapping.assignment[insn]) == 0:
            # logger.warning("instruction {} is not mapped to any ports".format(insn))
            continue
        cycles = iwm.cycles
        sim_res = pm_proc.execute([insn])
        # if abs(cycles - sim_res['cycles']) > params.true_tp_epsilon:
        if sim_res['cycles'] > cycles + params.true_tp_epsilon:
            # It's only really a problem if the experiment executes faster than what the port mapping allows
            # The other direction could be plausible with dependencies between uops.
            msg = "throughput of instruction {} is NOT consistent with inferred mapping: {} vs. {}".format(insn, cycles, sim_res['cycles'])
            if rejects is not None:
                rejects.append(RejectedInsn(insn, msg))
            logger.warning(msg)
            fails.append(insn)

    return fails

def find_mapping(params, proc, *, log_dir=None,
                 start_with_checkpoint=None,
                 num_checkpoints=None,
                 checkpoint_base={},
                 check_pairs=False,
                 preferred_blockinginsns=None,
                 discard_blockinginsns=None,
                 preferred_mapping=None,
                 added_improper_blockinginsns=None,
                 restrict_insns=None,
            ):
    """ The high-level online inference algorithm.
    """

    STAGE = cp.CheckPointer(
                        start_with=start_with_checkpoint,
                        run_steps=num_checkpoints,
                        checkpoint_base=checkpoint_base,
                        store_dir = log_dir,
                        preserve_all=True, # write everything in prev automatically to res
                    )

    for prev, res in STAGE('measure_singletons'):
        # measure the throughput and number of uops of all individual instructions
        rejects = []
        res.insns_with_measurements = measure_singletons(params, proc, rejects=rejects)
        res.rejects = rejects

    for prev, res in STAGE('characterize_blocking_insn_candidates'):
        # From the measurements of the previous stage, decide which
        # instructions are blocking instruction candidates, i.e., which only
        # use one uop, and determine on how many ports these uops can execute.
        res.candidates_per_num_ports = characterize_blocking_insn_candidates(params, proc, prev.insns_with_measurements, rejects=prev.rejects)

        occuring_uop_widths = list(sorted(res.candidates_per_num_ports.keys()))
        logger.info("uops with these widths occur in blocking instructions: {}".format(", ".join(map(str, occuring_uop_widths))))

        if log_dir is not None:
            with open(log_dir / 'blocking_insn_candidates.txt', 'w') as f:
                for width, insns in res.candidates_per_num_ports.items():
                    for i in insns:
                        print(i, file=f)

    # This is an expensive check
    # check_instruction_pairs(params, proc, insns_with_measurements, rejects=rejects)

    def log_blocking_insns(res):
        for width, insns in res.blocking_insns_per_num_ports.items():
            lines = []
            for i in insns:
                equiv_class_size = 1 + len(res.equivalent_candidates_per_blocking_insn[i])
                lines.append(f"{i} (representing a class of {equiv_class_size} equivalent candidate(s))")
            logger.info(f"blocking instructions of width {width}:\n" + textwrap.indent("\n".join(lines), "  "))


    for prev, res in STAGE('filter_equal_blocking_insns'):
        # Measure for each pair of blocking insn candidates if their
        # throughputs are additive, i.e., executing them together requires the
        # sum of the cycles required to execute them individually.
        # If that's the case, they are equivalent. We only need to consider one
        # representative blocking instruction per equivalence class.
        res.blocking_insns_per_num_ports, res.equivalent_candidates_per_blocking_insn = filter_equal_blocking_insns(params, proc, prev.candidates_per_num_ports, rejects=prev.rejects)

        equiv_map = dict()
        for k, vs in res.equivalent_candidates_per_blocking_insn.items():
            equiv_class = {k}
            equiv_class.update(vs)
            for i in equiv_class:
                equiv_map[i] = equiv_class
        res.equiv_map = equiv_map

        log_blocking_insns(res)

    for prev, res in STAGE('replace_with_preferred_blocking_insns'):
        # Replace the representative blocking instructions identified in the
        # previous stage with ones from a user-specified list of preferred
        # blocking instructions. This is not a semantically necessary step, but
        # helps to get consistent results across different runs.

        # ALSO: If manually specified, add additional improper blocking
        # instructions. These are instructions with more than one uop where one
        # of them does not have a (proper) blocking instruction. Ideally, we
        # assume that this does not occur, but in reality it does (for store
        # instructions). Such improper blocking instructions are assumed to
        # have exactly one (primary) uop that has no proper blocking
        # instruction. This allows us to add constraints to the SMT solver in
        # the next step that speed up inference considerably.

        # this relies on the preserve_all option of the CheckPointer

        if preferred_blockinginsns is not None:
            logger.info("trying to replace found blocking instructions with preferred ones")
            replace_map = dict() # maps chosen blocking insns to preferred replacements
            for p in preferred_blockinginsns:
                if p in res.equivalent_candidates_per_blocking_insn.keys():
                    logger.info(f"  - preferred blocking insn '{p}' was already chosen by the algorithm")
                    continue

                replacee = None
                for chosen, vs in res.equivalent_candidates_per_blocking_insn.items():
                    if p in vs:
                        assert replacee is None, f"one preferred blocking insn ('{p}') could replace two chosen instructions ('{chosen}' and '{replacee}'), this shouldn't be possible!"

                        if chosen in replace_map.keys():
                            logger.warning(f"  - preferred blocking insn '{p}' conflicts with other preferred insn '{replace_map[chosen]}' for chosen insn '{chosen}'")
                            continue
                        logger.info(f"  - preferred blocking insn '{p}' replaces '{chosen}'")
                        replace_map[chosen] = p
                        replacee = chosen

                if replacee is None:
                    logger.warning(f"  - preferred blocking insn '{p}' is no blocking insn candidate!")

            for old, new in replace_map.items():
                # update res.blocking_insns_per_num_ports
                for k, vs in res.blocking_insns_per_num_ports.items():
                    if old in vs:
                        vs.remove(old)
                        vs.append(new)
                        break

                # update res.equivalent_candidates_per_blocking_insn
                entry = res.equivalent_candidates_per_blocking_insn[old]
                del res.equivalent_candidates_per_blocking_insn[old]
                entry.remove(new)
                entry.append(old)
                res.equivalent_candidates_per_blocking_insn[new] = entry

                # the res.equiv_map does not need to be changed

            logger.info("blocking instructions after replacement with preferred ones:")
            log_blocking_insns(res)
        else:
            logger.info("no preferred blocking instructions specified, continuing")

        improper_blockinginsns_with_num_uops = {}
        if added_improper_blockinginsns is not None:
            logger.info("adding manually specified improper blocking instructions")
            proper_blocking_insns = set(res.equiv_map.keys())
            for i in added_improper_blockinginsns:
                if i in proper_blocking_insns:
                    logger.warning(f"  - manually specified improper blocking insn '{i}' was found to be a proper blocking insn by the algorithm, ignoring it")
                else:
                    improper_blockinginsns_with_num_uops[i] = -1

            actually_added = set(improper_blockinginsns_with_num_uops.keys())
            for iwm in prev.insns_with_measurements:
                i = iwm.insn
                if i not in actually_added:
                    continue
                num_uops = iwm.num_uops
                improper_blockinginsns_with_num_uops[i] = num_uops
                logger.info(f"  - added improper blocking insn '{i}' with {num_uops} uops")

        else:
            logger.info("no improper blocking instructions specified, continuing")
        res.improper_blockinginsns_with_num_uops = improper_blockinginsns_with_num_uops

    for prev, res in STAGE('measure_external_bottleneck'):
        # Try various ways of combining blocking instructions into experiments
        # to find the maximal number of instructions that we can possibly put
        # through the processor in a cycle. This bottleneck is typically not
        # imposed by the port mapping but by an external (from the port mapping
        # model) source, like the bandwidth of the frontend or retirement.

        # It could make sense to also consider the added improper blocking
        # insns here, but it shouldn't be necessary for our considered
        # architectures, so we skip them for now.
        res.bottleneck_ipc = measure_external_bottleneck(params, proc, prev.blocking_insns_per_num_ports, rejects=prev.rejects)

    for prev, res in STAGE('compute_mapping_for_blocking_insns'):
        # Use the counter-example-guided SMT algorithm to find a port mapping
        # for the blocking instructions. Because of the improper blocking
        # instructions, the result is no longer a two-level port mapping, but a
        # three-level one (of constrained shape).
        # Optionally: discard some blocking instructions before.
        if discard_blockinginsns is not None:
            # this relies on the preserve_all option of the CheckPointer since
            # the mapping has already been copied
            for k, vs in res.blocking_insns_per_num_ports.items():
                new_entry = []
                for i in vs:
                    if i in discard_blockinginsns:
                        logger.info(f"discarding blocking instruction '{i}'")
                    else:
                        new_entry.append(i)
                res.blocking_insns_per_num_ports[k] = new_entry

        res.ports_for, res.secondary_uops, res.synth_config, res.elist = compute_mapping_for_blocking_insns(
                    params, proc,
                    res.blocking_insns_per_num_ports,
                    prev.bottleneck_ipc,
                    improper_blockinginsns_with_num_uops=prev.improper_blockinginsns_with_num_uops,
                    rejects=prev.rejects,
                    dump_experiments=log_dir)

        # add the improper blocking insns to the data structures
        new_blocking_insns_per_num_ports = deepcopy(res.blocking_insns_per_num_ports)
        for insn in prev.improper_blockinginsns_with_num_uops.keys():
            num_ports = len(res.ports_for[insn])
            if num_ports not in new_blocking_insns_per_num_ports:
                new_blocking_insns_per_num_ports[num_ports] = []
            new_blocking_insns_per_num_ports[num_ports].append(insn)

        res.blocking_insns_per_num_ports = new_blocking_insns_per_num_ports

        for width, insns in res.blocking_insns_per_num_ports.items():
            mapping_str = ""
            for i in insns:
                mapping_str += str(i) + " - ports: " + (",".join(map(str, res.ports_for[i]))) 

                secondary_uops = res.secondary_uops.get(i, [])
                if len(secondary_uops) > 0:
                    mapping_str += " (with secondary uops: " + (", ".join( "[" + (",".join(map(str, sec_uop))) + "]" for sec_uop in secondary_uops ) ) + ")"

                mapping_str += "\n"

            logger.info(f"blocking instructions of width {width}:\n" + textwrap.indent(mapping_str, "  "))

    for prev, res in STAGE('validate_preferred_mapping'):
        # If a preferred mapping for the blocking instructions is specified
        # manually, check if it also satisifies the experiments encountered
        # during the inference in the previous stage. If it does, it could have
        # been chosen by the algorithm as well, so we can use it instead (which
        # we do). This is not semantically necessary, but again makes for more
        # consistent results.
        # This does not imply that the inferred and the preferred port mappings
        # are isomorphic!
        # Secondary uops in the preferred mapping (if required) can be
        # specified as a dict mapping instructions to lists of uops (which, in
        # turn, are lists of ports) under the "secondary_uops" key in the
        # "metadata" field of the port mapping json. They are added as a
        # secondary_uops field to the preferred_mapping (rather hacky).

        if preferred_mapping is not None:
            logger.info("validating preferred mapping")
            # this raises an exception if the preferred mapping is not valid
            res.ports_for, res.secondary_uops = validate_preferred_mapping(params, proc,
                                                       synth_config=prev.synth_config,
                                                       elist=prev.elist,
                                                       blocking_insns_per_num_ports=prev.blocking_insns_per_num_ports,
                                                       bottleneck_ipc=prev.bottleneck_ipc,
                                                       preferred_mapping=preferred_mapping,
                                                       rejects=prev.rejects)

            # dump the resulting mapping again, for good measure
            for width, insns in res.blocking_insns_per_num_ports.items():
                mapping_str = ""
                for i in insns:
                    mapping_str += str(i) + " - ports: " + (",".join(map(str, res.ports_for[i])))

                    secondary_uops = res.secondary_uops.get(i, [])
                    if len(secondary_uops) > 0:
                        mapping_str += " (with secondary uops: " + (", ".join( "[" + (",".join(map(str, sec_uop))) + "]" for sec_uop in secondary_uops ) ) + ")"

                    mapping_str += "\n"

                logger.info(f"blocking instructions of width {width}:\n" + textwrap.indent(mapping_str, "  "))

        else:
            logger.info("no preferred mapping specified, continuing")

    for prev, res in STAGE('compute_full_mapping'):
        # Follow the uops.info algorithm to measure how all instructions
        # conflict with the blocking instructions and form a corresponding port
        # mapping.
        res.mapping, res.compute_full_mapping_log = compute_full_mapping_improper_blocking_insns(params, proc, prev.insns_with_measurements, prev.ports_for, prev.secondary_uops, restrict_insns=restrict_insns, equiv_map=prev.equiv_map, rejects=prev.rejects)

        # res.mapping, res.compute_full_mapping_log = compute_full_mapping(params, proc, prev.insns_with_measurements, prev.blocking_insns_per_num_ports, prev.ports_for, equiv_map=prev.equiv_map, rejects=prev.rejects)

    for prev, res in STAGE('check_singleton_throughputs_in_full_mapping'):
        # Check if the resulting port mapping actually explains the throughputs
        # that we have measured. A port usage could still be correct if that's
        # not the case, but it's a good thing to be concious about.
        res.fails = check_singleton_throughputs_in_full_mapping(params, prev.insns_with_measurements, prev.mapping, rejects=prev.rejects)

    return res


