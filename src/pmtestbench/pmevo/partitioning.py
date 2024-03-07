""" Code for partitioning instructions into equivalence classes based on
throughput measurements.
"""

from collections import defaultdict
import itertools

from ..common.architecture import Architecture
from ..common.experiments import ExperimentList
from ..common.portmapping import Mapping3

import logging
logger = logging.getLogger(__name__)

def create_partition(elems, equiv_map):
    """ Partition a collection of elements into buckets of equivalent elements.
    Two elements e1, e2 are considered equivalent if and only if
    equiv_map[(e1, e2)] == True.
    Returns the list of buckets and a mapping of elements to buckets.
    """
    elem_to_bucket = { i: {i} for i in elems }
    for (i1, i2), equiv in equiv_map.items():
        if equiv:
            bucket_i1 = elem_to_bucket[i1]
            bucket_i2 = elem_to_bucket[i2]
            new_bucket = bucket_i1.union(bucket_i2)
            for i in new_bucket:
                elem_to_bucket[i] = new_bucket
    buckets = []
    covered_elems = []
    for i, b in elem_to_bucket.items():
        if i in covered_elems:
            continue
        covered_elems += b
        buckets.append(list(b))
    return buckets, elem_to_bucket

def equals(a, b, *, epsilon):
    a = float(a)
    b = float(b)
    return 2 * abs(a - b) <= epsilon  * (a + b)

def partition_instructions(elist, singleton_elist, epsilon):

    arch = elist.arch
    insns = arch.insn_list

    singleton_results = dict()
    for e in singleton_elist:
        assert len(e.iseq) == 1
        i = e.iseq[0]
        t = e.get_cycles()
        singleton_results[i] = t

    # instructions can only be equivalent if they have equal singleton measurements
    singleton_equiv_map = { (i, j): equals(singleton_results[i], singleton_results[j], epsilon=epsilon) for (i, j) in itertools.combinations(insns, 2) }
    insn_buckets, insn_to_bucket = create_partition(insns, singleton_equiv_map)

    complex_exps = defaultdict(lambda: defaultdict(list))
    for e in elist:
        insn_set = set(e.iseq)
        assert len(insn_set) == 2
        i, j = insn_set
        complex_exps[i][j].append(e)
        complex_exps[j][i].append(e)

    def check_equivalent_complex(i1, i2):
        # two instructions i1 and i2 are equivalent wrt. pair experiments if
        # each experiment with i1 and some other instruction j has an equal
        # measurement to the corresponding experiment with i2 and j.
        i1_exps = complex_exps[i1]
        i2_exps = complex_exps[i2]
        for i in insns:
            if i == i1 or i == i2:
                continue
            i1i_exps = sorted(i1_exps[i], key = lambda x: len(x.iseq))
            i2i_exps = sorted(i2_exps[i], key = lambda x: len(x.iseq))
            for e1, e2 in zip(i1i_exps, i2i_exps):
                if not (len(e1.iseq) == len(e2.iseq)):
                    logger.warning("Corresponding experiments with differing length:\n  {}\n  {}\n".format(repr(e1), repr(e2)))
                    return False
                assert len(e1.iseq) == len(e2.iseq)
                if not equals(e1.get_cycles(), e2.get_cycles(), epsilon=epsilon):
                    logger.debug("Distinguishing experiments for {} and {}:\n  {}\n  {}\n".format(i1, i2, repr(e1), repr(e2)))
                    return False
        return True

    equality_map = dict()
    for bucket in insn_buckets:
        for i1, i2 in itertools.combinations(bucket, 2):
            equality_map[(i1, i2)] = check_equivalent_complex(i1, i2)

    final_buckets, insn_to_final_bucket = create_partition(insns, equality_map)

    return final_buckets, insn_to_final_bucket

def compute_representatives(elist, singleton_elist, epsilon):
    """ Partition the set of instructions in the Architecture into ones that
    are not distinguishable by experiments in singleton_elist and elist
    allowing deviations in the measurements of epsilon.

    Returns a list of representative instructions (one for each equivalence
    class) and a dictionary mapping each original instruction to its
    representative. The former can be used in the restrict_elist() function,
    the latter in generalize_mapping().
    """

    representatives = []
    insn_to_representative = dict()

    buckets, insn_to_bucket = partition_instructions(elist, singleton_elist, epsilon)

    for b in buckets:
        sorted_bucket = sorted(b)
        representative = sorted_bucket[0]
        representatives.append(representative)
        for i in b:
            insn_to_representative[i] = representative

    return representatives, insn_to_representative


def restrict_elist(elist, insn_representatives):
    """ Restrict an ExperimentList to experiments that only contain
    instructions from the insn_representative list.

    Returns the new, restricted ExperimentList.
    """

    arch = elist.arch
    new_arch = Architecture()
    whitelist = insn_representatives
    new_arch.add_insns(( i for i in arch.insn_list if i in whitelist ))

    new_elist = ExperimentList(new_arch)
    for e in elist:
        new_iseq = []
        for i in e.iseq:
            if i not in whitelist:
                new_iseq = None
                break
            new_iseq.append(i)
        if new_iseq is None:
            continue
        new_exp = new_elist.create_exp(new_iseq)
        new_exp.result = e.result
        new_exp.other_results = e.other_results

    return new_elist


def generalize_mapping(new_arch, mapping, insn_to_representative):
    """ Generalize the given Mapping to the given new Architecture (whose
    instructions need to be a superset of the instructions treated in the
    Mapping).
    insn_to_representative needs to map each instruction from new_arch to an
    instruction present in the Mapping.

    Returns the new Mapping.
    """
    if isinstance(mapping, Mapping3):
        new_mapping = Mapping3(new_arch)
        insns = new_arch.insn_list
        for i in insns:
            representative = insn_to_representative[i]
            new_mapping.assignment[i] = mapping.assignment[representative][:]
        return new_mapping
    else:
        assert False, f"Unknown mapping type for generalization: {type(mapping)}"
