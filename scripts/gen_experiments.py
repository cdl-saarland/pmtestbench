#! /usr/bin/env python3

""" This multi-stage script generates experiments and performs measurements.

The separation in stages allows e.g. to make and evaluate experiments on
different machines.
"""

import argparse
from datetime import datetime
from pathlib import Path
import itertools
import math
import os
import random
import re
import sys
import textwrap

import_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.append(import_path)

from pmtestbench.common.processors import Processor
from pmtestbench.common.experiments import ExperimentList
from pmtestbench.pmevo.partitioning import create_partition, equals
from pmtestbench.common.utils import increase_filename

from iwho.configurable import load_json_config, store_json_config
from iwho.utils import parse_args_with_logging

def load_processor(proc_file):
    processor_config = load_json_config(proc_file)
    proc = Processor(config=processor_config)
    arch = proc.get_arch()
    return proc, arch

class tcolors:
    BLUE = '\033[94m'
    END = '\033[0m'



VERBOSE = True
PROG = sys.argv[0]

def instruction(msg):
    if VERBOSE:
        important_instruction(msg)

def important_instruction(msg):
    print(textwrap.dedent(msg).strip())
    print("")

def intro(args):
    instruction(f"""
    This script handles the process of generating experiments for PMEvo. The
    process is separated into a number of stages:
      - make singleton experiments
      - evaluate singleton experiments
      - make pair experiments
      - evaluate pair experiments

    Each stage is performed by an execution of this script with the appropriate
    subcommands. Subsequent stages use the results of the previous ones. The
    process is designed so that the generation (make-*) and the evaluation
    (eval-*) of experiments can be done on separate machines (copying the
    intermediate results from one to another). If a model for a slow machine is
    to be inferred, the make part can be sped up by executing it on a faster
    machine.
    """)

    important_instruction(f"""
    {tcolors.BLUE}Start the process of generating PMEvo-style INFERENCE experiments by running the following command:{tcolors.END}
      {PROG} make-singletons <path/to/proc_config.json>
    """)

    instruction(f"""
    If you do not intend to execute these stages separately, you can also run
    the all stages in one go.
    """)

    important_instruction(f"""
    {tcolors.BLUE}Perform the entire process of generating PMEvo-style INFERENCE experiments in by running the following command:{tcolors.END}
      {PROG} full-pmevo <path/to/proc_config.json> --epsilon <X>
    """)

    instruction(f"""
    Similar stages exist to generate random validation experiments:
      - make validation experiments
      - evaluate validation experiments
    These are not necessary to infer a port mapping, but can be used to
    evaluate an inferred port mapping.
    """)

    important_instruction(f"""
    {tcolors.BLUE}Start the process of generating VALIDATION experiments by running the following command:{tcolors.END}
      {PROG} make-validation <path/to/proc_config.json>
    """)

def make_singletons(args):
    proc, arch = load_processor(args.processor)

    insns = arch.insn_list

    outfile_name = args.output
    if outfile_name is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        outfile_name = f"./pmevo_exps_{timestamp}_01.json"

    important_instruction(f"Generating singleton experiments for {len(insns)} instructions.")

    elist = ExperimentList(arch)
    for i in insns:
        elist.create_exp([i])


    with open(outfile_name, 'w') as f:
        elist.to_json(f)

    instruction(f"""
    Done. The experiments have been written to '{outfile_name}'.
    In the next step, these experiments need to be evaluated.
    """)

    important_instruction(f"""
    {tcolors.BLUE}Continue on the benchmarking machine by running the following command:{tcolors.END}
      {PROG} eval-singletons {args.processor} {outfile_name}
    """)
    return outfile_name


def eval_singletons(args):
    proc, arch = load_processor(args.processor)

    with open(args.input, 'r') as f:
        elist = ExperimentList.from_json(f, arch)

    outfile_name = args.output
    if outfile_name is None:
        outfile_name = increase_filename(args.input)

    important_instruction(f"Evaluating {len(elist)} singleton experiments.")

    proc.eval_batch(elist)

    with open(outfile_name, 'w') as f:
        elist.to_json(f)

    instruction(f"""
    Done. The evaluated experiments have been written to '{outfile_name}'.
    In the next step, these measurements are used to generate suitable pair
    experiments.
    """)
    # TODO document epsilon

    important_instruction(f"""
    {tcolors.BLUE}Continue by running the following command:{tcolors.END}
      {PROG} make-pairs {args.processor} {outfile_name} --epsilon <X>
    """)
    return outfile_name

def make_pairs(args):
    proc, arch = load_processor(args.processor)

    insns = arch.insn_list

    with open(args.input, 'r') as f:
        elist = ExperimentList.from_json(f, arch)

    outfile_name = args.output
    if outfile_name is None:
        outfile_name = increase_filename(args.input)

    important_instruction(f"Generating pair experiments for {len(insns)} instructions.")

    epsilon = args.epsilon

    singleton_results = dict()
    for e in elist:
        if len(e.iseq) != 1:
            continue
        i = e.iseq[0]
        t = e.get_cycles()
        singleton_results[i] = t

    singleton_equiv_map = { (i, j): equals(singleton_results[i], singleton_results[j], epsilon=epsilon) for (i, j) in itertools.combinations(insns, 2) }

    # Partition the instructions according to equivalent singleton results and
    # use the maximal occuring singleton result of the class of equivalent
    # instructions. This is needed so that equivalent instructions get
    # corresponding experiments with identical sizes so that these experiments
    # can be used for preprocessing later on.
    insn_buckets, insn_to_bucket = create_partition(insns, singleton_equiv_map)
    insn_to_max_t = { i: max(( singleton_results[j] for j in b )) for i, b in insn_to_bucket.items() }

    for i, j in itertools.combinations(insns, 2):
        elist.create_exp([i, j])
        ti = insn_to_max_t[i]
        tj = insn_to_max_t[j]
        if ti < tj:
            i, j = j, i
            ti, tj = tj, ti
        factor = math.ceil(ti / tj)
        if factor == 1:
            continue
        iseq = [i]
        iseq += [j for x in range(factor)]
        elist.create_exp(iseq)

    with open(outfile_name, 'w') as f:
        elist.to_json(f)

    instruction(f"""
    Done. All experiments have been written to '{outfile_name}'.
    In the next step, the new experiments need to be evaluated.
    """)

    important_instruction(f"""
    {tcolors.BLUE}Continue on the benchmarking machine by running the following command:{tcolors.END}
      {PROG} eval-pairs {args.processor} {outfile_name}
    """)
    return outfile_name

def eval_general(args, recompute=False):
    proc, arch = load_processor(args.processor)

    if isinstance(args.input, list) or  isinstance(args.input, tuple):
        infiles = args.input
        do_return = False
    else:
        infiles = [args.input]
        do_return = True

    for infile in infiles:
        with open(infile, 'r') as f:
            elist = ExperimentList.from_json(f, arch)

        outfile_name = args.output
        if outfile_name is None:
            outfile_name = increase_filename(infile)

        insns = arch.insn_list
        important_instruction(f"Evaluating {len(elist)} experiments.")

        bak_name = outfile_name + ".partial"

        def progress_callback(num_processed, num_total, new_iseqs, new_results):
            print(f"Processed {num_processed}/{num_total} experiments.", end='\r')
            with open(bak_name, 'w') as f:
                elist.to_json(f)

        proc.eval_batch(elist, recompute=recompute, progress_callback=progress_callback)

        with open(outfile_name, 'w') as f:
            elist.to_json(f)

    if do_return:
        return outfile_name
    else:
        return None


def eval_pairs(args):
    outfile_name = eval_general(args, recompute=False)

    instruction(f"""
    Done. The evaluated experiments have been written to '{outfile_name}'.
    The resulting experiments can now be used for port mapping inference
    through PMEvo.
    """)
    return outfile_name


def all_singletons_pairs(args):
    proc, arch = load_processor(args.processor)

    class DummyArgs:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def __getattr__(self, name):
            # called if name is not an existing attribute
            return None

    if args.output is None:
        intermediate_name = None
    else:
        intermediate_dir = Path(args.output).parent
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        intermediate_name = intermediate_dir / f"pmevo_exps_{timestamp}_01.json"

    fname = make_singletons(DummyArgs(processor=args.processor, output=intermediate_name))
    fname = eval_singletons(DummyArgs(processor=args.processor, input=fname))

    fname = make_pairs(DummyArgs(processor=args.processor, input=fname, epsilon=args.epsilon))
    fname = eval_pairs(DummyArgs(processor=args.processor, input=fname, output=args.output))
    return fname



def make_validation(args):
    num_experiments = args.num_experiments
    length = args.length

    proc, arch = load_processor(args.processor)
    insns = arch.insn_list

    outfile_name = args.output
    if outfile_name is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        outfile_name = f"./eval_exps_{timestamp}_01.json"

    important_instruction(f"Generating {num_experiments} experiments with {length} instructions each. {len(insns)} instructions are available for sampling.")

    elist = ExperimentList(arch)

    for i in range(num_experiments):
        iseq = random.choices(insns, k=length)
        elist.create_exp(iseq)

    with open(outfile_name, 'w') as f:
        elist.to_json(f)

    instruction(f"""
    Done. The experiments have been written to '{outfile_name}'.
    In the next step, these experiments need to be evaluated.
    """)

    important_instruction(f"""
    {tcolors.BLUE}Continue on the benchmarking machine by running the following command:{tcolors.END}
      {PROG} eval-validation {args.processor} {outfile_name}
    """)
    return outfile_name


def eval_validation(args):
    outfile_name = eval_general(args, recompute=False)

    instruction(f"""
    Done. The evaluated experiments have been written to '{outfile_name}'.
    """)
    return outfile_name

def make_lenseries(args):
    num_experiments = args.num_experiments
    start_length = args.start
    end_length = args.end
    step = args.step

    proc, arch = load_processor(args.processor)
    insns = arch.insn_list

    outdir = Path(args.outdir)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_name = f"./lenseries_exps_{timestamp}_len{{:03}}_01.json"

    outfile_names = []
    for length in range(start_length, end_length+1, step):
        outfile_name = outdir / base_name.format(length)

        important_instruction(f"Generating {num_experiments} experiments with {length} instructions each. {len(insns)} instructions are available for sampling.")

        elist = ExperimentList(arch)

        for i in range(num_experiments):
            iseq = random.choices(insns, k=length)
            elist.create_exp(iseq)

        with open(outfile_name, 'w') as f:
            elist.to_json(f)

        instruction(f"""
        Done. The experiments have been written to '{outfile_name}'.
        """)
        outfile_names.append(outfile_name)

    outfile_names_str = " ".join(map(str, outfile_names))
    important_instruction(f"""
    {tcolors.BLUE}Continue on the benchmarking machine by running the following command:{tcolors.END}
      {PROG} eval {args.processor} {outfile_names_str}
    """)
    return outfile_names


def add_processor_argument(argparser):
    argparser.add_argument('processor', metavar='PROC',
        help='Path to a processor config in json format. JSONified port mappings can also be used directly as processor configs.')

def add_output_argument(argparser):
    argparser.add_argument('-o', '--output', metavar='OUTFILE', default=None,
        help='Path to a destination json file, where the resulting experiments should be written to.')

def main():
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.set_defaults(func=intro)

    argparser.add_argument('--verbose', dest='verbose', action='store_true',
        help='print extensive usage instructions all the way (default)')
    argparser.add_argument('--no-verbose', dest='verbose', action='store_false',
        help='do not print extensive usage instructions all the way')
    argparser.set_defaults(verbose=True)

    subparsers = argparser.add_subparsers(title='Available stages', metavar='STAGE')


    # make-singletons

    parser_make_singletons = subparsers.add_parser('make-singletons',
        help='generate all experiments with only one instruction per experiment')

    add_processor_argument(parser_make_singletons)
    add_output_argument(parser_make_singletons)

    parser_make_singletons.set_defaults(func=make_singletons)


    # eval-singletons

    parser_eval_singletons = subparsers.add_parser('eval-singletons',
        help='evaluate the experiments generated by the singletons stage')

    add_processor_argument(parser_eval_singletons)
    add_output_argument(parser_eval_singletons)
    parser_eval_singletons.add_argument('input', metavar="INFILE",
        help='input file with unevaluated singleton experiments')
    parser_eval_singletons.set_defaults(func=eval_singletons)


    # make-pairs

    parser_make_pairs = subparsers.add_parser('make-pairs',
        help='generate pair experiments based on singleton measurements')

    add_processor_argument(parser_make_pairs)
    add_output_argument(parser_make_pairs)
    parser_make_pairs.add_argument('input', metavar="INFILE",
        help='input file with evaluated singleton experiments')
    parser_make_pairs.add_argument('-e', '--epsilon', required=True, type=float,
        help='if the difference between to measurements is less than this, they are considered equal')
    parser_make_pairs.set_defaults(func=make_pairs)


    # eval-pairs

    parser_eval_pairs = subparsers.add_parser('eval-pairs',
        help='evaluate previously generated pair experiments')

    add_processor_argument(parser_eval_pairs)
    add_output_argument(parser_eval_pairs)
    parser_eval_pairs.add_argument('input', metavar="INFILE",
        help='input file with unevaluated pair experiments')
    parser_eval_pairs.set_defaults(func=eval_pairs)


    # full-pmevo

    parser_full_pmevo = subparsers.add_parser('full-pmevo',
        help='perform all stages for pmevo-style experiments in one go')

    add_processor_argument(parser_full_pmevo)
    add_output_argument(parser_full_pmevo)
    parser_full_pmevo.add_argument('-e', '--epsilon', required=True, type=float,
        help='if the difference between to measurements is less than this, they are considered equal')
    parser_full_pmevo.set_defaults(func=all_singletons_pairs)


    # make-validation

    parser_make_validation = subparsers.add_parser('make-validation',
        help='generate validation experiments')

    add_processor_argument(parser_make_validation)
    add_output_argument(parser_make_validation)
    parser_make_validation.add_argument('-n', '--num-experiments', metavar="N", required=True,
                                        help='the number of experiments to generate', type=int)
    parser_make_validation.add_argument('-l', '--length', metavar="L", required=True,
                                        help='the length of each experiment', type=int)
    parser_make_validation.set_defaults(func=make_validation)


    # eval-validation

    parser_eval_validation = subparsers.add_parser('eval-validation',
        help='evaluate validation experiments')

    add_processor_argument(parser_eval_validation)
    add_output_argument(parser_eval_validation)
    parser_eval_validation.add_argument('input', metavar="INFILE",
        help='input file with unevaluated validation experiments')
    parser_eval_validation.set_defaults(func=eval_validation)


    # make-lenseries

    parser_make_lenseries = subparsers.add_parser('make-lenseries',
        help='generate a sequence of experiment lists with increasing length')

    add_processor_argument(parser_make_lenseries)
    parser_make_lenseries.add_argument('-o', '--outdir', metavar='OUTDIR', default='.',
        help='Path of the desired output directory, where the resulting experiment lists should be written to.')
    parser_make_lenseries.add_argument('-n', '--num-experiments', metavar="N", required=True,
                                        help='the number of experiments to generate per series entry', type=int)
    parser_make_lenseries.add_argument('-s', '--start', metavar="L", required=True,
                                        help='the start length of the experiments (inclusive)', type=int)
    parser_make_lenseries.add_argument('-e', '--end', metavar="L", required=True,
                                        help='the end length of the experiments (inclusive)', type=int)
    parser_make_lenseries.add_argument('-t', '--step', metavar="L", default=1,
                                        help='the increment between each step (default: 1)', type=int)
    parser_make_lenseries.set_defaults(func=make_lenseries)

    # eval

    parser_eval = subparsers.add_parser('eval',
        help='evaluate any list of experiments')

    add_processor_argument(parser_eval)
    add_output_argument(parser_eval) # using this does not make much sense
    parser_eval.add_argument('input', nargs='+', metavar="INFILE",
        help='input files with unevaluated experiments')
    parser_eval.add_argument('-r', '--recompute', action='store_true',
        help='evaluate all experiments, even if they have already been evaluated')
    parser_eval.set_defaults(func=eval_general)

    args = parse_args_with_logging(argparser, "info")

    global VERBOSE
    VERBOSE = args.verbose

    args.func(args)


if __name__ == "__main__":
    main()

