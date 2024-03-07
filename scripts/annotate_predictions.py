#!/usr/bin/env python3

""" Annotate throughput predictions of a predictor to a list of experiments.

The experiments should be produced and evaluated with reference benchmarks via
    `./scripts/gen_experiments.py make-validation`
and
    `./scripts/gen_experiments.py eval-validation`
The result is a new list of experiments with the predicted throughput annotated
under the specified key, additionally to all previous evaluations. This step
can be performed multiple times with different predictors.
"""

import argparse

import os
import sys

import_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.append(import_path)

from pmtestbench.common.processors import Processor
from pmtestbench.common.experiments import ExperimentList
from pmtestbench.common.utils import increase_filename

from iwho.configurable import load_json_config
from iwho.utils import parse_args_with_logging

def main():
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('processor', metavar='PROC',
        help='Path to a processor config in json format. JSONified port mappings can also be used directly as processor configs.')

    argparser.add_argument('-o', '--output', metavar="OUTFILE", default=None,
        help='the destination path for the annotated output experiment list')

    argparser.add_argument('-i', '--result-id', metavar="ID", required=True,
        help='the key under which the throughput prediction is stored in the experiment list')

    argparser.add_argument('input', metavar="INFILE", nargs='+',
        help='the input experiment list(s), in json format')

    args = parse_args_with_logging(argparser, "info")

    processor_config = load_json_config(args.processor)
    proc = Processor(config=processor_config)
    arch = proc.get_arch()

    for infile in args.input:
        with open(infile, 'r') as f:
            # elist = ExperimentList.from_json(f, arch)
            elist = ExperimentList.from_json(f)

        result_id = args.result_id
        print("Annotating {} experiments with throughput predictions for {}".format(len(elist), result_id))

        for exp in elist:
            res = proc.execute(exp.iseq)
            exp.add_other_result(result_id, res)

        outfile_name = args.output
        if outfile_name is None:
            outfile_name = increase_filename(infile)

        with open(outfile_name, 'w') as f:
            elist.to_json(f)

        print("Done. The annotated experiment list has been written to {}".format(outfile_name))

    return 0

if __name__ == "__main__":
    sys.exit(main())
