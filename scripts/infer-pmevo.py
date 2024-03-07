#! /usr/bin/env python3

""" Run PMEvo's evolutionary algorithm to infer a port mapping from a list of experiments.
The list of experiments needs to be generated beforehand by running `scripts/gen_experiments.py` on the system under investigation.
This inference script performs no more microbenchmarks and can run on any system (multicore CPUs are recommended).

The C++ implementation of the evolutionary algorithm is expected to be built in the `lib/cpp-evolution` directory of the repository.
"""


import argparse
import json
from pathlib import Path
import os
import sys
import textwrap

import_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.append(import_path)

from pmtestbench.common.experiments import ExperimentList
from pmtestbench.common.synthesizers import Synthesizer

from iwho.utils import parse_args_with_logging


PMEVO_PATH = Path(__file__).parent.parent / "lib" / "cpp-evolution" / "build" / "last_build"

def main():
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('-o', '--output', metavar='OUTFILE', default=None, help='path to file to write the resulting mapping to')

    argparser.add_argument('-n', '--num-ports', metavar='N', type=int, default=4,
        help='number of ports to assume in the inferred mapping')

    argparser.add_argument('-e', '--epsilon', metavar='EPSILON', type=float, default=0.05,
        help='epsilon for equivalence checking')

    argparser.add_argument('-c', '--config', metavar='CONFIG', default=None,
        help='path to the PMEvo configuration file')

    argparser.add_argument('--tmp-dir', metavar='DIR', default='/tmp/',
        help='temporary directory to use for logging and temporary files')

    argparser.add_argument('--pmevo-bin-path', metavar='PATH', default=PMEVO_PATH,
        help='path to the PMEvo binary')

    argparser.add_argument('input', metavar="INFILE",
        help='input file with evaluated experiments')

    args = parse_args_with_logging(argparser, "info")



    num_ports = args.num_ports
    epsilon = args.epsilon
    pmevo_bin_path = Path(args.pmevo_bin_path)
    pmevo_config_path = args.config

    # directory to use for logging and temporary files
    pmevo_tmp_dir = args.tmp_dir

    with open(args.input, 'r') as f:
        elist = ExperimentList.from_json(f)

    arch = elist.arch

    print(f"Inferring a port mapping from {len(elist)} experiments.")

    synth_config = {
        "synthesizer_kind": "partition",
        "equivalence_epsilon": epsilon,
        "wrapped_config": {
            "synthesizer_kind": "pmevo",
            "mapping_class": "Mapping3",
            "num_ports": num_ports,

            "pmevo_bin_path": pmevo_bin_path,
            "pmevo_config_path": pmevo_config_path,
            "pmevo_temp_directory": pmevo_tmp_dir,
        }
    }

    synth = Synthesizer(config=synth_config)

    m = synth.infer(elist)

    if m is None:
        print("Failed to infer mapping!")
        return 1

    if args.output is not None:
        with open(args.output, "w") as f:
            m.to_json(f)
    print("Found mapping:")
    print(textwrap.indent(m.to_json_str(), '  '))

    return 0


if __name__ == "__main__":
    sys.exit(main())
