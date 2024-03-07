#! /usr/bin/env python3

""" Run the counter-example-guided port mapping inference algorithm on its own.
For practical applications, this is most likely prohibtively slow.
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
from pmtestbench.common.processors import Processor

from iwho.utils import parse_args_with_logging
from iwho.configurable import load_json_config



def main():
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('-o', '--output', metavar='OUTFILE', default=None, help='path to file to write the resulting mapping to')

    argparser.add_argument('-p', '--processor', metavar='PROC', required=True,
        help='Path to a processor config in json format. JSONified port mappings can also be used directly as processor configs.')

    argparser.add_argument('-s', '--synthesizer', metavar='SYNTH', required=True,
        help='Path to a synthesizer config in json format.')

    args = parse_args_with_logging(argparser, "info")

    proc_config = load_json_config(args.processor)
    proc = Processor(config=proc_config)

    synth_config = load_json_config(args.synthesizer)

    synth_kind = synth_config.get("synthesizer_kind", None)
    if synth_kind != "smt":
        logger.warning(f"unexpected synthesizer kind specified in config: {synth_kind}")

    synth = Synthesizer(config=synth_config)

    m = synth.synthesize(proc)

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


