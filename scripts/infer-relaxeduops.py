#!/usr/bin/env python3

""" A script to find a port mapping using an adjusted version of the uops.info
algorithm that does not require per-port uop counters.
"""

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import json
import pickle
import random

import os
import sys

from iwho.utils import init_logging
from iwho.configurable import load_json_config, pretty_print


import_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.append(import_path)

from pmtestbench.common.processors import Processor
from pmtestbench.common.portmapping import Mapping

import pmtestbench.relaxeduops.core as core
from pmtestbench.relaxeduops.utils import AlgoParameters

import logging
logger = logging.getLogger(__name__)


def main():
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('-o', '--output', metavar="OUTFILE", required=True,
        help='the output file')

    argparser.add_argument('-s', '--start-with', metavar="CHECKPOINTFILE", default=None,
        help='a checkpoint file of a previous run')

    argparser.add_argument('--preferred-blockinginsns', metavar="FILE", default=None,
        help='a file containing a newline-separated list of insn schemes that should be used as representative blocking instructions')

    argparser.add_argument('--add-improper-blockinginsns', metavar="FILE", default=None,
        help='a file containing a newline-separated list of insn schemes that should be considered as improper blocking instructions,' +
                           ' i.e., they use more than one uop, but we need to use them anyway because one of those uops does not have a proper blocking insn')

    argparser.add_argument('--preferred-mapping', metavar="FILE", default=None,
        help='a json file containing a two-level port mapping that should be used for the blocking instructions (once it is validated against the experiments that occur in the SMT algo)')

    argparser.add_argument('--discard-blockinginsns', metavar="FILE", default=None,
        help='a file containing a newline-separated list of insn schemes that should be discarded before the SMT phase')

    argparser.add_argument('--restrict-insns', metavar="FILE", default=None,
        help='a file containing a newline-separated list of insn schemes to which the final stage of the algorithm should be restricted')

    argparser.add_argument('-t', '--single-step', action="store_true",
        help='only run a single checkpoint (shines when combined with --start-with)')

    argparser.add_argument('--steps', metavar="N", type=int, default=None,
        help='only run N checkpoints')

    argparser.add_argument('-c', '--check-pairs', action="store_true",
        help='do not attempt full inference, just check whether singleton and pair results are consistent')

    argparser.add_argument('-b', '--relax-bounds', action="store_true",
        help="slightly increase the bounds on throughputs that lead to rejecting an instruction, so that we don't get spurious rejects for instructions right at the border.")

    argparser.add_argument('-r', '--report-dir', metavar="DIR", default="relaxed_uops_reports",
        help='a directory to put reports of how the algorithm ran')

    argparser.add_argument('--random-seed', metavar="SEED", type=int, default=424242,
        help='a seed for the random number generator')

    argparser.add_argument('-p', '--params', metavar="PARAMS", default=None,
        help='a json file containing the parameters for the algorithm')

    argparser.add_argument('proc_config', metavar="PROCESSOR",
        help='a processor config in json format (or a json-dumped port mapping)')

    args = argparser.parse_args()

    random.seed(args.random_seed)

    # checkpoint mechanism:
    # When, during a run of this script, a stage is completed, the results so
    # far are stored (pickled) into a checkpoint file representing the progress
    # so far.
    # By calling this script with --start-with and such a checkpoint file, the
    # script can continue with with the state represented by this file. This
    # may obviously break if this script functionaly changes significantly.

    # create a directory to put in all kinds of reports
    report_dir = Path(args.report_dir).resolve()
    timestamp = datetime.now().replace(microsecond=0).isoformat()
    report_dir = report_dir / f'relaxed_uops_{timestamp}'
    os.makedirs(report_dir)

    # loglevel = 'info'
    loglevel = 'debug'
    init_logging(loglevel=loglevel, logfile=report_dir / 'report.log')

    proc_config = load_json_config(args.proc_config)

    with open(report_dir / 'proc_config.json', 'w') as f:
        f.write(pretty_print(proc_config))

    if args.params is not None:
        with open(args.params, 'r') as f:
            params = AlgoParameters.from_json_file(f)
    else:
        logger.warning("No parameters file given, using defaults")
        params = AlgoParameters()

    if args.relax_bounds:
        params.insn_skip_cycles += 2.0

    if args.start_with is not None:
        with open(args.start_with, 'rb') as f:
            checkpoint = pickle.load(f)
        # Without those, we are in danger of running meaningless experiments,
        # with mismatching configs for different stages.
        assert checkpoint['proc_config'] == proc_config
        assert checkpoint['params'] == params
        start_with_checkpoint = checkpoint
    else:
        start_with_checkpoint = None

    if args.preferred_blockinginsns is not None:
        with open(args.preferred_blockinginsns, 'r') as f:
            preferred_blockinginsns = []
            for l in f:
                l = l.strip()
                if len(l) == 0:
                    continue
                preferred_blockinginsns.append(l)
    else:
        preferred_blockinginsns = None

    if args.add_improper_blockinginsns is not None:
        with open(args.add_improper_blockinginsns, 'r') as f:
            added_improper_blockinginsns = []
            for l in f:
                l = l.strip()
                if len(l) == 0:
                    continue
                added_improper_blockinginsns.append(l)
    else:
        added_improper_blockinginsns = None

    if args.preferred_mapping is not None:
        with open(args.preferred_mapping, 'r') as f:
            preferred_mapping = Mapping.read_from_json(f)

            # that's a hack
            f.seek(0)
            json_data = json.load(f)
            secondary_uops = json_data.get("metadata", {}).get("secondary_uops", {})
            preferred_mapping.secondary_uops = secondary_uops
    else:
        preferred_mapping = None

    if args.discard_blockinginsns is not None:
        with open(args.discard_blockinginsns, 'r') as f:
            discard_blockinginsns = []
            for l in f:
                l = l.strip()
                if len(l) == 0:
                    continue
                discard_blockinginsns.append(l)
    else:
        discard_blockinginsns = None

    if args.restrict_insns is not None:
        with open(args.restrict_insns, 'r') as f:
            restrict_insns = set()
            for l in f:
                l = l.strip()
                if len(l) == 0:
                    continue
                restrict_insns.add(l)
    else:
        restrict_insns = None

    if args.single_step:
        num_checkpoints = 1
    else:
        num_checkpoints = args.steps

    proc = Processor(proc_config, enable_result_logging=True)

    arch = proc.get_arch()
    insns = set(arch.insn_list)

    def check_insnlist(insnlist, name):
        invalid_insns = set(insnlist) - insns
        if len(invalid_insns) > 0:
            raise ValueError(f"{name} {invalid_insns} are not part of the architecture")

    check_insnlist(preferred_blockinginsns or [], 'preferred blocking insns')
    check_insnlist(discard_blockinginsns or [], 'discarded blocking insns')
    check_insnlist(added_improper_blockinginsns or [], 'improper blocking insns')

    with open(report_dir / 'parameters.json', 'w') as f:
        f.write(pretty_print(asdict(params)))

    checkpoint_base = {
            'proc_config': proc_config,
            'params': params,
        }

    check_pairs = args.check_pairs

    rc = 0
    try:
        res = core.find_mapping(params, proc, log_dir=report_dir,
                           start_with_checkpoint=start_with_checkpoint,
                           num_checkpoints=num_checkpoints,
                           checkpoint_base=checkpoint_base,
                           check_pairs=check_pairs,
                           preferred_blockinginsns=preferred_blockinginsns,
                           discard_blockinginsns=discard_blockinginsns,
                           preferred_mapping=preferred_mapping,
                           added_improper_blockinginsns=added_improper_blockinginsns,
                           restrict_insns=restrict_insns,
                           )
        rejects = res.rejects

        if res.has_key('mapping'):
            mapping = res.mapping
            with open(args.output, 'w') as f:
                f.write(mapping.to_json_str())
            with open(report_dir / 'result_mapping.json', 'w') as f:
                f.write(mapping.to_json_str())
    except Exception as e:
        logger.exception("Terminated with an exception!")
        rc = 1

    if rc == 0 and len(rejects) != 0:
        logger.warning(f"Rejected {len(rejects)} instructions, see rejects.json for details")
        with open(report_dir / 'rejects.json', 'w') as f:
            f.write(pretty_print(rejects))

    with open(report_dir / 'measurement_log.json', 'w') as f:
        # TODO it might make sense to dump these at intermediate points
        f.write(pretty_print(proc.result_log))

    return rc

if __name__ == "__main__":
    sys.exit(main())
