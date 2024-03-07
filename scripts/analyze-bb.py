#! /usr/bin/env python3

""" Display port mapping information and a throughput estimation for a given
basic block.

Requires an exported port mapping from the relaxed uops algorithm.
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
from pmtestbench.common.portmapping import Mapping3
from pmtestbench.common.architecture import Architecture

from iwho.utils import parse_args_with_logging
import iwho

import logging
logger = logging.getLogger(__name__)

def format_port_usage(port_usage):
    return " + ".join([f"{num} * {pset}" for pset, num in port_usage])

def main():
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument("-x", "--hex", metavar="HEXSTR", default=None, help="decode instructions from the bytes represented by the specified hex string")

    argparser.add_argument("-a", "--asm", metavar="ASMSTR", default=None, help="analyze instructions from the specified asm string")

    argparser.add_argument("-f", "--file", action="store_true", help="allow passing a file path as the argument for --hex or --asm")

    argparser.add_argument("portmapping", metavar="PORTMAPPING", help="path to a port mapping file")

    args = parse_args_with_logging(argparser, "info")


    logger.debug(f"Loading port mapping from {args.portmapping}")

    with open(args.portmapping, "r") as f:
        portmapping = json.load(f)

    print_fields = [ "uarchname", "peakipc", "version", "date", ]
    pm_info_str = "Port Mapping Information:"
    for field in print_fields:
        pm_info_str += f"\n  {field}: {portmapping[field]}"
    logger.debug(pm_info_str)

    iwhoctx_id = portmapping["iwhoctx"]
    peakipc = portmapping["peakipc"]


    logger.debug(f"Loading iwho context '{iwhoctx_id}'")
    ctx = iwho.get_context_by_name(iwhoctx_id)

    logger.debug(f"Parsing/decoding instructions")
    if args.hex is not None:
        if args.file:
            with open(args.hex, "r") as f:
                hexstr = f.read()
        else:
            hexstr = args.hex
        insns = ctx.decode_insns(hexstr)
    elif args.asm is not None:
        if args.file:
            with open(args.asm, "r") as f:
                asmstr = f.read()
        else:
            asmstr = args.asm
        insns = ctx.parse_asm(asmstr)
    else:
        raise ValueError("No input provided")

    ischeme_lines = [f"  {ii}  # scheme: {ii.scheme}" for ii in insns]
    ischeme_str = textwrap.indent("\n".join(ischeme_lines), "  ")

    logger.debug(f"found {len(insns)} instruction(s):\n{ischeme_str}")

    pmdata = portmapping["data"]

    num_not_found = 0

    max_insn_str = max(map(lambda x: len(str(x)), insns))
    column_width = max(max_insn_str, len("Instruction"))

    # create an Architecture from the used instructions
    arch = Architecture()
    ischeme_strs = set(map(lambda x: str(x.scheme), insns))
    arch.add_insns(ischeme_strs)

    mapping3 = Mapping3(arch)

    print("Port usage per instruction:")
    # print(f"{'Instruction'.ljust(column_width)}  |  Port Usage")

    for insn in insns:
        key = str(insn.scheme)
        pmdata_entry = pmdata.get(key, None)
        if pmdata_entry is not None:
            port_usage = pmdata_entry["portusage"]
            port_usage_str = format_port_usage(port_usage)

            if len(mapping3.assignment[key]) == 0:
                for pset, num in port_usage:
                    for x in range(num):
                        mapping3.assignment[key].append(pset)
        else:
            port_usage_str = "N/A"
            num_not_found += 1
        print(f"{str(insn).ljust(column_width)}  |  {port_usage_str}")

    if num_not_found > 0:
        logger.debug(f"{num_not_found} instruction(s) not found in port mapping data")

    # use a SimProcessor to get the simulated inverse throughput
    proc = Processor(mapping3)

    cycles = proc.get_cycles(list(map(lambda x: str(x.scheme), insns)))

    print("")
    print("Inverse throughput according to the port mapping: {:.2f} cycles/iteration".format(cycles))

    min_cycles = len(insns) / peakipc
    adjusted_cycles = max(min_cycles, cycles)
    print("Inverse throughput adjusted for the peak IPC:     {:.2f} cycles/iteration".format(adjusted_cycles))

    return 0

if __name__ == "__main__":
    sys.exit(main())

