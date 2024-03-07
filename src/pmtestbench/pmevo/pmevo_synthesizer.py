""" A wrapper to implement the Synthesizer interface using the C++ PMEvo
development.
"""

from collections import defaultdict
import json
import os
import sys
import textwrap
from subprocess import Popen, PIPE
from pathlib import Path

from ..common.synthesizers import SynthesizerImpl
from ..common.experiments import ExperimentList
from ..common.portmapping import Mapping


import logging
logger = logging.getLogger(__name__)

def export_explist(elist):
    indent = " " * 4
    result = ""
    result += "architecture:\n"
    result += indent + "instructions:\n"
    arch = elist.arch
    for i in arch.insn_list:
        result += indent * 2 + "\"{}\"\n".format(i)
    result += "\n"
    for e in elist:
        result += "experiment:\n"
        result += indent + "instructions:\n"
        for i in e.iseq:
            result += indent * 2 + "\"{}\"\n".format(i)
        result += indent + "cycles: {}\n".format(e.get_cycles())
        result += "\n"
    return result


def export_mapping(mapping):
    # TODO untested
    indent = " " * 4
    result = "mapping:\n"
    for i, uops in mapping.assignment.items():
        result += "\"{}\":\n".format(i)
        uop_map = defaultdict(lambda:0)
        for u in uops:
            rep = ""
            for p in sorted(u, key=lambda x: int(x)):
                num = int(p)
                assert 0 <= num and num < 26
                rep += chr(ord("A") + num)
            uop_map[rep] += 1

        for u, n in uop_map.items():
            if n <= 0:
                continue
            result += indent + "{}: {}\n".format(u, n)
        result += "\n"
    return result



class PMEvoSynthesizer(SynthesizerImpl):

    def __init__(self, config):
        self.config = config
        self.bin_path = Path(config["pmevo_bin_path"])
        self.config_path = config["pmevo_config_path"]
        self.tmp_dir = Path(config["pmevo_temp_directory"])
        self.journal_path = self.tmp_dir / "pmtestbench_tmp_evo_journal.log"
        self.num_ports = config["num_ports"]

        self.cmd = [
                self.bin_path,            # call the PMEvo binary
                "-i",                     # read experiments from stdin
                "-j",                     # print the resulting mapping as json to stdout
                "-n1",                    # print only the best mapping
                f"-x{self.journal_path}", # do some logging
                "-q{}".format(self.num_ports),
            ]

        if self.config_path is not None:
            self.cmd.append(f"-c{self.config_path}") # use the specified config

    def infer(self, exps):
        singleton_exps = ExperimentList(exps.arch)
        singleton_exps.exps = [e for e in exps if len(e.iseq) == 1]
        singleton_elist_path = self.tmp_dir / "pmtestbench_tmp_singleton.exps"
        singleton_elist_str = export_explist(singleton_exps)
        with open(singleton_elist_path, "w") as ef:
            print(singleton_elist_str, file=ef)

        # transform exps into string
        expstr = export_explist(exps)

        # start binary with config and exps
        cmd = self.cmd + ["-e{}".format(singleton_elist_path)]
        p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        outs, errs = p.communicate(input=expstr.encode('utf-8'))
        out_str = outs.decode('utf-8')
        err_str = errs.decode('utf-8')
        retval = p.returncode

        # logger.debug("PMEvo output on stdout:\n" + textwrap.indent(out_str, "  "))
        logger.debug("PMEvo output on stderr:\n" + textwrap.indent(err_str, "  "))

        if retval != 0:
            logger.error(
                f"PMEvo binary returned with non-zero return code: {retval}!\n" +
                "  Command: {}".format(" ".join(map(str, cmd))) +
                "\n  output on stdout:\n" + textwrap.indent(out_str, "    ") +
                "\n  output on stderr:\n" + textwrap.indent(err_str, "  "))
            return None

        # read mapping
        mapping = Mapping.read_from_json_str(out_str, arch=exps.arch)

        return mapping

    def synthesize(self, proc, exps):
        raise RuntimeError("Synthesis is not supported by this synthesizer!")


