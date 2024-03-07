#!/usr/bin/env python3

""" Export a port mapping inferred with infer-relaxeduops.py into a range of formats.
"""

import argparse
import copy
from datetime import datetime
from collections import Counter, defaultdict
import json
from pathlib import Path
import pickle
import subprocess

import os
import sys

import_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.append(import_path)

from pmtestbench.common.portmapping import Mapping, Mapping3
from pmtestbench.relaxeduops.utils import translate_port
from iwho.configurable import load_json_config, pretty_print
from iwho import Config
import iwho.x86 as x86

import logging
logger = logging.getLogger(__name__)


def punorm(list_of_uops):
    if list_of_uops is None:
        return frozenset()
    c = Counter()
    for uop in list_of_uops:
        c[frozenset(uop)] += 1
    return frozenset([(k, v) for k, v in c.items()])

default_preamble_str = r"""
\title{Inferred Port Mapping}
\maketitle
"""

latex_template = r"""
\documentclass[a4paper,english,fontsize=9]{scrartcl}

\usepackage[top=1.5cm, bottom=2.5cm, left=1.8cm, right=1.8cm]{geometry}

\usepackage[main=english]{babel}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{enumitem}
\usepackage{hyperref}
\usepackage{multirow}

\usepackage{array}
\newcolumntype{L}[1]{>{\raggedright\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}
\newcolumntype{C}[1]{>{\centering\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}
\newcolumntype{R}[1]{>{\raggedleft\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}

\newcommand{\tinyspace}{ \kern 0.08em }

\begin{document}

\date{$DATE -- Version $VERSION}

$PREAMBLE


\begin{longtable}{p{.54\textwidth}ccL{.25\textwidth}}
  \toprule
  \textbf{Instruction Scheme} & \(\mathit{tp}^{-1}\) & \textbf{\#\(\mu\)ops} & \textbf{Port Usage}\\
  \toprule
  \endhead
  $TABLE
  \bottomrule
\end{longtable}

\end{document}
"""

def format_port_usage(port_usage, translate_map=None):
    if port_usage is None:
        return ""
    res_uops = Counter()
    for uop in port_usage:
        res_ports = []
        for port in uop:
            res_ports.append(str(translate_port(port, translate_map, force_int=False)))
        res_uops[tuple(sorted(res_ports))] += 1

    formatted_uops = [f"{count}\\(\\times\\)[{','.join(uop)}]" for uop, count in res_uops.items()]
    # formatted_uops = [f"\\({count}\\tinyspace\\times\\tinyspace[{','.join(uop)}]\\)" for uop, count in res_uops.items()]
    return " + ".join(formatted_uops)


def generate_latex(mapping, insns_with_measurements, date_str, version_str, preamble_str, translate_map=None, varying_report=None, uop_adjustments_for_insn={}, iwho_ctx=None):
    measurement_dict = {}
    for iwm in insns_with_measurements:
        measurement_dict[iwm.insn] = iwm

    lines = []
    for insn, port_usage in mapping.assignment.items():
        md = measurement_dict[insn]
        itp = md.cycles
        num_uops = md.num_uops

        varying = False
        if varying_report is not None:
            report_entry = varying_report.get(insn, None)
            if report_entry is not None:
                varying = report_entry['varying']

        if varying:
            wrong_port_usage_marker = ""
            pretty_port_usage = "\\emph{unstable}"
        else:
            wrong_port_usage_marker = ""
            if port_usage is not None and len(port_usage) != num_uops and len(port_usage) != 0:
                wrong_port_usage_marker = "*"
            pretty_port_usage = format_port_usage(port_usage, translate_map)

        uop_adjustment = uop_adjustments_for_insn.get(insn, 0)
        if uop_adjustment > 0:
            base_num_uops = num_uops - uop_adjustment
            # pretty_num_uops = "\\({}\\tinyspace+\\tinyspace{}\\)".format(base_num_uops, uop_adjustment)
            pretty_num_uops = "{}+{}".format(base_num_uops, uop_adjustment)
        else:
            pretty_num_uops = str(num_uops)

        insn_str = insn
        if iwho_ctx is not None:
            features = iwho_ctx.get_features(insn)
            if features is not None:
                # uopsinfo_url = ifeatures[0]['uops_info_url']
                ref_url = features[0]['ref_url']
                insn_str = "\\href{{https://{}}}{{{}}}".format(ref_url, insn_str)

        lines.append("  \\texttt{{{}}} & {:.2f} & {}{} & {} \\\\\n".format(insn_str, itp, pretty_num_uops, wrong_port_usage_marker, pretty_port_usage))

    table = "  \\midrule\n".join(lines)
    table = table.replace("_", "\\_")
    table = table.replace("al,bl,bpl,cl,dil,dl,r10b,r11b,r12b,r13b,r14b,r15b,r8b,r9b,sil,spl", "GPR:8")
    table = table.replace("XMM0..15", "XMM")
    table = table.replace("YMM0..15", "YMM")

    return latex_template.replace("$TABLE", table).replace("$DATE", date_str).replace("$VERSION", version_str).replace("$PREAMBLE", preamble_str)

def find_insns_with_adjusted_uop_count(measurement_log_file):
    if not measurement_log_file.exists():
        return dict()

    uop_adjustments_for_insn = dict()
    with open(measurement_log_file, 'r') as f:
        measurement_log = json.load(f)

    for (date, exp, measurement) in measurement_log:
        if len(exp) != 1:
            continue
        insn = exp[0]
        num_uops_increased_by = measurement.get('num_uops_increased_by', 0)
        if num_uops_increased_by > 0:
            uop_adjustments_for_insn[insn] = num_uops_increased_by

    return uop_adjustments_for_insn


def generate_jsondict(mapping, metadata_dict, insns_with_measurements, date_str, version_str, varying_report=None, uop_adjustments_for_insn={}, iwho_ctx=None):
    # Example format:
    # {
    #     "uarchname": "AMD Zen+",
    #     "version": "0.1.0",
    #     "date": "...",
    #     "iwhoctx": "x86",
    #     "peakipc": 5.0,
    #     "ports": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #     "data": {
    #         "add": {
    #             "portusage": [[[6, 7, 8, 9], 1]] ,
    #             "itp": 0.25,
    #             "numuops": 1,
    #             "numuops_unadjusted": 1
    #         }
    #     }
    # }
    res_dict = copy.deepcopy(metadata_dict)
    res_dict['version'] = version_str
    res_dict['date'] = date_str

    all_ports = set()
    data = dict()

    measurement_dict = {}
    for iwm in insns_with_measurements:
        measurement_dict[iwm.insn] = iwm

    for insn, port_usage in mapping.assignment.items():
        md = measurement_dict[insn]
        itp = md.cycles
        num_uops = md.num_uops

        if port_usage is None:
            formatted_port_usage = []
            varying = True
        else:
            c = Counter()
            for uop in port_usage:
                all_ports.update(uop)
                c[frozenset(uop)] += 1
            formatted_port_usage = [(sorted(k), v) for k, v in c.items()]

        entry = {
                "portusage": formatted_port_usage,
                "itp": itp,
                "numuops": num_uops,
            }

        varying = False
        if varying_report is not None:
            report_entry = varying_report.get(insn, None)
            if report_entry is not None:
                varying = report_entry['varying']
        if varying:
            entry['varying'] = True

        uop_adjustment = uop_adjustments_for_insn.get(insn, 0)
        if uop_adjustment > 0:
            base_num_uops = num_uops - uop_adjustment
            entry['numuops_unadjusted'] = base_num_uops
        else:
            pretty_num_uops = str(num_uops)
            entry['numuops_unadjusted'] = num_uops

        insn_str = insn
        if iwho_ctx is not None:
            features = iwho_ctx.get_features(insn)
            if features is not None:
                # uopsinfo_url = ifeatures[0]['uops_info_url']
                ref_url = features[0]['ref_url']
                entry['ref_url'] = ref_url

        data[insn_str] = entry

    res_dict['ports'] = list(sorted(all_ports))
    res_dict['data'] = data
    return res_dict

class InsnGroup:
    keys = ["regular_base", "regular_read", "regular_write", "ymm_base", "ymm_read", "ymm_write"]
    def __init__(self, mnemonic, base_width):
        self.mnemonic = mnemonic
        self.base_width = base_width
        self.regular_base = [] # only register operands, width N, N <= 128
        self.regular_read = [] # with a read memory operand, width N
        self.regular_write = [] # with a written memory operand, width N
        self.ymm_base = [] # only register operands, width 2*N = 256
        self.ymm_read = [] # with a read memory operand, width 2*N = 256
        self.ymm_write = [] # with a written memory operand, width 2*N = 256

    def __repr__(self):
        res = f"InsnGroup({self.mnemonic}, {self.base_width}):"
        for x in self.keys:
            ls = getattr(self, x)
            if len(ls) > 0:
                ls_str = "; ".join(map(str, ls))
                res += f"\n  {x}: {ls_str}"
        return res

    def is_trivial(self):
        entries = 0
        for x in self.keys:
            entries += min(1, len(getattr(self, x)))
        return entries <= 1

    def get_schemes(self):
        all_schemes = []
        for x in self.keys:
            all_schemes += getattr(self, x)
        return all_schemes



def find_normal_insns(mapping, insns_with_measurements, iwho_ctx):
    """ Based on the regular "normal" case described in the paper
    """

    # cluster by mnemonic
    ischemes_by_mnemonic = defaultdict(list)

    for insn_str in mapping.assignment.keys():
        ischeme = iwho_ctx.str_to_scheme[insn_str]
        mnemonic = iwho_ctx.extract_mnemonic(ischeme)
        ischemes_by_mnemonic[mnemonic].append(ischeme)

    logger.info("Found {} unique mnemonics".format(len(ischemes_by_mnemonic)))

    insn_groups = dict()
    def get_insn_group(mnemonic, base_width):
        res = insn_groups.get((mnemonic, base_width), None)
        if res is None:
            res = InsnGroup(mnemonic, base_width)
            insn_groups[(mnemonic, base_width)] = res
        return res

    def has_mem(op_keys, require_write=False):
        for k, op_scheme in op_keys:
            if op_scheme.is_fixed():
                if isinstance(op_scheme.fixed_operand, x86.MemoryOperand):
                    if not require_write:
                        return True
                    elif op_scheme.is_written:
                        return True
            else:
                if isinstance(op_scheme.operand_constraint, x86.MemConstraint):
                    if not require_write:
                        return True
                    elif op_scheme.is_written:
                        return True
        return False

    for mnemonic, ischemes in ischemes_by_mnemonic.items():
        for i in ischemes:
            op_keys = i.operand_keys
            if len(op_keys) == 0:
                # this one is obviously not normal
                continue
            op_scheme = op_keys[0][1]
            if op_scheme.is_fixed():
                width = op_scheme.fixed_operand.width
            else:
                width = op_scheme.operand_constraint.width

            if width == 256:
                base_width = 128
            else:
                base_width = width

            insn_group = get_insn_group(mnemonic, base_width)

            if width == 256:
                kind = "ymm"
            else:
                kind = "regular"

            if has_mem(op_keys, require_write=True):
                kind += '_write'
            elif has_mem(op_keys, require_write=False):
                kind += '_read'
            else:
                kind += '_base'

            ls = getattr(insn_group, kind)
            ls.append(i)

    normal = []
    violations = []
    ungrouped = []
    for k, v in insn_groups.items():
        if v.is_trivial():
            ungrouped += v.get_schemes()
            continue

        # print(repr(v))
        not_normal_reason = insn_group_is_normal(v, mapping)
        if not_normal_reason is not None:
            print(f"not normal, {not_normal_reason}: ", end='')
            print(repr(v))
            violations += v.get_schemes()
        else:
            normal += v.get_schemes()
    print(f"normal: {len(normal)}")
    print(f"violations: {len(violations)}")
    print(f"ungrouped: {len(ungrouped)}")
    return normal, violations, ungrouped



def insn_group_is_normal(insn_group, mapping):
    ref_entries = {}
    base_width = insn_group.base_width
    for x in ["regular_base", "regular_read", "regular_write", "ymm_base", "ymm_read", "ymm_write"]:
        entry = getattr(insn_group, x)
        if len(entry) >= 1:
            ref_pu = punorm(mapping.assignment[str(entry[0])])
            if len(ref_pu) == 0:
                return f"port usage for {x} ischemes is empty"
            for other in entry[1:]:
                other_pu = punorm(mapping.assignment[str(other)])
                if ref_pu != other_pu:
                    return f"entries for {x} ischemes are not uniform ({ref_pu} vs {other_pu})"
            ref_entries[x] = ref_pu

    if reg_base := ref_entries.get("regular_base"):
        if reg_read := ref_entries.get("regular_read"):
            if reg_read != reg_base | punorm([[4, 5]]):
                return f"regular_base ({reg_base}) and regular_read ({reg_read}) are not normal"
        if reg_write := ref_entries.get("regular_write"):
            if base_width >= 32:
                if reg_write != reg_base | punorm([[5]]):
                    return f"regular_base ({reg_base}) and regular_write ({reg_write}) are not normal for {base_width} bits"
            else:
                if reg_write != reg_base | punorm([[4, 5], [5]]):
                    return f"regular_base ({reg_base}) and regular_write ({reg_write}) are not normal for {base_width} bits"

        if ymm_base := ref_entries.get("ymm_base"):
            cmp_val = frozenset([ (k, 2*v) for k, v in reg_base])
            if ymm_base != cmp_val:
                return f"ymm_base ({ymm_base}) is not regular_base ({reg_base}) doubled"

    if ymm_base := ref_entries.get("ymm_base"):
        if ymm_read := ref_entries.get("ymm_read"):
            if ymm_read != ymm_base | punorm([[4, 5], [4, 5]]):
                return f"ymm_base ({ymm_base}) and ymm_read ({ymm_read}) are not normal"

    return None



def compute_category_split(mapping, insns_with_measurements, uop_adjustments_for_insn, iwho_ctx):
    res = {
            'no_result': [],
            'ucoded': [],
            'normal': [],
            'other': [],
        }

    normal, violations, ungrouped = find_normal_insns(mapping, insns_with_measurements, iwho_ctx)
    normal_set = set(map(str, normal))


    for iwm in insns_with_measurements:
        insn = iwm.insn
        port_usage = mapping.assignment[insn]

        uop_adjustment = uop_adjustments_for_insn.get(insn, 0)
        unadjusted_uops = iwm.num_uops - uop_adjustment

        if port_usage is None or len(port_usage) == 0:
            res['no_result'].append(insn)
        elif insn in normal_set:
            res['normal'].append(insn)
        elif unadjusted_uops > 2:
            res['ucoded'].append(insn)
        else:
            res['other'].append(insn)

    return res


def main():
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('-d', '--dest', metavar="DEST", default="./export",
        help='the output directory')

    argparser.add_argument('-m', '--mode', metavar="MODE", default="json",
        help='the output mode, "json", "latex", or "categories"', choices=['json', 'latex', 'categories'])

    argparser.add_argument('--metadata', metavar="FILE", default=None,
        help='json file with metadata to put into generated json mappings')

    argparser.add_argument('-c', '--iwho-config', metavar="CONFIG", default=None,
        help='an iwho config in json format, for adding metadata to the output')

    argparser.add_argument('-v', '--version', metavar="X.Y", default="0.0",
        help='the version string to use in the output')

    argparser.add_argument('-p', '--preamble', metavar="FILE", default=None,
        help='a file containing preamble text to use for latex output')


    # argparser.add_argument('input', metavar="INFILE",
    #     help='the input port mapping')

    argparser.add_argument('inputs', metavar="INPUTDIR", nargs='+',
        help='a result directory of a relaxed algorithm run')


    # unsupported options:
    # argparser.add_argument('-t', '--translate', metavar="FILE", default=None,
    #     help='if given, translate the ports of the port mapping using the given json file, which needs to contain a dict mapping old port numbers to new port names')
    # argparser.add_argument('--varying-report', metavar="FILE", default=None,
    #     help='if a json report produced by the check_variance script is given, mark instructions with varying results')


    args = argparser.parse_args()

    dest_dir = Path(args.dest).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    translate_map = None
    varying_report = None

    all_insns = set()
    all_mappings = []
    all_insns_with_measurements = []
    all_uop_adjustments = [] # those should really always be the same

    for curr_input in args.inputs:
        indir = Path(curr_input).resolve()

        checkpoint_file = indir / "checkpoint_09_check_singleton_throughputs_in_full_mapping.pickle"

        with open(checkpoint_file, 'rb') as f:
            cp = pickle.load(f)

        stage = cp['stage']

        assert (stage in ['compute_full_mapping', 'check_singleton_throughputs_in_full_mapping']), "checkpoint is not from the last stage"


        curr_mapping = cp['stored_data']['mapping']

        curr_insns_with_measurements = cp['stored_data']['insns_with_measurements']


        measurement_log_file = indir / "measurement_log.json"
        curr_uop_adjustments_for_insn = find_insns_with_adjusted_uop_count(measurement_log_file)

        translation_table_file = indir / "translation_table.json"

        # translate_map = None
        if translation_table_file.exists():
            assert False, "translation table not supported"
        #     with open(translation_table_file, 'r') as f:
        #         translate_map = json.load(f)
        all_mappings.append(curr_mapping)
        all_insns_with_measurements.append({ iwm.insn: iwm for iwm in curr_insns_with_measurements})
        all_uop_adjustments.append(curr_uop_adjustments_for_insn)

        all_insns.update(curr_mapping.assignment.keys())


    mapping = Mapping3(all_mappings[0].arch)
    insns_with_measurements = []
    uop_adjustments_for_insn = {}

    default_insns_with_measurements = {}
    default_uop_adjustments_for_insn = {}
    for a, b in zip(all_insns_with_measurements[::-1], all_uop_adjustments[::-1]):
        default_insns_with_measurements.update(a)
        default_uop_adjustments_for_insn.update(b)

    assert len(all_mappings) <= 3, "the implemented check for variance does not make much sense for more than three port mappings"

    num_varying = 0
    for insn in all_insns:
        c = Counter()
        for m in all_mappings:
            c[punorm(m.assignment[insn])] += 1

        if len(c.keys()) > 2:
            # As it is, the entries of the considered port mappings are not
            # considered problematic if there are at most two different
            # results, and if there are two, the more common one is used. This
            # is probably not what one wants if more than three port mappings
            # are considered.
            mapping.assignment[insn] = None
            insns_with_measurements.append(default_insns_with_measurements[insn])
            if insn in default_uop_adjustments_for_insn:
                uop_adjustments_for_insn[insn] = default_uop_adjustments_for_insn[insn]
            num_varying += 1
            logger.warning(f"Instruction {insn} has different port usage in different mappings: {c}")
            continue
        chosen_pu = c.most_common(1)[0][0]
        for x, m in enumerate(all_mappings):
            entry = m.assignment[insn]
            if punorm(entry) == chosen_pu:
                mapping.assignment[insn] = entry
                insns_with_measurements.append(all_insns_with_measurements[x][insn])
                if insn in all_uop_adjustments[x]:
                    uop_adjustments_for_insn[insn] = all_uop_adjustments[x][insn]
                break

    # varying_report = None
    # if args.varying_report is not None:
    #     with open(args.varying_report, 'r') as f:
    #         varying_report = json.load(f)


    iwho_ctx = None
    if args.iwho_config is not None:
        iwho_config = load_json_config(args.iwho_config)
        iwho_ctx = Config(iwho_config).context

    metadata = {}
    if args.metadata is not None:
        with open(args.metadata, 'r') as f:
            metadata = json.load(f)

    # with open(args.input, 'r') as f:
    #     mapping = Mapping.read_from_json(f)

    date_str = datetime.now().strftime("%Y-%m-%d")
    version_str = args.version

    if args.preamble is not None:
        with open(args.preamble, 'r') as f:
            preamble_str = f.read()
    else:
        preamble_str = default_preamble_str



    if args.mode == "json":
        logger.info("Generating JSON output")
        json_dict = generate_jsondict(mapping, metadata, insns_with_measurements, date_str, version_str,
                                      varying_report=varying_report,
                                      uop_adjustments_for_insn=uop_adjustments_for_insn,
                                      iwho_ctx=iwho_ctx)
        with open(dest_dir / "port_mapping.json", 'w') as f:
            f.write(pretty_print(json_dict))

    elif args.mode == "latex":
        logger.info("Generating LaTeX code")
        latex_code = generate_latex(mapping, insns_with_measurements, date_str, version_str, preamble_str,
                                    translate_map=translate_map,
                                    varying_report=varying_report,
                                    uop_adjustments_for_insn=uop_adjustments_for_insn,
                                    iwho_ctx=iwho_ctx)

        with open(dest_dir / "port_mapping.tex", 'w') as f:
            f.write(latex_code)

        logger.info("Compiling PDF from LaTeX code")
        subprocess.run(["latexmk", "-pdf", "port_mapping.tex"], cwd=dest_dir)
    elif args.mode == "categories":
        category_split = compute_category_split(mapping, insns_with_measurements, uop_adjustments_for_insn=uop_adjustments_for_insn, iwho_ctx=iwho_ctx)
        print("categories:")
        for k, v in category_split.items():
            print(f"  {k}: {len(v)}")

            with open(dest_dir / f"category_{k}.txt", 'w') as f:
                for i in v:
                    print(i, file=f)

    else:
        raise ValueError("invalid mode")


    return 0

if __name__ == "__main__":
    sys.exit(main())
