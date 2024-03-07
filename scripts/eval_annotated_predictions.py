#!/usr/bin/env python3

""" Compute statistics and generate plots for annotated experiment lists.

The input are annotated experiment lists, which are generated and evaluated via
    `./scripts/gen_experiments.py make-validation`
and
    `./scripts/gen_experiments.py eval-validation`
and then annotated with throughput predictions via one or more runs of
    `./scripts/annotate_predictions.py`
"""

import argparse
import copy
from collections import Counter, namedtuple
import math
from pathlib import Path

import os
import sys

from numpy import mean, median
from scipy.stats import pearsonr, spearmanr, kendalltau

import matplotlib as mpl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.append(import_path)

from pmtestbench.common.experiments import ExperimentList

# uncomment this for better looking plots
# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r"""
# \usepackage{amsmath}
# \usepackage[tt=false, type1=true]{libertine}
# \usepackage[varqu]{zi4}
# \usepackage[libertine]{newtxmath}
# """)


def make_heatmaps(ref_list, other_results, metric, base_outfile_on=None):
    # metric may be overwritten later on

    # general purpose configuration
    config_flexible = {
            'filetype': 'png',
            'add_text': True,
            'filter_max_cycles': None,
            'num_bins': 30,
            'custom_ticks': False,
            'ylim': None,
            'xlim': None,
            'stepsize': None,
            'minor_split': None,
            'tick_fmt': '.0f',
            'ysize': 2.5,
            'xsize': 3.2,
            'cmap': "Blues",
            'metric': None,
            'export_dpi': 300,
        }

    # configuration for the paper heatmaps
    config_paper = {
            'filetype': 'pdf',
            'add_text': False,
            'filter_max_cycles': None,
            'num_bins': 27,
            'custom_ticks': True,
            'ylim': 9,
            'xlim': 6,
            'stepsize': 1,
            'minor_split': 1,
            'tick_fmt': '.0f',
            'ysize': 2.1,
            'xsize': 1.9,
            'cmap': "Blues",
            'vmax': 1200,
            'metric': None,
        }

    config = config_flexible
    # config = config_paper


    filetype = config['filetype']
    add_text = config['add_text']
    filter_max_cycles = config['filter_max_cycles']
    num_bins = config['num_bins']
    custom_ticks = config['custom_ticks']
    ylim = config['ylim']
    xlim = config['xlim']
    stepsize = config['stepsize']
    minor_split = config['minor_split']
    tick_fmt = config['tick_fmt']
    ysize = config['ysize']
    xsize = config['xsize']
    cmap = config['cmap']
    vmin = config.get('vmin', None)
    vmax = config.get('vmax', None)
    square = config.get('square', None)
    cbar_ticks = config.get('cbar_ticks', None)
    export_dpi = config.get('export_dpi', None)

    if config['metric'] is not None:
        metric = config['metric']


    for identifier, predictions_list in other_results.items():
        print("statistics for '{}':".format(identifier))

        rel_errors = []

        min_sim = math.inf
        max_sim = -math.inf

        min_ref = math.inf
        max_ref = -math.inf

        filtered_ref_list = []
        filtered_predictions_list = []

        for ref, sim in zip(ref_list, predictions_list):
            ref = round(ref * 100) / 100
            sim = round(sim * 100) / 100
            if filter_max_cycles is not None and (ref > filter_max_cycles):
                continue
            filtered_ref_list.append(ref)
            filtered_predictions_list.append(sim)

            min_sim = min(min_sim, sim)
            max_sim = max(max_sim, sim)

            min_ref = min(min_ref, ref)
            max_ref = max(max_ref, ref)

            rel_error = abs(sim - ref) / ref
            rel_errors.append(rel_error)
        min_error = min(rel_errors)
        max_error = max(rel_errors)

        print(f"after filtering, {len(filtered_ref_list)} out of {len(ref_list)} experiments remain")

        if not custom_ticks:
            ylim = max(max_sim, max_ref)
            xlim = ylim

        am_error = mean(rel_errors)
        pearson_corr, pearson_p = pearsonr(filtered_ref_list, filtered_predictions_list)
        spearman_corr, spearman_p = spearmanr(filtered_ref_list, filtered_predictions_list)
        kendall_tau, kendall_p = kendalltau(filtered_ref_list, filtered_predictions_list)


        stats_str = ""
        stats_str += "  am_error:      {}\n".format(am_error)
        stats_str += "  pearsonr:      {} (p: {})\n".format(pearson_corr, pearson_p)
        stats_str += "  spearmanr:     {} (p: {})\n".format(spearman_corr, spearman_p)
        stats_str += "  kendalltau:    {} (p: {})\n".format(kendall_tau, kendall_p)
        stats_str += "  -----------\n"
        stats_str += "  min_sim:    {}\n".format(min_sim)
        stats_str += "  max_sim:    {}\n".format(max_sim)
        stats_str += "  min_ref:    {}\n".format(min_ref)
        stats_str += "  max_ref:    {}\n".format(max_ref)
        stats_str += "  min_error:  {}\n".format(min_error)
        stats_str += "  max_error:  {}\n".format(max_error)
        print(stats_str)

        fig, ax = plt.subplots(figsize=(xsize,ysize), subplot_kw={ "aspect": "equal" })

        ax.plot([0, min(xlim, ylim)], [0, min(xlim, ylim)], linewidth=0.8, color="orange", rasterized=False)

        if square is not None:
            ax.plot([square, square], [0, square], linewidth=0.8, color="orange", rasterized=False)
            ax.plot([0, square], [square, square], linewidth=0.8, color="orange", rasterized=False)

        h, xedges, yedges, im = ax.hist2d(filtered_ref_list, filtered_predictions_list,
                                          bins=(round(num_bins * xlim / ylim ), num_bins),
                                          range=((0,xlim), (0, ylim)),
                                          cmap=cmap,
                                          norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), rasterized=False)
        max_val = np.max(h)
        print(f"max_val: {max_val}")


        if custom_ticks:
            for axis in [ax.xaxis, ax.yaxis]:
                axis.set_major_locator(ticker.MultipleLocator(stepsize))
                axis.set_major_formatter('{x:' + tick_fmt + '}')
                axis.set_minor_locator(ticker.MultipleLocator(stepsize / minor_split))

        toolid = ""

        cb = fig.colorbar(im)
        cb.ax.tick_params(which="both", width = 0.4)
        if cbar_ticks is not None:
            cb.set_ticks(cbar_ticks)
        # cb.set_ticklabels(np.linspace(vmin, vmax, num_ticks))


        ax.set_xlabel(f'measured {metric}')
        ax.set_ylabel(f'predicted {metric}{toolid}')


        if add_text:
            annotation_str = "MAPE: ${:.2f}\\%$".format(am_error * 100)
            annotation_str += "\n$\\rho_P$: ${:.2f}$".format(pearson_corr)
            annotation_str += "\n$\\tau_k$: ${:.2f}$".format(kendall_tau)

            # Textbox
            props = dict(boxstyle='round', facecolor='orange', alpha=0.3)
            ax.text(0.05, 0.95, annotation_str, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=props, horizontalalignment='left')
            # props = dict(boxstyle='round', facecolor='orange', alpha=0.3)
            # ax.text(0.42, 0.96, annotation_str, transform=ax.transAxes, fontsize=9,
            #         verticalalignment='top', bbox=props, horizontalalignment='right')

        fig.tight_layout()

        outfile = f"heatmap_{metric}_{identifier}.{filetype}"
        if base_outfile_on is not None:
            base_outfile_on_path = Path(base_outfile_on).resolve()
            outfile = base_outfile_on_path.parent / (base_outfile_on.stem + "_" + outfile)

        kwargs = {}
        if export_dpi is not None:
            kwargs['dpi'] = export_dpi
        plt.savefig(outfile, **kwargs)



ErrListEntry = namedtuple("ErrListEntry", ["error", "experiment", "ref_val", "pred_val"])

def compute_errors_per_experiment(elist, chosen_identifier, metric):
    error_for_exp = []
    assert chosen_identifier is not None

    for e in elist:
        num_insns = len(e.iseq)
        ref_val = extract_metric(e.get_cycles(), num_insns, metric)
        for r in e.other_results:
            identifier = r["result_id"]
            if identifier == chosen_identifier:
                pred_val = extract_metric(r["cycles"], num_insns, metric)
                break
        else:
            raise ValueError("no result for '{}'".format(chosen_identifier))
        error = abs(pred_val - ref_val) / ref_val
        error_for_exp.append(ErrListEntry(error, e, ref_val, pred_val))

    return error_for_exp

def find_likely_culprits(elist, chosen_identifier, metric, topx=0.3):
    error_for_exp = compute_errors_per_experiment(elist, chosen_identifier, metric)

    sorted_error_for_exp = sorted(error_for_exp, key=lambda x: x[0], reverse=True)

    split_point = int(topx * len(sorted_error_for_exp))
    c_bad = Counter()
    for error, e, ref_val, pred_val in sorted_error_for_exp[:split_point]:
        c_bad.update(e.iseq)

    c_good = Counter()
    for error, e, ref_val, pred_val in sorted_error_for_exp[split_point:]:
        c_good.update(e.iseq)

    c_diff = c_bad - c_good

    print(f"likely culprits:")
    for insn, count in c_diff.most_common():
        if count <= 1:
            break
        print("  {}, score: {} ({} bad occurrences, {} good occurrences)".format(insn, count, c_bad[insn], c_good[insn]))


def make_error_list(elist, chosen_identifier, metric, threshold=0.1, ref_cutoff=3.0):
    error_for_exp = compute_errors_per_experiment(elist, chosen_identifier, metric)

    filtered_because_of_cutoff = []
    filtered_because_of_threshold = []

    filtered_error_for_exp = []
    for ele in error_for_exp:
        if ele.ref_val > ref_cutoff:
            filtered_because_of_cutoff.append(ele)
            continue
        filtered_error_for_exp.append(ele)


    sorted_error_for_exp = sorted(filtered_error_for_exp, key=lambda x: x.error, reverse=True)

    num_printed = 0
    print("worst predictions:")
    for ele in sorted_error_for_exp:
        error, e, ref_val, pred_val = ele
        if error <= threshold:
            break
        print("  error: {} ({} vs. {})\n    {}".format(error, ref_val, pred_val, e))
        filtered_because_of_threshold.append(ele)
        num_printed += 1

    print("showing {} of {} experiments, limited by threshold".format(num_printed, len(sorted_error_for_exp)))

    print("blacklist:")
    for ls, description in [(filtered_because_of_cutoff, "cutoff"), (filtered_because_of_threshold, "threshold")]:
        for error, e, ref_val, pred_val in ls:
            exp_str = "; ".join(map(str, e.iseq))
            print(f"{exp_str}  # {description} with error {error:.2f} (ref: {ref_val:.2f}, pred: {pred_val:.2f})")

    print(f"cutoff: {len(filtered_because_of_cutoff)}")
    print(f"threshold: {len(filtered_because_of_threshold)}")
    print(f"total: {len(filtered_because_of_cutoff) + len(filtered_because_of_threshold)}")



def extract_metric(cycles, num_insns, metric):
    cycles = float(cycles)
    if cycles <= 0:
        return -1.0
    match (metric):
        case 'IPC':
            return num_insns / cycles
        case 'CPI':
            return cycles / num_insns
        case 'cycles':
            return cycles
        case _:
            raise ValueError("unknown metric '{}'".format(metric))

def main():
    argparser = argparse.ArgumentParser(description=__doc__)

    # argparser.add_argument('-o', '--output', metavar="OUTFILE", default=None,
    #     help='the output file')

    argparser.add_argument('-m', '--metric', metavar='METRIC', default='IPC',
                           choices = ['IPC', 'CPI', 'cycles'],
        help="the metric to use (one of 'IPC', 'CPI', 'cycles'; default: 'IPC')")

    argparser.add_argument('--mode', metavar='MODE', default='heatmap', choices = ['heatmap', 'errhist', 'listerrs', 'findculprits'],
        help="the evaluation mode to use (one of 'heatmap', 'listerrs', 'findculprits'; default: 'heatmap')")

    argparser.add_argument('-i', '--identifier', metavar='ID', default=None,
        help="use only the results with the given identifier (default: use all)")

    argparser.add_argument('input', metavar="INFILE", nargs='+',
        help='the input experiment list(s), in json format')

    args = argparser.parse_args()

    for infile in args.input:
        inpath = Path(infile).resolve()
        print("processing '{}'".format(inpath))

        with open(infile, 'r') as f:
            elist = ExperimentList.from_json(f)

        metric = args.metric

        chosen_identifier = args.identifier

        if args.mode == 'listerrs':
            threshold = 0.1
            ref_cutoff = 3.0
            make_error_list(elist, chosen_identifier, metric, threshold=threshold, ref_cutoff=ref_cutoff)
            continue

        if args.mode == 'findculprits':
            topx = 0.1
            find_likely_culprits(elist, chosen_identifier, metric, topx)
            continue

        ref_list = []

        other_results = dict()

        for e in elist:
            num_insns = len(e.iseq)

            ref_val = extract_metric(e.get_cycles(), num_insns, metric)
            ref_list.append(ref_val)
            for r in e.other_results:
                identifier = r["result_id"]
                if chosen_identifier is not None and identifier != chosen_identifier:
                    continue
                other_val = extract_metric(r["cycles"], num_insns, metric)
                dest = other_results.get(identifier, [])
                dest.append(other_val)
                other_results[identifier] = dest

        match (args.mode):
            case 'heatmap':
                make_heatmaps(ref_list, other_results, metric, base_outfile_on=inpath)
            case _:
                raise ValueError("unknown mode '{}'".format(args.mode))

    return 0

if __name__ == "__main__":
    sys.exit(main())
