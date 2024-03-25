# pmtestbench - Port Mapping Testbench

A set of tools to infer port mappings of out-of-order CPUs from throughput microbenchmarks.
It also serves as the Artifact for the ASPLOS paper "Explainable Port Mapping Inference with Sparse Performance Counters for AMD's Zen Architectures".

This artifact is also available as a Vagrant/VirtualBox virtual machine image at [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10794887).

## Maturity
This is a research prototype, expect things to break!

## Installation

** These steps are not required when using the prebuilt virtual machine. **

Make sure that you have `llvm-mc` on your path (most likely by installing [LLVM](https://llvm.org/)).
It is used to handle basic instruction (dis)assembly tasks.
Furthermore, you need a python setup with at least version 3.10 and with the `venv` standard module available.

1. Get the repository and its submodule(s):
    ```
    git clone <repo> pmtestbench
    cd pmtestbench
    git submodule update --init
    ```
2. Set up a virtual environment and install python dependencies and the
   pmtestbench package itself there:
   ```
   ./setup_venv.sh
   ```
   Whenever you run commands from a shell, you need to have activated the
   virtual environment in this shell before:
   ```
   source ./env/pmtestbench/bin/activate
   ```
3. Run the tests:
   ```
   ./tests/run_tests.sh
   ```

If you want to use the faster native port mapping throughput prediction algorithm, you need to build the corresponding package:
```
cd lib/cppfastproc
make
```

If you want to infer port mappings with PMEvo, you need to build the PMEvo binary (requires g++ or clang++ and Make):
```
cd lib/cpp-evolution
CFG=<config> make
```
where `<config>` is the name (without extension) of a config file in `lib/cpp-evolution/build_configs` you want to use.
If your processor supports the x86 AVX2 ISA Extension (i.e. it is a relatively recent x86 processor by Intel or AMD), we recommend `fastest` (if the `gold` linker is available) or `omp`.
If your processor does not support AVX2, we recommend `omp_no_avx`.


## Usage

We discuss several ways to use this artifact in the following sections.

### Inspecting Results from the Paper

All configurations for and results of the experiments in the paper are available in the `data/zenp` directory:
- `data/zenp/relaxeduops/results/port_mapping.pdf` contains the inferred port mapping for AMD's Zen+ microarchitecture in a human-readable form.
- `data/zenp/relaxeduops/results/port_mapping.json` contains the inferred port mapping for AMD's Zen+ microarchitecture in a machine-readable form.
- Logs of the algorithm runs that produced this inferred port mapping can be found in `data/zenp/relaxeduops/results/individual_results/run*/report.log`.
- `data/zenp/pmevo/inferred_mapping.json` contains the port mapping that we inferred with PMEvo for the quantitative comparison.
- `data/zenp/palmed/palmed_zen_extracted_2021-11-25.json` contains the conjunctive resource mapping that we extracted from the available Palmed implementation for the quantitative comparison.
- `data/zenp/eval_experiments_annotated.json` contains raw data for the heatmaps in Figure 5 of the paper.

### Interesting Code Locations

- The core port mapping inference algorithm of the paper is implemented  in `src/pmtestbench/relaxeduops/core.py`.
- The SMT-solver-based counter-example-guided inference algorithm is implemented in `src/pmtestbench/cegpmi/smt_synthesizer.py`.
  The high-level algorithm is located in the `synthesize` method of the `SMTSynthesizer` class, the low-level SMT encodings are implemented in the methods of the `ConstrainedMapping3Handler`.
- The microbenchmarking mechanism is implemented in
    * `src/pmtestbench/common/processors/iwho_processor.py` - the code for generating dependency-free instruction sequences for experiments
    * `lib/iwho/iwho/predictors/pite_frame.c` - the program template that is instantiated with the benchmarked instruction sequence, compiled, and executed to measure the throughput
    * `lib/iwho/iwho/predictors/pite_predictor.py` - the code that instantiates, compiles, and runs the above program template
- The PMEvo port mapping inference algorithm is implemented in `lib/cpp-evolution` and `src/pmtestbench/pmevo/`.

### Reproducing the Heatmaps from the Paper
The following steps reproduce the heatmaps from Figure 5 of the paper:
1. Generate throughput predictions for the evaluation experiments with the inferred port mapping:
    ```
    ./scripts/annotate_predictions.py -i rep-relaxeduops ./data/zenp/relaxeduops/exported_pm_processor.json ./data/zenp/eval_experiments_clean.json
    ```
2. Generate throughput predictions for the evaluation experiments with the port mapping inferred with PMEvo:
    ```
    ./scripts/annotate_predictions.py -i rep-pmevo ./data/zenp/pmevo/inferred_mapping.json ./data/zenp/eval_experiments_clean_01.json
    ```
3. Generate throughput predictions for the evaluation experiments with the extracted Palmed resource mapping:
    ```
    ./scripts/annotate_predictions.py -i rep-palmed ./data/zenp/palmed/palmed_processor.json ./data/zenp/eval_experiments_clean_02.json
    ```
4. Generate the heatmaps:
    ```
    ./scripts/eval_annotated_predictions.py -m IPC --mode heatmap ./data/zenp/eval_experiments_clean_03.json
    ```
5. Find the results as
    ```
    data/zenp/eval_experiments_clean_03_heatmap_IPC_rep-*.png
    ```

### Displaying the Port Usage of an Instruction Sequence

You can use the `scripts/analyze-bb.py` script to display the port usage and the estimated inverse throughput of an instruction sequence based on an inferred port mapping, e.g.:
```
./scripts/analyze-bb.py -f -a ./tests/resources/test.s ./data/zenp/relaxeduops/results/port_mapping.json
```

### Inferring a Port Mapping...

Reproducing this step requires access to the microarchitecture under
investigation (Zen+ in the paper) and the ability to run exact microbenchmarks
on it. This might require considerable effort to ensure stable measurements and
is not recommended for the casual reader.

Adjusting the configuration to other microarchitectures can be expected to
require even more fine-tuning effort.

Both algorithms are run with small artificial inputs as part of the tests,
specifically in `tests/integration-relaxeduops.py` and in `tests/integration-pmevo.py`.

#### ...with the Relaxed Inference algorithm

Given a proper setup, the following command could be used to apply the relaxeduops inference algorithm as described in the case study in the paper:
```
./scripts/infer-relaxeduops.py \
    --output result.json \
    --preferred-blockinginsns ./data/zenp/relaxeduops/config/preferred_blocking_insns_zenp.txt \
    --add-improper-blockinginsns ./data/zenp/relaxeduops/config/improper_blockinginsns.txt \
    --preferred-mapping ./data/zenp/relaxeduops/config/preferred_mapping_proper.json \
    --params ./data/zenp/relaxeduops/results/individual_results/run01/parameters.json \
    ./data/zenp/relaxeduops/config/pite_config.json
```

The resulting port mapping could be exported in human-readable form as follows:
```
./scripts/relaxed_uops_export_portmapping.py -m latex \
    --metadata ./data/zenp/relaxeduops/config/metadata.json \
    -c ./data/zenp/iwho_config-spec17.json \
    relaxed_uops_reports/<directory generated by the previous script>
```


#### ...with PMEvo


Given a proper setup, the following command could be used to generate and benchmark experiments for the PMEvo inference algorithm:

```
./scripts/gen_experiments.py full-pmevo \
            --epsilon 0.05 \
            --output pmevo_exps.json \
            ./data/zenp/relaxeduops/config/pite_config.json
```

Then, the evolutionary algorithm could be run on the generated experiments:
```
./scripts/infer-pmevo.py \
    --output pmevo_mapping.json \
    --num-ports N \
    --epsilon 0.05 \
    --config ./data/zenp/pmevo/pmevo_run.cfg \
    pmevo_exps.json
```

