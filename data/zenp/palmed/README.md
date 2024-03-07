# Extracted Resource Mapping of Palmed


The file `palmed_zen_extracted_2021-11-25.json` contains the Zen resource mapping in Palmed's conjunctive model that we extracted from `scw-zen.2021-11-25.mapping` in [Palmed's available source code](https://gitlab.inria.fr/nderumig/palmed/-/tree/master/results/data?ref_type=heads).
It contains a mapping from their instruction identifiers to identifiers of representatives with the same resource usage (`instr_to_class`) and a mapping from these representatives to the resources they use (`class_to_resources`).

The file `fixed_palmed_insn_map.json` contains a mapping of our instruction scheme identifiers to matching Palmed instruction identifiers. This mapping was created in a partly manual and partly automated process.

This mapping has been used in the evaluation of our paper "Explainable Port Mapping Inference with Sparse Performance Counters for AMD’s Zen Architectures".

The original data has been produced by Derumigny et al. for the paper "PALMED: throughput characterization for superscalar architectures" (2022).

