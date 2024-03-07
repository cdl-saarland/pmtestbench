#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

VM_BASE=$SCRIPT_DIR/build_env

cd $VM_BASE
rm -rf ./pmtestbench
git clone https://github.com/cdl-saarland/pmtestbench.git pmtestbench

cd $VM_BASE/pmtestbench
git submodule update --init

