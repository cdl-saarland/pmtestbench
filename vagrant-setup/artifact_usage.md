# Artifact Usage

by Fabian Ritter and Sebastian Hack, Saarland University

## Getting Started Guide

This artifact is a Vagrant VM with our development and all dependencies
installed.

### Requirements

You will need an x86-64 processor to evaluate this artifact.
ARM Processors, like in recent Mac(Book)s, are not supported by the VM.

The artifact was only tested on 64-bit Linux systems.

### Setup

The artifact is provided as a Vagrant Virtual Machine using the Virtual Box
provider. To run it, you will need to install Vagrant as described here:

> https://www.vagrantup.com/intro/getting-started/install.html

and also Virtual Box as available here:

> https://www.virtualbox.org/

Depending on your hardware platform, it might be necessary to enable
virtualization options in your system's BIOS to use virtual box.

In the provided archive, you will find the `pmtestbench.box` Vagrant box and a
`Vagrantfile` for configuring your instance of the virtual machine. It comes
with defaults for the number of CPU cores and the available RAM that can be
modified as required.
Once you have adjusted the settings to your liking and installed Vagrant and
Virtual Box, use the following commands to make the box known to Vagrant, to
start the VM, and to establish an SSH connection into the VM:

```
vagrant box add --name pmtestbench pmtestbench.box
vagrant up
vagrant ssh
```

The VM is based on a 64-bit stock Debian 12.4 (bookworm) image. Additionally to
common tools, we pre-installed tmux, vim, and nano for convenience. By default,
Vagrant mounts the directory that includes the Vagrantfile in the host system
as a shared directory in the VM, at `/vagrant/`. You can use this directory to
move files into or out of the VM. Inside the VM, the `copyhost FILE*` command
can be used to copy files to this shared directory.

The VM can be shut down from the host via `vagrant halt`. Changes to the file
system are persistent, just run `vagrant up` and `vagrant ssh` again to boot
the VM and connect to it again. Once it is no longer used, the VM can be
deleted from vagrant with `vagrant destroy`. The pmtestbench base box can then
be removed with `vagrant box remove pmtestbench`.


### Testing the Setup

To quickly verify that all components of the artifact are working, run the
following commands in the VM via the ssh connection in the home folder of the
default user `vagrant`:

```
cd ~/pmtestbench
./tests/run_tests.sh
```

These tests exercise the major components of the artifact and should run
without error.


## Artifact Contents

The `pmtestbench` directory in the home folder of the `vagrant` user contains
all code and data of the artifact, it is a clone of our public Github
repository.
`pmtestbench/README.md` describes its content in more detail.
The installation steps listed there have already been performed in the VM.

This artifact includes all code and configurations necessary to reproduce the
results of the paper. It further includes the raw results of the case study and
the evaluation described in the paper.

We suggest the following steps to evaluate the artifact:
- Inspect results and code locations listed in the sections "Inspecting Results
  from the Paper" and "Interesting Code Locations" in `pmtestbench/README.md`.
- Explore the inferred port mapping for a specific basic blocks as described in
  the "Displaying the Port Usage of an Instruction Sequence" section in
  `pmtestbench/README.md`.
- Follow the steps in the "Reproducing the Heatmaps from the Paper" section to
  reproduce the heatmaps in Figure 5 of the paper. Afterwards, you can use the
  following command to copy the generated heatmaps to the host system for
  visual inspection and compare them to the heatmaps in the paper:
  ```copyhost data/zenp/eval_experiments_clean_03_heatmap_IPC_rep-*.png```

We specifically recommend against running the actual port mapping inference
algorithm. While the necessary code is included in the artifact for the benefit
of interested future researchers, running the algorithm requires access to a
specific microarchitecture (Zen+), extensive preparation to ensure stable
measurements, and a significant amount of time.

The inference algorithm is exercised for a very small artificial example in the
`tests/integration-relaxeduops.py` test that runs as part of the `run_tests.sh`
script.


