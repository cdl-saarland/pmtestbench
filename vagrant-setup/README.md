# Setting up the VM

This directory contains scripts and configurations to package an artifact
provided as a Vagrant Virtual Machine using the Virtual Box provider. To run
it, you will need to install Vagrant as described here:

    https://www.vagrantup.com/intro/getting-started/install.html

and also Virtual Box as available here:

    https://www.virtualbox.org/

Depending on your hardware platform, it might be necessary to enable
virtualization options in your system's BIOS to use virtual box.

Use the following command to build the VM:
```
cd build_env
vagrant up
```

Once this is done, ssh to the VM and test it:
```
vagrant ssh
cd pmtestbench
./tests/run_tests.sh
```

Next, leave the VM and package it up:
```
exit
vagrant package --output ../pmtestbench.box
```

Take the resulting box and put it into an archive together with the shipping
vagrant file:
```
cd ..
tar -czvf pmtestbench-artifact.tgz Vagrantfile artifact_usage.md pmtestbench.box
```

Done!

