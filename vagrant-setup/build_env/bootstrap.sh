#!/usr/bin/env bash

set -ex

apt-get update
apt-get install -y vim tmux htop nano
apt-get install -y python3 python3-pip python3-dev python3-venv
apt-get install -y llvm-16 clang-16

ln -s /bin/llvm-mc-16 /bin/llvm-mc
ln -s /bin/clang-16 /bin/clang
ln -s /bin/clang++-16 /bin/clang++

apt-get install -y git ninja-build libelf-dev libffi-dev cmake

apt-get install -y texlive

VAGRANT_HOME=/home/vagrant/

# copy everything to the homefolder so that it is persistent independent of the
# run folder.
cp -r /vagrant/* $VAGRANT_HOME
cp -r /vagrant/.inputrc $VAGRANT_HOME
cd $VAGRANT_HOME


# add a sensible vim config
mkdir -p .vim/pack/tpope/start
cd .vim/pack/tpope/start
git clone https://tpope.io/vim/sensible.git
cd $VAGRANT_HOME


# set up the python environment
cd pmtestbench

./setup_venv.sh

source ./env/pmtestbench/bin/activate

# add copyhost command
echo "function copyhost { cp -r \"\$@\" /vagrant/; }" >> $VAGRANT_HOME/.bashrc

# make the python environment available to the user by default
echo "source /home/vagrant/pmtestbench/env/pmtestbench/bin/activate" >> $VAGRANT_HOME/.bashrc

cd $VAGRANT_HOME

cd ./pmtestbench/lib/cppfastproc
make
cd $VAGRANT_HOME

cd ./pmtestbench/lib/cpp-evolution
# use conservative (and slow) build settings for compatibility reasons
CFG=no_avx make
cd $VAGRANT_HOME

# make everything owned by the default user instead of root
chown -R vagrant:vagrant /home/vagrant/
