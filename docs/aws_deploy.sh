#!/usr/bin/env bash

# Tested on Amazon Linux 2, but should work on most RPM-based Linux distros

sudo yum -y install git gcc

git clone https://github.com/ExcitedStates/qfit-3.0.git
cd qfit-3.0/
pip install -r requirements.txt

# Optionally, uncomment the following line to set a specific version of qFit
#git checkout v3.1.2
pip install .
